import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from prepare import load_dataset, get_anki_dataset, get_target_dataset
import pickle


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights
    

def train_seq2seq(path_to_file_anki, path_to_file_target, num_examples=None,
                  direction='ua-eng', test_size=0.0,
                  model_checkpoint_dir='./seq2seq_checkpoints',
                  inp_tokenizer_file='input_tokenizer', targ_tokenizer_file='target_tokenizer', 
                  epochs=10, batch_size=64, embedding_dim=256, units=1024):
    
    en_tensors, ua_tensors, en_lang_tokenizer, ua_lang_tokenizer = load_dataset(path_to_file_anki, path_to_file_target, num_examples=num_examples)
    
    # Choose translation direction
    if direction=='ua-eng':
        input_tensor, inp_lang = ua_tensors, ua_lang_tokenizer
        target_tensor, targ_lang = en_tensors, en_lang_tokenizer
    elif direction=='eng-ua':
        input_tensor, inp_lang = en_tensors, en_lang_tokenizer
        target_tensor, targ_lang = ua_tensors, ua_lang_tokenizer
    else:
        pass
    
    # Save tokenizers
    with open(inp_tokenizer_file, 'wb') as handle:
        pickle.dump(inp_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(targ_tokenizer_file, 'wb') as handle:
        pickle.dump(targ_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=test_size)
    # Show length
    print("Dataset size")
    print("Train")
    print(len(input_tensor_train), len(target_tensor_train))
    print("Validation")
    print(len(input_tensor_val), len(target_tensor_val))
    
    # Calculate max_length of the target tensors
    max_length_inp, max_length_targ = input_tensor.shape[1], target_tensor.shape[1]
    print("Max input size")
    print(max_length_inp, max_length_targ)
    
    # Show dictionary size
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1
    print("Vocabulary size")
    print(vocab_inp_size, vocab_tar_size)
    
    buffer_size = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train)//batch_size

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
    
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = loss_object(real, pred)
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
      return tf.reduce_mean(loss_)
    
    checkpoint_prefix = os.path.join(model_checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    
    @tf.function
    def train_step(inp, targ, enc_hidden):
      loss = 0

      with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
          # passing enc_output to the decoder
          predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
          loss += loss_function(targ[:, t], predictions)
          # using teacher forcing
          dec_input = tf.expand_dims(targ[:, t], 1)

      batch_loss = (loss / int(targ.shape[1]))
      variables = encoder.trainable_variables + decoder.trainable_variables
      gradients = tape.gradient(loss, variables)
      optimizer.apply_gradients(zip(gradients, variables))

      return batch_loss
    
    print(" Run training")
    for epoch in range(epochs):
      start = time.time()

      enc_hidden = encoder.initialize_hidden_state()
      total_loss = 0

      for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / steps_per_epoch))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    
def main():
    
    path_to_file_anki = get_anki_dataset('ukr-eng.zip')
    path_to_file_target = get_target_dataset('ua-target-201704.zip')
    
    train_seq2seq(path_to_file_anki, path_to_file_target, num_examples=None,
                  direction='ua-eng', test_size=0.0,
                  model_checkpoint_dir='./seq2seq_checkpoints',
                  inp_tokenizer_file = 'input_tokenizer', targ_tokenizer_file = 'target_tokenizer', 
                  epochs=10, batch_size=64, embedding_dim=256, units=1024)
