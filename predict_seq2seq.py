import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from prepare import preprocess_sentence
from train_seq2seq import Encoder, Decoder
import model_params


def evaluate(sentence, inp_lang, targ_lang,
             encoder, decoder,
             max_length_inp, max_length_targ,
             units):
    
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate_seq2seq(sentence, config):

    with open(config['inp_tokenizer_file'], 'rb') as handle:
        inp_lang = pickle.load(handle)
    with open(config['targ_tokenizer_file'], 'rb') as handle:
        targ_lang = pickle.load(handle)

    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1

    encoder = Encoder(vocab_inp_size, config['embedding_dim'], config['units'], config['batch_size'])
    decoder = Decoder(vocab_tar_size, config['embedding_dim'], config['units'], config['batch_size'])
    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(config['model_checkpoint_dir']))
    result, sentence, attention_plot = evaluate(sentence, inp_lang, targ_lang, encoder, decoder, 
                                                config['prediction_max_length_inp'], config['prediction_max_length_targ'], config['units'])

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


def main():

    def translate_(sent):
        return translate_seq2seq(sent, config=model_params.SEQ2SEQ_MODEL_PARAMS)

    translate_(u'ти згоден')
    translate_(u'Том весь вечір дивиться телевізор.')
    translate_(u'Я щойно згадав, що мені треба щось зробити.')
