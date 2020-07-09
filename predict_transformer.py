import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from prepare import preprocess_sentence
from train_transformer import Transformer, create_masks
import model_params


def evaluate(sentence, inp_lang, targ_lang,
             transformer,
             max_length_inp, max_length_targ):
    
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    encoder_input = tf.convert_to_tensor(inputs)

    result = ''

    output = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    for i in range(max_length_targ):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if targ_lang.index_word[predicted_id.numpy()[0][0]] == '<end>':
            return result, sentence, attention_weights

        result += targ_lang.index_word[predicted_id.numpy()[0][0]] + ' '

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return result, sentence, attention_weights


# function for plotting the attention weights
def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = sentence.split(' ')
    result = result.split(' ')

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result)-1.5, -0.5)

        ax.set_xticklabels(
            sentence, 
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels(result, 
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head+1))

    plt.tight_layout()
    plt.show()


def translate_transformer(sentence, config, plot=''):

    with open(config['inp_tokenizer_file'], 'rb') as handle:
        inp_lang = pickle.load(handle)
    with open(config['targ_tokenizer_file'], 'rb') as handle:
        targ_lang = pickle.load(handle)

    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1

    transformer = Transformer(config['num_layers'], config['d_model'], config['num_heads'], config['dff'],
                              vocab_inp_size, vocab_tar_size, 
                              pe_input=vocab_inp_size, 
                              pe_target=vocab_tar_size,
                              rate=config['dropout_rate'])
    
    checkpoint = tf.train.Checkpoint(transformer=transformer)
    
    checkpoint.restore(tf.train.latest_checkpoint(config['model_checkpoint_dir'])).expect_partial()
    
    result, sentence, attention_weights = evaluate(sentence, inp_lang, targ_lang, transformer, config['prediction_max_length_inp'], config['prediction_max_length_targ'])

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


def main():

    def translate_(sent):
        return translate_transformer(sent, config=model_params.TRANSFORMER_MODEL_PARAMS)

    translate_(u'ти згоден')
    translate_(u'Том весь вечір дивиться телевізор.')
    translate_(u'Я щойно згадав, що мені треба щось зробити.')
