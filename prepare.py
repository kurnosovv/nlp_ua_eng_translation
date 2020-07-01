import unicodedata
import re
import numpy as np
import os
import io
import zipfile
import tensorflow as tf


def get_anki_dataset(path_to_zip):
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall()
    path_to_file_anki = os.getcwd()+os.path.dirname(path_to_zip)+"/ukr-eng/ukr.txt"
    return path_to_file_anki


def get_target_dataset(path_to_zip):
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall()
    
    files_all = os.listdir(os.getcwd()+'/UA-TARGET')
    files_en = [file for file in files_all if file.endswith('en-u8.ph')]
    files_ua = [file.replace('en-u8.ph', 'ua-u8.ph') for file in files_en]

    path_to_files_en = [os.getcwd()+os.path.dirname(path_to_zip)+'/UA-TARGET/'+file for file in files_en]
    path_to_files_ua = [os.getcwd()+os.path.dirname(path_to_zip)+'/UA-TARGET/'+file for file in files_ua]
    path_to_file_target = zip(path_to_files_en, path_to_files_ua)
    return path_to_file_target


# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Zа-яА-Яіїєй?.!,']+", " ", w)

  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w


def create_dataset_anki(path, num_examples=None):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]]  for l in lines[:num_examples]]
  return zip(*word_pairs)


def create_dataset_uatarget(path_list, num_examples=None):
  lines_en, lines_ua = list(), list()
  for path_file_en, path_file_ua in path_list:
    print(path_file_en)
    new_lines_en = io.open(path_file_en, encoding='UTF-8').read().strip().split('\n')
    lines_en.extend(new_lines_en)
    new_lines_ua = io.open(path_file_ua, encoding='UTF-8').read().strip().split('\n')
    lines_ua.extend(new_lines_ua)
    print(len(new_lines_en), len(new_lines_ua))

  lines_en = [preprocess_sentence(l) for l in lines_en[:num_examples]]
  lines_ua = [preprocess_sentence(l) for l in lines_ua[:num_examples]]

  return lines_en, lines_ua


def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)
    
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
    
  return tensor, lang_tokenizer


def load_dataset(path_anki, path_target, num_examples=None):
  # creating cleaned input, output pairs
  en_sentences_anki, ua_sentences_anki = [], []
  if path_anki is not None:
    en_sentences_anki, ua_sentences_anki = create_dataset_anki(path_anki)
  
  en_sentences_target, ua_sentences_target = [], [] 
  if path_target is not None:
    en_sentences_target, ua_sentences_target = create_dataset_uatarget(path_target)

  # combine datasets
  en_sentences = [*en_sentences_anki, *en_sentences_target]
  ua_sentences = [*ua_sentences_anki, *ua_sentences_target]

  # cut datasets size
  if num_examples is not None:
    en_sentences = en_sentences[:num_examples]
    ua_sentences = ua_sentences[:num_examples]

  en_tensors, en_lang_tokenizer = tokenize(en_sentences)
  ua_tensors, ua_lang_tokenizer = tokenize(ua_sentences)

  return en_tensors, ua_tensors, en_lang_tokenizer, ua_lang_tokenizer


def show_tokens(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))


def main():
  path_to_file_anki = get_anki_dataset('ukr-eng.zip')
  path_to_file_target = get_target_dataset('ua-target-201704.zip')
    
  en_tensors, ua_tensors, en_lang_tokenizer, ua_lang_tokenizer = load_dataset(path_to_file_anki, path_to_file_target)
    
  print("English Language; index to word mapping")
  show_tokens(en_lang_tokenizer, en_tensors[-1])
  print()
  print("Ukrainian Language; index to word mapping")
  show_tokens(ua_lang_tokenizer, ua_tensors[-1])
