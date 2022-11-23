import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle
import re
import tqdm
contractions_dict = { "ain't": "are not", "'s":"is", "aren't": "are not", "can't": "cannot",
                     "can't've": "cannot have", "‘cause": "because", "could've": "could have",
                     "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
                     "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
                     "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
                     "he'll": "he will", "he'll've": "he will have", "how'd": "how did", "how'd'y": "how do you",
                     "how'll": "how will", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                     "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                     "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                     "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                     "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", 
                     "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
                     "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                     "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                     "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
                     "she'll": "she will", "she'll've": "she will have", "should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                     "that'd": "that would", "that'd've": "that would have", "there'd": "there would", 
                     "there'd've": "there would have", "they'd": "they would", "they'd've": "they would have",
                     "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                     "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                     "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                     "we've": "we have", "weren't": "were not","what'll": "what will", 
                     "what'll've": "what will have", "what're": "what are", "what've": "what have",
                     "when've": "when have", "where'd": "where did", "where've": "where have", "who'll": "who will",
                     "who'll've": "who will have", "who've": "who have", "why've": "why have",
                     "will've": "will have", "won't": "will not", "won't've": "will not have",
                     "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                     "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                     "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                     "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 
                     "you're": "you are", "you've": "you have", "what's": "what is"}
def text_norm(documents):
    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, s)
    contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))
    line_return = documents.lower()
    line_return = expand_contractions(line_return)
    line_return = re.sub(r'[^\w\s]', ' ', line_return)
    return line_return
def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):
    test_sentences1 = [x[0].lower() for x in test_sentences_pair]
    test_sentences2 = [x[1].lower() for x in test_sentences_pair]

    test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

    return test_data_1, test_data_2
def query(form_data):
    MAX_SEQUENCE_LENGTH = 10
    token_emb_path = 'token_emb.dictionary'
    checkpoint_path='checkpoint.h5'
    model = load_model(checkpoint_path)
    with open(token_emb_path, 'rb') as config_dictionary_file:
        config_dictionary = pickle.load(config_dictionary_file)
    embedding_meta_data = {'tokenizer': config_dictionary['tokenizer'],
                    'embedding_matrix': config_dictionary['embedding_matrix']}
    x1=text_norm(form_data["Sentence 1:"])
    x2=text_norm(form_data["Sentence 2:"])

    test_sentence_pairs = [(x1, x2)]

    test_data_x1, test_data_x2 = create_test_data(embedding_meta_data['tokenizer'],
                                                          test_sentence_pairs,  MAX_SEQUENCE_LENGTH)
    preds = list(model.predict([test_data_x1, test_data_x2], verbose=1))
    print(form_data["Sentence 1:"],'\n',form_data["Sentence 2:"])
    print(preds)
    return preds
