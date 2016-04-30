import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re

from keras.datasets import imdb, reuters
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.text import Tokenizer


def loadMDSD(input_text, input_label):
    path = "./mdsd/"

    ff = [path + "pos/" + x for x in os.listdir(path + "pos")] + \
         [path + "neg/" + x for x in os.listdir(path + "neg")]

    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text)

    count = 0
    errorN = 0
    for f in ff:
        with open(f) as fin:
            try:
                input_text += [remove_tags(" ".join(fin.readlines()))]
                count += 1
            except Exception:
                print(f)
                errorN += 1
    print(errorN / count)

input_text = []
input_label = ([1] * 12500 + [0] * 12500) * 2

loadMDSD(input_text, input_label)
num_words = 2000
trainDataN = 25000
tok = Tokenizer(num_words)
tok.fit_on_texts(input_text[:trainDataN])
words = []
for iter in range(num_words):
    words += [key for key,value in tok.word_index.items() if value==iter+1]

print(words[:10])


X_train = tok.texts_to_sequences(input_text[:trainDataN])
X_test  = tok.texts_to_sequences(input_text[trainDataN:])
y_train = input_label[:trainDataN]
y_test  = input_label[trainDataN:]

X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test  = sequence.pad_sequences(X_test,  maxlen=100)

def reconstruct_text(index, words):
    text = []
    for ind in index:
        if ind != 0:
            text += [words[ind-1]]
        else:
            text += [""]
    return text

print(input_text[100])
print("\n\n")
print(reconstruct_text(X_train[100][:40], words))