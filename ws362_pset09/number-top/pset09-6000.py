import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import operator

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

#Helper Functions
def tokenize(tok, textList, lower = 0, upper = 2000):
	tokenized = []
	for text in textList:
		tokenizedText = []
		for word in text:
			index = tok.word_index[word]
			if index > lower and index < upper:
				tokenizedText.append(index)
		tokenized.append(tokenizedText)
	return tokenized

#Read Data
path = "../mdsd/"
textAll = {}
errorCount = 0
maxWords = 250
tokCount = 6000
testLens = {}

for commentType in ['neg', 'pos']:
	for files in os.listdir(path + commentType):
		try:
			fin = open(path + commentType + "/" + files, 'r')
			if textAll.get(commentType) is None:
				textAll[commentType] = []
			textAll[commentType] += [" ".join(fin.readlines())]
		except UnicodeDecodeError:
			errorCount += 1
	print(commentType + " errorCount: " + str(errorCount))
	errorCount = 0
	testLens[commentType] = int(len(textAll[commentType]) / 3)

#Parse Data
tok = Tokenizer(tokCount)
tok.fit_on_texts(textAll['neg'] + textAll['pos'])

words = []
sortedDict = sorted(tok.word_index.items(), key = operator.itemgetter(1))

X_train = tok.texts_to_sequences(textAll['neg'][testLens['neg']: ] + textAll['pos'][testLens['pos']: ])
X_test = tok.texts_to_sequences(textAll['neg'][ :testLens['neg']] + textAll['pos'][ :testLens['pos']])

X_train = sequence.pad_sequences(X_train, maxlen = maxWords)
X_test = sequence.pad_sequences(X_test, maxlen = maxWords)

Y_train = [1] * (len(textAll['neg']) - testLens['neg']) + [0] * (len(textAll['pos']) - testLens['pos'])
Y_test = [1] * testLens['neg'] + [0] * testLens['pos']

model = Sequential()

#word embedding:
model.add(Embedding(tokCount, 32, input_length = maxWords))
model.add(Dropout(0.25))

model.add(SimpleRNN(16, return_sequences = False))

model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')

model.fit(X_train, Y_train, batch_size = 32, nb_epoch = 10, verbose = 1, callbacks=[EarlyStopping(monitor='val_loss', patience=2)], show_accuracy=True, validation_data = (X_test, Y_test))