import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization

# load the STL-10 crime data into Python. you need to first
# download these from here:
#    http://euler.stat.yale.edu/~tba3/class_data/stl10

X_train = np.genfromtxt('X_train_new.csv', delimiter=',')
Y_train = np.genfromtxt('Y_train.csv', delimiter=',')
X_test = np.genfromtxt('X_test_new.csv', delimiter=',')
Y_test = np.genfromtxt('Y_test.csv', delimiter=',')

# print(X_train.shape)
# print out (5000, 4096)
# print(Y_train.shape)
# print out (5000, 10)

# 1. Dense Neural Network
model = Sequential()

model.add(Dense(128, input_shape=(4096,), init="glorot_normal"))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=32, nb_epoch=25,
          verbose=1, show_accuracy=True, validation_split=0.1)

print('1: %d\nClassifcation rate %02.3f\n\n' % (
    1, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))