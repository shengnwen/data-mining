import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.utils import np_utils
import keras.layers.convolutional as kcnn
# set this to false once you have tested your code!
# TEST = True

# function to read in and process the cifar-10 data; set the
# number of classes you want
def load_data(nclass):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # down-sample to three classes
    X_train = X_train[(y_train < nclass).reshape(50000)]
    y_train = y_train[(y_train < nclass).reshape(50000)]
    X_test = X_test[(y_test < nclass).reshape(10000)]
    y_test = y_test[(y_test < nclass).reshape(10000)]
    # create responses
    Y_train = np_utils.to_categorical(y_train, nclass)
    Y_test = np_utils.to_categorical(y_test, nclass)
    if TEST:
        X_train = X_train[:1000]
        Y_train = Y_train[:1000]
        X_test = X_test[:1000]
        Y_test = Y_test[:1000]
    return X_train, Y_train, X_test, Y_test


# Note: You'll need to do this manipulation to construct the
# output of the autoencoder. This is because the autoencoder
# will have a flattend dense layer on the output, and you need
# to give Keras a flatted version of X_train
TEST = True
(X_train, Y_train, X_test, Y_test) = load_data(2)
print(X_test.shape)
print(Y_test.shape)
# apply a 2x2 convolution with 64 output filters on a 256x256 image:
X_train_auto_output = X_train.reshape(X_train.shape[0], 3072)
for i in range(3):
    model = Sequential()
    k_size = 2 * i + 1
    model.add(kcnn.Convolution2D(32, k_size, k_size, border_mode='same', input_shape=(3, 32, 32)))
    model.add(kcnn.MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3072))
    # model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    model.fit(X_train, X_train_auto_output, batch_size=32, nb_epoch=25, verbose=1,
              show_accuracy=True, validation_split=0.2)
    print('2: %d\nClassifcation rate %02.3f\n\n' % (
        1, model.evaluate(X_test, X_test.reshape(X_test.shape[0], 3072)
                          , show_accuracy=True)[1]))






