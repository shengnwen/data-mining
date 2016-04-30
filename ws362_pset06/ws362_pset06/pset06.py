import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Adagrad
from keras.utils import np_utils

# function to read in and process the cifar-10 data
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(50000, 3072)
    X_test = X_test.reshape(10000, 3072)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # take only first 3 classes
    X_train = X_train[(y_train < 3).reshape(50000)]
    y_train = y_train[(y_train < 3).reshape(50000)]
    X_test = X_test[(y_test < 3).reshape(10000)]
    y_test = y_test[(y_test < 3).reshape(10000)]
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 3)
    Y_test = np_utils.to_categorical(y_test, 3)
    return X_train, Y_train, X_test, Y_test

# copy the first nlayers of model 'model' and freeze
# them. Note that these are layers in the keras sense,
# that is, it counts a drop-out layer or activation layer
# as an actual layer
def copy_freeze_model(model, nlayers = 1):
    new_model = Sequential()
    for l in model.layers[:nlayers]:
      l.trainable = False
      new_model.add(l)
    return new_model

# read in the dataset
(X_train, Y_train, X_test, Y_test) = load_data()

# for testing your code, you can downsample the data.
# for example, here we use just the first 1000 observations
# X_train = X_train[:1000]
# X_test = X_test[:1000]
# Y_train = Y_train[:1000]
# Y_test = Y_test[:1000]

# # simple example: one hidden layer with 16 hidden nodes
# model = Sequential()
# model.add(Dense(16, input_shape=(3072,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(3))
# model.add(Activation('softmax'))
#
# rms = RMSprop()
# model.compile(loss='categorical_crossentropy', optimizer=rms)
# model.fit(X_train, Y_train, batch_size=32, nb_epoch=25, verbose=1,
#           show_accuracy=True, validation_split=0.2)



# print('Classifcation rate %02.3f' % model.evaluate(X_test, Y_test, show_accuracy=True)[1])



# 1. width of model (2, 8, 32, 128, 512)
# simple example: one hidden layer with 16 hidden nodes
# hdn is hidden nodes number
# for i in range(5):
#     hdn = 2 * pow(4, i)
#     model = Sequential()
#     model.add(Dense(hdn, input_shape=(3072,)))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(3))
#     model.add(Activation('softmax'))
#
#     rms = RMSprop()
#     model.compile(loss='categorical_crossentropy', optimizer=rms)
#     model.fit(X_train, Y_train, batch_size=32, nb_epoch=25, verbose=0,
#               show_accuracy=True, validation_split=0.2)
#     print('Hidden Layer Number: %d\nClassifcation rate %02.3f\n\n' % (hdn, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
# 2. hln is hidden layers number
for i in range(5):
    model = Sequential()
    for hln in range(i + 1):
        model.add(Dense(512, input_shape=(3072,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=25, verbose=1,
              show_accuracy=True, validation_split=0.2)
    print('Hidden Layer Number: %d\nClassifcation rate %02.3f\n\n' % (i + 1, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))


# 3. freezing layers
model = Sequential()
for i in range(5):
    if i == 0:
        model.add(Dense(512, input_shape=(3072,)))
    else:
        model.add(Dense(512, input_shape=(512,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=25, verbose=1,
          show_accuracy=True, validation_split=0.2)
    print('Freezing Layer Number: %d\nClassifcation rate %02.3f\n\n' % (
        i + 1, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
    model = copy_freeze_model(model, (i+1)*3)

# 4. Autoencoder layers
model = Sequential()
for i in range(4):
    hdn = 32 * pow(4, i)
    model.add(Dense(hdn, input_shape=(3072,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(3072))
    # model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    # model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1,
    #           show_accuracy=True, validation_split=0.2)
    # print('Autoencoder r: %d\nClassifcation rate %02.3f\n\n' % (
    #     hdn, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
    model.fit(X_train, X_train, batch_size=32, nb_epoch=25, verbose=1,
              show_accuracy=True, validation_split=0.2)

    # pre - x_train)**2

    print('Autoencoder r: %d\nClassifcation rate %02.3f\n\n' % (
        hdn, model.evaluate(X_test, X_test, show_accuracy=True)[1]))
# 5. Autoencoder layers
model = Sequential()
hdn = 1024
model.add(Dense(hdn, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(3072))
# model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
# model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1,
#           show_accuracy=True, validation_split=0.2)
# print('Autoencoder r: %d\nClassifcation rate %02.3f\n\n' % (
#     hdn, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
model.fit(X_train, X_train, batch_size=32, nb_epoch=25, verbose=1,
          show_accuracy=True, validation_split=0.2)

# pre - x_train)**2

print('Autoencoder r: %d\nClassifcation rate %02.3f\n\n' % (
    hdn, model.evaluate(X_test, X_test, show_accuracy=True)[1]))
model = copy_freeze_model(model, 1 * 3)


for i in range(5):
    if i == 0:
        model.add(Dense(512, input_shape=(1024,)))
    else:
        model.add(Dense(512, input_shape=(512,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=25, verbose=1,
          show_accuracy=True, validation_split=0.2)
    print('Freezing Layer Number: %d\nClassifcation rate %02.3f\n\n' % (
        i + 1, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
    model = copy_freeze_model(model, (i+2)*3)

# 6. Autoencoder layers
model = Sequential()
hdn = 1024
model.add(Dense(hdn, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(3072))
# model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
# model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1,
#           show_accuracy=True, validation_split=0.2)
# print('Autoencoder r: %d\nClassifcation rate %02.3f\n\n' % (
#     hdn, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))
model.fit(X_train, X_train, batch_size=32, nb_epoch=25, verbose=1,
          show_accuracy=True, validation_split=0.2)

# pre - x_train)**2

print('Autoencoder r: %d\nClassifcation rate %02.3f\n\n' % (
    hdn, model.evaluate(X_test, X_test, show_accuracy=True)[1]))
model = copy_freeze_model(model, 1 * 3)

model.add(Dense(512, input_shape=(1024,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softplus'))

rms = RMSprop()
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
adg = Adagrad()
# al  = Adadelta()
adm = Adam() # 0.640
# admX = Adamax() # 0.507
model.compile(loss='categorical_crossentropy', optimizer=adg)
model.fit(X_train, Y_train, batch_size=32, nb_epoch=25, verbose=1,
          show_accuracy=True, validation_split=0.2)
print('Freezing Layer Number: %d\nClassifcation rate %02.3f\n\n' % (
    1, model.evaluate(X_test, Y_test, show_accuracy=True)[1]))