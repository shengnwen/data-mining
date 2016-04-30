import numpy as np
from sklearn import linear_model
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD, RMSprop
# from keras.layers.normalization import BatchNormalization

# load the STL-10 crime data into Python. you need to first
# download these from here:
#    http://euler.stat.yale.edu/~tba3/class_data/stl10
TEST = False
if TEST:
    X_train = np.genfromtxt('X_train_new.csv', delimiter=',', max_rows= 100)
    Y_train = np.genfromtxt('Y_train.csv', delimiter=',', max_rows= 100)
    X_test = np.genfromtxt('X_test_new.csv', delimiter=',', max_rows= 100)
    Y_test = np.genfromtxt('Y_test.csv', delimiter=',', max_rows= 100)
else:
    X_train = np.genfromtxt('X_train_new.csv', delimiter=',')
    Y_train = np.genfromtxt('Y_train.csv', delimiter=',')
    X_test = np.genfromtxt('X_test_new.csv', delimiter=',')
    Y_test = np.genfromtxt('Y_test.csv', delimiter=',')

# print(X_train.shape)
# print out (5000, 4096)
# print(Y_train.shape)
# print out (5000, 10)

# 3. Lasso

clf = linear_model.Lasso(alpha = 0.1)
clf.fit(X_train, Y_train)
predicted = clf.predict(X_test)
predicted = np.argmax(predicted, axis = 1)
expected = np.where(Y_test == 1)[1]
print ('3: \nClassifcation rate %02.3f\n\n' % (sum(predicted == expected) / Y_test.shape[0]))


