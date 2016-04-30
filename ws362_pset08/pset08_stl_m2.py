import numpy as np

# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD, RMSprop
# from keras.layers.normalization import BatchNormalization
from sklearn import svm

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

# 2. SVM


Y_train = np.where(Y_train == 1)[1]
Y_test = np.where(Y_test == 1)[1]
svc = svm.SVC(kernel='poly', degree = 1)
svc.fit(X_train, Y_train)
predicted = svc.predict(X_test)
print ('2: \nClassifcation rate %02.3f\n\n' % (sum(predicted == Y_test) / Y_test.shape[0]))

