#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn import neighbors
from scipy.interpolate import interp1d

# A useful function for calculating the nearest neighbors:
#   neighbors.KNeighborsRegressor(n_neighbors)

# Use linear interpolation to predict new values on the test data
#   interp1d



# Save the results as "results.csv"
#Parse command line for input
# 1. parse arguments
parser = argparse.ArgumentParser(description='Parse input string')
#position input argument
parser.add_argument('trainFile', help='Input train filename')
parser.add_argument('testFile', help='Input test filename')
args = parser.parse_args()
trainFilename = args.trainFile
testFilename = args.testFile


# 2. read training data
trainData = np.genfromtxt(trainFilename, delimiter=',')
testData = np.genfromtxt(testFilename, delimiter=',')

train_shape = trainData.shape
factor_N = train_shape[1] - 1
data_N = train_shape[0]
test_N = testData.shape[0]
train_x = trainData[:, 1:]
train_y = trainData[:,0]

# 3. backfitting algorithm implementation
alpha = np.average(train_y)
g = np.zeros((data_N, factor_N),dtype=float) # gi for factor i
new_g = np.zeros((data_N, factor_N), dtype=float)
r = np.zeros((data_N, factor_N) , dtype=float)
for iter in range(25):
    # do 25 iterations
    for j in range(factor_N):
        for i in range(data_N):
            r[i][j] = (train_y[i] - alpha - sum(g[i, :] * train_x[i, :]) + g[i,j] * train_x[i,j])
        neigh = neighbors.KNeighborsRegressor(n_neighbors=20)
        neigh.fit(train_x[:,j].reshape((-1, 1)), r[:, j])
        new_g[:, j] = neigh.predict(train_x[:,j].reshape((-1, 1))) / train_x[:, j]
        new_g[:, j] = new_g[:, j] - sum(new_g[:, j] * train_x[:,j]) / data_N
    g = new_g
result = np.zeros(testData.shape[0], dtype=float)
result += alpha
for j in range(factor_N):
    interF = interp1d(train_x[:, j], g[:,j])
    result += interF(testData[:,j]) * testData[:, j]
with open('results.csv', 'w') as f:
    res = ""
    for i in result:
        res += str(i) + "\n"
    f.write(res)
# print(result)