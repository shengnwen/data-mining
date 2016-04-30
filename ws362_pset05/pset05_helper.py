import numpy as np
import pickle
import csv

def load_mnist_train():
  mnist_train = []
  with open('data/mnist_train.csv', 'rt') as csvfile:
    cread = csv.reader(csvfile)
    for row in cread:
      vals = np.array([float(x) / 256 for x in row[1:]])
      vals = vals.reshape((784,1))
      res = np.zeros((10, 1))
      res[int(row[0])] = 1
      mnist_train.append([vals, res])
  return mnist_train


def load_mnist_test():
  mnist_test = []
  with open('data/mnist_test.csv', 'rt') as csvfile:
    cread = csv.reader(csvfile)
    for row in cread:
      vals = np.array([float(x) / 256 for x in row[1:]])
      vals = vals.reshape((784,1))
      res = np.int64(row[0])
      mnist_test.append([vals, res])
  return mnist_test


def load_cifar_train():
  mnist_train = []
  with open('data/cifar_train.csv', 'rt') as csvfile:
    cread = csv.reader(csvfile)
    for row in cread:
      vals = np.array([float(x) / 256 for x in row[1:]])
      vals = vals.reshape((3072,1))
      res = np.zeros((10, 1))
      res[int(row[0])] = 1
      mnist_train.append([vals, res])
  return mnist_train


def load_cifar_test():
  mnist_test = []
  with open('data/cifar_test.csv', 'rt') as csvfile:
    cread = csv.reader(csvfile)
    for row in cread:
      vals = np.array([float(x) / 256 for x in row[1:]])
      vals = vals.reshape((3072,1))
      res = np.int64(row[0])
      mnist_test.append([vals, res])
  return mnist_test

