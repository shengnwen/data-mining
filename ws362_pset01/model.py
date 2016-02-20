from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def knnModel(k, X, T, y):
    knn = KNeighborsRegressor(k, 'uniform', 'auto', 28,2)
    # knn = KNeighborsRegressor(k, 'uniform', 'auto',30, 2)
    knn.fit(X, y)
    pre_y = knn.predict(T)
    return pre_y


def rigRegModel(X, T, y):
    ridge = RidgeCV(alphas=[0.1, 2.0, 5.0])
    ridge.fit(X, y)
    pre_y = ridge.predict(T)
    # print(pre_y)
    return pre_y

def linRegModel(X, T, y):
    ols = linear_model.LinearRegression()
    ols.fit(X, y)
    pre_y = ols.predict(T)
    print(pre_y)
    return pre_y

def plot_figs(fig_num, elev, azim, X, pre_y):
    fig = plt.figure(fig_num, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)
    ax.scatter(X[:, 0], X[:,1], pre_y, c='k', marker='8')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Leave Probability')
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])
def main():
    # use three function to predict
    trainFile = "./nyc_train.csv"
    testFile = "./nyc_test.csv"
    data = pandas.read_csv(trainFile,sep=",")
    trainData = pandas.read_csv(testFile,sep=",")
    #knnModel(100, data, trainData, y)
    y = []
    for i in data['dropoff_BoroCode']:
        if i == 1:
            # print("0")
            y.append(0)
        else:
            # print("1")
            y.append(1)
    X1 = [[] for i in range(len(data['pickup_datetime']))]
    T1 = [[] for i in range(len(trainData['pickup_datetime']))]
    AllBeforeEncode = [[] for i in range(len(data['pickup_datetime']) + len(trainData['pickup_datetime']))]
    for i in range(len(data['pickup_datetime'])):
        X1[i] = [data['pickup_longitude'][i], data['pickup_latitude'][i]]
        AllBeforeEncode[i] = [int(data['pickup_datetime'][i][11:13]), int(data['pickup_NTACode'][i][2:])]
    for i in range(len(trainData['pickup_datetime'])):
        T1[i] = [trainData['pickup_longitude'][i], trainData['pickup_latitude'][i]]
        AllBeforeEncode[i + len(data['pickup_datetime'])] = \
            [int(trainData['pickup_datetime'][i][11:13]), int(trainData['pickup_NTACode'][i][2:])]
    enc = OneHotEncoder(sparse = False)
    # print(AllBeforeEncode[0:100])
    afterEncoded = enc.fit_transform(AllBeforeEncode)
    X2 = afterEncoded[:len(data['pickup_datetime'])]
    T2 = afterEncoded[len(data['pickup_datetime']):]
    y1 = knnModel(100, X1, T1, y)
    y2 = linRegModel(X2, T2, y)
    y3 = rigRegModel(X2, T2, y)
    with open('pset01.csv', 'w') as f:
        res = "__knn__, __linear__, __ridge__\n"
        for i in range(len(y1)):
            res += str(y1[i]) + ',' + str(y2[i]) + ',' +  str(y3[i]) + '\n'
        f.write(res)
    elev = 43.5
    azim = -110
    T1_origin = np.array(T1)
    T2_origin = np.array(AllBeforeEncode[len(data['pickup_datetime']):])
    # plot_figs(1, elev, azim, T2_origin, y2)
    #
    # elev = -.5
    # azim = 0
    # plot_figs(2, elev, azim, T2_origin, y2)
    #
    # elev = -.5
    # azim = 90
    # plot_figs(3, elev, azim, T2_origin, y2)

    plot_figs(1, elev, azim, T1_origin, y1)

    elev = -.5
    azim = 0
    plot_figs(2, elev, azim, T1_origin, y1)

    elev = -.5
    azim = 90
    plot_figs(3, elev, azim, T1_origin, y1)
    plt.show()
main()