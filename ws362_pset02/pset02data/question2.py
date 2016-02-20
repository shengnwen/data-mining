#In these questions you will fit a random forest estimator to the LEHD data from the state of
#Connecticut (ct).
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score

def retrieveXY(state = 'ct', delete_names = ['h_geocode','C000','CE01','CE02','CE03', 'createdate']):
    data_file = "./" + state + "_rac_S000_JT00_2013.csv"
    data = np.genfromtxt(data_file, delimiter=',', names=True)
    # print(data.shape)

    data_c000 = data['C000']
    train_y = data['CE03'] / data_c000
    names = list(data.dtype.names)
    # print(names)
    for name in delete_names:
        names.remove(name)
    train_x = data[names]
    # print(data[0:10])
    for name in names:
        if name == 'h_geocode':
            continue
        train_x[name] = train_x[name] / data_c000
    train_x = train_x.view(np.float64).reshape(list(train_x.shape) + [len(train_x.dtype)])
    return (names, train_x, train_y)



def findBestRandomForestModel(state = 'ct'):
    _, train_x, train_y = retrieveXY(state)
    best_tree_number = 25 # 25 - 50
    min_error = 1
    best_model = None
    for treeNumber in range(25, 51):
        model = RandomForestRegressor(n_estimators=treeNumber, criterion='mse',
                                  max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                  max_features=10, max_leaf_nodes=None, bootstrap=True,
                                  oob_score=True, n_jobs=1, random_state=None, verbose=0,
                                  warm_start=False)
        model.fit(train_x, train_y)
        error = mean_squared_error(train_y, model.oob_prediction_)
        print("Tree Number: " + str(treeNumber) + ", mse:" + str(error) +"\n")
        if error < min_error:
            min_error = error
            best_tree_number = treeNumber
            best_model = model
    print("Best tree number:" + str(best_tree_number) + "\nMin Error:" + str(min_error))
    return model


def randomForestModel(train_x, train_y):
    model = RandomForestRegressor(n_estimators=50, criterion='mse',
                                  max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                  max_features=10, max_leaf_nodes=None, bootstrap=True,
                                  oob_score=True, n_jobs=1, random_state=None, verbose=0,
                                  warm_start=False)
    model.fit(train_x, train_y)
    return model
# Q2
def otherStateMSE(model):
    states = ['ny', 'mt', 'ca']
    for s in states:
        _, x, y = retrieveXY(s)
        print("MSE: state " + s + ": " + str(mean_squared_error(y, model.predict(x))))

# Q3 linear regression model
def linearRegressionModel(train_x, train_y):
    model = LinearRegression()
    model.fit(train_x, train_y)
    print("MSE: state CT" + ": " + str(mean_squared_error(train_y, model.predict(train_x))))
    return model

states = ['ny', 'mt', 'ca']
# ctModel = findBestRandomForestModel()
# _, train_x, train_y = retrieveXY()
# rfModel = randomForestModel(train_x, train_y)
# otherStateMSE(rfModel)
# linearModel = linearRegressionModel(train_x, train_y)
# otherStateMSE(linearModel)

# Q4:
def findWorstCounties(rfModel):
    states = ['ny', 'ca']
    delete_names = ['C000','CE01','CE02','CE03', 'createdate']
    for s in states:
        countyDict = {}
        _, train_x, train_y = retrieveXY(s, delete_names)
        for idx in range(len(train_x)):
            key = ("{0:.0f}".format(train_x[idx][0]))[-13:-10]
            if key not in countyDict:
                countyDict[key] = [[],[]]
            countyDict[key][0].append(train_x[idx][1:])
            countyDict[key][1].append(train_y[idx])
        print("States " + s)
        res = []
        for key in countyDict:
            predict_y = rfModel.predict(countyDict[key][0])
            res.append([key, mean_squared_error(countyDict[key][1], predict_y)])
        res.sort(key=lambda x:x[1], reverse=True)
        for i in res:
            print(s + "-" + i[0] + ":" + str(i[1]))

# Q4:
names, train_x, train_y = retrieveXY()
rfModel = randomForestModel(train_x, train_y)

# Q5
importance_metrics = rfModel.feature_importances_
print(importance_metrics)
sortIdx = sorted(range(len(importance_metrics)), key=lambda k: importance_metrics[k], reverse=True)
print("5 morst important varibles(high - low)")
for idx in range(5):
    print(names[sortIdx[idx]])
# findWorstCounties(rfModel)
# otherStateMSE(rfModel)

