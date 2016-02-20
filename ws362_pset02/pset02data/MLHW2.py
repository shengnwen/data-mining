import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression


class County:
    def __init__(self, index, err):
        self.ind = [index]
        self.err = err

class state:

    def __init__(self, data):
        unwanted = ['h_geocode', 'C000', 'CE01', 'CE02', 'CE03', 'createdate']
        self.total = data['C000']
        self.rich = data['CE03'] / self.total
        self.geoCode = data[unwanted[0]]
        
        fields = list(data.dtype.names)
        for field in unwanted:
            fields.remove(field)
        data = data[fields]
        for field in data.dtype.names:
            data[field] /= self.total
        self.fields = data.dtype.names
        self.data = data.view(np.float64).reshape(list(data.shape) + [len(data.dtype)])

    def getPrunedData(self):
        deColli = ['CS02', 'CD04', 'CA03', 'CNS20', 'CR07', 'CT02']
        tobeDel = []
        for field in deColli:
            tobeDel.append(self.fields.index(field))
        return np.delete(self.data, tobeDel, 1)

    def arrangeCounty(self, mse = None):
        self.countyDict = {}
        for (index, county) in enumerate(self.geoCode):
            county = "{0:.0f}".format(county)[-13:-10]
            if self.countyDict.get(county) == None:
                self.countyDict[county] = County(index, mse[index] if mse is not None else None)
            else:
                self.countyDict[county].ind += [index]
                self.countyDict[county].err += mse[index] if mse is not None else None
        for key in self.countyDict:
            self.countyDict[key].err = self.countyDict[key].err / len(self.countyDict[key].ind)

def meanMse(data1, data2):
    return np.average((data1 - data2) ** 2)

def printGeocode(state, geocodes):
    geoString = ""
    for each in geocodes:
        geoString += "{0:.0f}".format(state.geoCode[each])
        geoString += " "
        geoString += str(state.data[each])
    print(geoString)
from sklearn.metrics import mean_squared_error
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
e = meanMse(y_true, y_pred)
print(e)
# np.set_printoptions(suppress = True)
#
# ct = state(np.genfromtxt('ct_rac_S000_JT00_2013.csv', delimiter = ',', names = True))
# ca = state(np.genfromtxt('ca_rac_S000_JT00_2013.csv', delimiter = ',', names = True))
# mt = state(np.genfromtxt('mt_rac_S000_JT00_2013.csv', delimiter = ',', names = True))
# ny = state(np.genfromtxt('ny_rac_S000_JT00_2013.csv', delimiter = ',', names = True))
#
# minError = 1
# minTree = 90
#
# '''
# Q1: What is used in deciding the ultimate number of trees to use
#     First, cross-validation (not necessary)
#     Second, the oob score provided by RF model ( not sure how to use)
#     Finally, calculate the oob mean mse.
# '''
#
# #for nTree in range(80, 100):
# #    ctRfTree = RandomForestRegressor(n_estimators = nTree, max_features = 10, n_jobs = -1, oob_score = True)
# #    ctRfTree.fit(ct.data, ct.rich)
# #    mean_mse = np.average((ctRfTree.oob_prediction_ - ct.rich) ** 2)
# #    print("OOB error " + str(mean_mse) + " nTrees: " + str(nTree))
# #    if mean_mse < minError:
# #        minTree = nTree
# #        minError = mean_mse
# #print("minError: " + str(minError) + "; minTree: " + str(minTree))
#
# ctRfTree = RandomForestRegressor(n_estimators = minTree, max_features = 10, n_jobs = -1, oob_score = True)
# ctRfTree.fit(ct.data, ct.rich)
#
# print("q2: ")
#
# ct.mse = meanMse(ctRfTree.oob_prediction_, ct.rich)
# print("oob error " + str(ct.mse) + " ntrees: " + str(minTree))
#
# ca.mse = meanMse(ctRfTree.predict(ca.data), ca.rich)
# print("oob error " + str(ca.mse) + " ntrees: " + str(minTree))
#
# mt.mse = meanMse(ctRfTree.predict(mt.data), mt.rich)
# print("oob error " + str(mt.mse) + " ntrees: " + str(minTree))
#
# ny.mse = meanMse(ctRfTree.predict(ny.data), ny.rich)
# print("oob error " + str(ny.mse) + " ntrees: " + str(minTree))
#
# ctlr = LinearRegression(copy_X = False, n_jobs = -1)
# ctlr.fit(ct.getpruneddata(), ct.rich)
#
# print("q3: ")
#
# ct.mse = meanMse(ctlr.predict(ct.getpruneddata()), ct.rich)
# print("lr error: " + str(ct.mse))
#
# ca.mse = meanMse(ctlr.predict(ca.getpruneddata()), ca.rich)
# print("lr error: " + str(ca.mse))
#
# mt.mse = meanMse(ctlr.predict(mt.getpruneddata()), mt.rich)
# print("lr error: " + str(mt.mse))
#
# ny.mse = meanMse(ctlr.predict(ny.getpruneddata()), ny.rich)
# print("lr error: " + str(ny.mse))
#
# print("Q4: ")
#
# caPredict = ctRfTree.predict(ca.data)
# ca.arrangeCounty((caPredict - ca.rich) ** 2)
# caCountyRank = sorted(list(ca.countyDict.items()), key = lambda x: x[1].err)
# print("worst CA county:")
# for i in range(0, 5):
#     print(caCountyRank[-1 - i])
#
# nyPredict = ctRfTree.predict(ny.data)
# ny.arrangeCounty((nyPredict - ny.rich) ** 2)
# nyCountyRank = sorted(list(ny.countyDict.items()), key = lambda x: x[1].err)
# print("worst NY county:")
# for i in range(0, 5):
#     print(nyCountyRank[-1 - i])
#
# print("Q5:")
#
# #features = np.argsort(list(ctRfTree.feature_importances_))[::-1][:10]
# #badFeatures = np.argsort(list(ctRfTree.feature_importances_))[:10]
# #ctRfTree.fit(arrCtData[:, features], richJobs)
# #mean_mse = np.average((ctRfTree.oob_prediction_ - richJobs) ** 2)
# #print("OOB error " + str(mean_mse) + " nTrees: " + str(minTree))
# #ctRfTree.fit(arrCtData[:, badFeatures], richJobs)
# #mean_mse = np.average((ctRfTree.oob_prediction_ - richJobs) ** 2)
#print("OOB error " + str(mean_mse) + " nTrees: " + str(minTree))