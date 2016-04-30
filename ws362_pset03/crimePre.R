setwd("/Users/shengwen/Dropbox/Yale/courses/DataMining/ws362_pset03/");
# getting train data
train <- read.csv("chiCrimeTrain.csv", sep="|", as.is=TRUE)
test <- read.csv("chiCrimeTest.csv", sep="|", as.is=TRUE)
train = subset(train, select=-c(testFlag, communityArea))
test = subset(test, select=-c(testFlag, communityArea))
ytrain = as.matrix(train[, 1])
Xtrain = as.matrix(train[,-1])
Xtrain[,'loc'] =  factor(Xtrain[,'loc'])
Xtrain[,'district'] =  factor(Xtrain[,'district'])
#Xtrain[,'ward'] =  factor(Xtrain[,'ward'])
#Xtrain[,'communityArea'] =  factor(Xtrain[,'communityArea'])


# getting test dat
ytest = as.matrix(test[, 1])
Xtest = as.matrix(test[,-1])
Xtest[,'loc'] = factor(Xtest[,'loc']);
Xtest[,'district'] =  factor(Xtest[,'district'])
#Xtest[,'ward'] =  factor(Xtest[,'ward'])
#Xtest[,'communityArea'] =  factor(Xtest[,'communityArea'])


# knn library(FNN)
#library(FNN)
#predKnn <- knn(Xtrain, Xtest, ytrain, k = 5)
# random forest
library(randomForest)
outRf <- randomForest(Xtrain,  factor(ytrain), ntree=11)
predRf <- predict(outRf, Xtest)

#calculate accuracy
index = !is.na(ytest)
error = (predRf[index] != ytest[index])
print(sum(error)/length(error))
# write result
#write.table(predRf, "./pset03.csv", row.names=FALSE, col.names=FALSE, sep=",")






