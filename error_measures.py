#!/usr/bin/python2
from __future__ import print_function
from math import sqrt

__author__ = 'anirudhasundaresan'

# For regression problems:
target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

error = []
for i in range(len(target)):
    error.append(target[i] - prediction[i])

print("Errors: ", error)

squared_err, abs_err = [], []

for val in error:
    squared_err.append(val*val)
    abs_err.append(abs(val))

print("Sqaured error: ", squared_err)
print("Mean Abs. error: ", abs_err)

print("MSE = ", sum(squared_err)/len(squared_err))
print("RMSE = ", sqrt(sum(squared_err)/len(squared_err)))
print("MAE = ", sum(abs_err)/len(abs_err))

targetMean = sum(target)/len(target)

targetDeviation = []
for val in target:
    targetDeviation.append((val-targetMean)*(val-targetMean))

print("Variance of target: ", sum(targetDeviation)/len(targetDeviation))
print("SD of target: ", sqrt(sum(targetDeviation)/len(targetDeviation)))

'''
RMSE is usually more usable number to calculate; MSE comes out more different than RMSE and MAE due to squaring. Lesser the MSE/ RMSE, better is the model. Also, the target variance and SD must be compared to the MSE and RMSE of prediction errors. If MSE turns out to be close to variance of the targets, then it means the algorithm is equivalent to a model where all the predictions are close to the mean of the target vector - not a good model.
In the calculations above, RMSE is around half of SD of targets - hence, fairly good model. Also, looking at boxplots of the error and quantile plots, degree of normality, etc. will highlight insights into error sources.
'''

# For classification problems:
'''
Confusion matrix/ contingency tables are used. For a classification, a result could be either TP/ FP/ FN/ TN.
TP - predicted label is +class and real label is +class.
FP - predicted label is +class, but real label is -class.
FN - predicted label is -class, but real label is +class.
TN - predicted label is -class and real label is -class.
Classification algos spit out output probabilities for a test data point, based on a threshold. That is, for example, if prob is > 0.7, then it is class 1, else class 2.
Setting threshold depends on the problem statement and costs/ rewards associated with the falses/ trues in the confusion matrix/ contingency tables. Changing threshold leads to a change in the numbers of confusion matrix.
'''
# Build confusion matrix for rocksVMines dataset:

import urllib2
import random
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl
import pdb

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data = urllib2.urlopen(target_url)
xList, labels = [], []

for line in data:
    ls = line.strip().split(',')
    if ls[-1]=='M':
        ls[-1] = '1.00'
    else:
        ls[-1] = '0.00'
    labels.append(float(ls[-1]))
    ls.pop()
    xList.append([float(num) for num in ls])

# divide attribute matrix and label vector into training (2/3 of data) and testing sets (1/3 of data).

indices = range(len(xList))
# scramble the data and assign to train and test splits
xListTest = [xList[i] for i in indices if i%3==0]
xListTrain = [xList[i] for i in indices if i%3!=0]
labelsTest = [labels[i] for i in indices if i%3==0]
labelsTrain = [labels[i] for i in indices if i%3!=0]

# sklearn takes in numpy array; so convert them
xTrain = np.array(xListTrain)
xTest = np.array(xListTest)
yTrain = np.array(labelsTrain)
yTest = np.array(labelsTest)

print(len(xList))
print(len(labels))

print("Shape of xTrain: ", xTrain.shape)
print("Shape of xTest: ", xTest.shape)
print("Shape of yTrain: ", yTrain.shape)
print("Shape of yTest: ", yTest.shape)

# We will be using linear regression with 'M' class as 1.0 and 'R' class as 0.0.
# OLS model used here. This model will mostly predict values in [0, 1] but not always. Depending on threshold of predicted value, we will arrive at a decision here. The predictions are not quite probabilties.
rocksVMinesModel = linear_model.LinearRegression()
rocksVMinesModel.fit(xTrain, yTrain)

# generate predicitons on in-sample error/ training error.
trainingPredictions = rocksVMinesModel.predict(xTrain)
testPredictions = rocksVMinesModel.predict(xTest)
print("Some values predicted from training set: ", trainingPredictions[:5], trainingPredictions[:-6:-1])

# We need to pass the actual labels and predicted labels with threshold to a confusion matrix function.
def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual): return -1
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    for ind, ele in enumerate(actual):
        if ind==1.0:
            if predicted[ind] > threshold:
                tp += 1
            else:
                fn += 1
        else:
            if predicted[ind] < threshold:
                tn += 1
            else:
                fp += 1
    return [tp, fn, fp, tn]

tp, fn, fp, tn = confusionMatrix(trainingPredictions, yTrain, 0.5)
print("On training data: ")
print('tp = ', tp, '; fn = ', fn, '; fp = ', fp, '; tn = ', tn)
print("Total number of errors: ", fp+fn)

tp, fn, fp, tn = confusionMatrix(testPredictions, yTest, 0.5)
print("On test data: ")
print('tp = ', tp, '; fn = ', fn, '; fp = ', fp, '; tn = ', tn)
print("Total number of errors: ", fp+fn)

# generate ROC curve for this prediciton:
# in-sample ROC:
fpr, tpr, thresholds = roc_curve(yTrain, trainingPredictions)
roc_auc = auc(fpr, tpr)
print("AUC for in-sample ROC curve: ", roc_auc)

pl.clf()
pl.plot(fpr, tpr, label = 'ROC curve: (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel("False + rate")
pl.ylabel("True + rate")
pl.legend(loc="lower right")
pl.show()

# out-of-sample ROC:
fpr, tpr, thresholds = roc_curve(yTest, testPredictions)
roc_auc = auc(fpr, tpr)
print("AUC for out-of-sample ROC curve: ", roc_auc)

pl.clf()
pl.plot(fpr, tpr, label = 'ROC curve: (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel("False + rate")
pl.ylabel("True + rate")
pl.legend(loc="lower right")
pl.show()

'''
The goal is to choose a threshold such that mis-classification rate (of out-of-sample set) is minimum. Also, the total cost of classification depends on the indivdual cost of the elements of confusion matrix. Minimizing the cost should give a better threshold. 100$ for FP and 1000$ for FN.
Also, the data set might be imbalanced to begin with; this must be dealt with before fitting models.

ROC curve is plotted between TPR and FPR.
TPR = TP/(all +) = TP/TP+FN
FPR = FP/(all -) = FP/TN+FP

If threshold is set too low, every data point will be classified +ve. So, FN = 0 and TPR = 1; TN = 0 and FPR = 1
If threshold is set high, every point will be classified -ve. TP = 0 and TPR = 0; FP = 0 and FPR = 0

AUROC will determine performance of classifier. Middle diagonal line (serves as reference point) is for random guessing and covers 50% area. Closer the ROC curve is to the top left corner, the better the model is. Ideal ROC is a line from 0,0 to 0,1 and then to 1,0.

If ROC curve is significantly under the middle diagonal line, check code for '-' errors.
For perfect classifier: AUC = 1 and random guessing has AUC = 0.5

These methods: AUC, ROC and confusion matrix all work for multi-class classification as well.
'''


