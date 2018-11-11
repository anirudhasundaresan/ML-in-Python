#!/usr/bin/python2

from __future__ import print_function
__author__ = 'anirudhasundaresan'

import numpy as np
import urllib2
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = urllib2.urlopen(target_url)

xList = []
labels = []
names = []
firstLine = True

for line in data:
    if firstLine:
        names = line.strip().split(';') # the strip() removes the '\n' at the end
        firstLine = False
    else:
        # split on semi-colon
        row = line.strip().split(';')
        labels.append(float(row[-1])) # labels in a separate array
        row.pop()
        floatRow = [float(num) for num in row]
        xList.append(floatRow)

indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3==0]
xListTrain = [xList[i] for i in indices if i%3!=0]
labelsTest = [labels[i] for i in indices if i%3==0]
labelsTrain = [labels[i] for i in indices if i%3!=0]

xTrain = np.array(xListTrain)
xTest = np.array(xListTest)
yTrain = np.array(labelsTrain)
yTest = np.array(labelsTest)

alphaList = [0.1**i for i in [0, 1, 2, 3, 4, 5, 6]]
# parameter of the model: alpha; adds in complexity.

rmsError = []
for alph in alphaList:
    wineRidgeModel = linear_model.Ridge(alpha=alph)
    wineRidgeModel.fit(xTrain, yTrain)
    rmsError.append(np.linalg.norm(yTest - wineRidgeModel.predict(xTest), 2)/sqrt(len(yTest)))

print("RMS Error and alpha: ")
for i in range(len(rmsError)):
    print(rmsError[i], alphaList[i])

# plot curve of oos error vs. alpha to choose model with lowest error
x = range(len(rmsError))
plt.plot(x, rmsError, 'k')
plt.xlabel('-log(alpha)')
plt.ylabel('Error - RMS')
plt.show()
# It is conventional to show least complex model on the left of the plot and most complex model on the right.

# find best model and retrain and plot hist of errors
indexBest = rmsError.index(min(rmsError))
alph = alphaList[indexBest]
# print(indexBest, alph)
wineRidgeModel = linear_model.Ridge(alpha=alph)
wineRidgeModel.fit(xTrain, yTrain)
errorVector = yTest - wineRidgeModel.predict(xTest)
plt.hist(errorVector)
plt.xlabel('Bin boundaries')
plt.ylabel('Counts')
plt.show()

plt.scatter(wineRidgeModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel('Predicted taste score')
plt.ylabel('Actual taste score')
plt.show()

'''
The fwd stepwise regression produced a family of models - one with 2 attributes, one with 3 attributes and so on. RR also produces a family of models based on the alpha values. Alpha defines the severity of the penalty on beta coefficients.
Here, range of alpha was fixed - logarithmically; actually, it should be exponential and must cover a good range of values.
'''

