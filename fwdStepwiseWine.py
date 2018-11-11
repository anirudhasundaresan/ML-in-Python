#!/usr/bin/python2

from __future__ import print_function
__author__ = 'anirudha sundaresan'

import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import urllib2
import matplotlib.pyplot as plt

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = urllib2.urlopen(target_url)

xList = []
labels = []
names = []

firstLine = True

for line in data:
    if firstLine==True:
        names.extend(line.strip().split(';'))
        firstLine = False
    else:
        row = line.strip().split(';')
        labels.append(float(row[-1]))
        row.pop()
        floatRow = [float(num) for num in row]
        xList.append(floatRow)

# divide xList and labels into training and test sets.
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3==0]
xListTrain = [xList[i] for i in indices if i%3!=0]
labelsTest = [labels[i] for i in indices if i%3==0]
labelsTrain = [labels[i] for i in indices if i%3!=0]

# function to extract certain columns from a matrix based on an index set passed to it
def xattrSelect(x, idxSet):
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return xOut

# build list of attributes one at a time -- start with empty
attributeList = []
index = range(len(xList[1]))
indexSet = set(index)
indexSeq = []
oosError = []

for i in index: # each pass selects a new attribute to be added to attributeList
    attSet = set(attributeList)
    attTrySet = indexSet - attSet #attributes not in list already; need to try them out
    attTry = list(attTrySet) #to list since we need to iterate over this
    errorList = []
    attTemp = []
    # now try each attribute not in set to see which one gives least oos error
    for iTry in attTry:
        attTemp = [] + attributeList # a way of copying so that they don't point to same memory address
        attTemp.append(iTry)
        # use attTemp to form training and testing sub matrices as list of lists
        xTrainTemp = xattrSelect(xListTrain, attTemp) # attTemp has indices of the columns you want now
        xTestTemp = xattrSelect(xListTest, attTemp)
        xTrain = np.array(xTrainTemp)
        xTest = np.array(xTestTemp)
        yTrain = np.array(labelsTrain)
        yTest = np.array(labelsTest)

        wineQModel = linear_model.LinearRegression()
        wineQModel.fit(xTrain, yTrain)
        rmsError = np.linalg.norm((yTest - wineQModel.predict(xTest)), 2)/sqrt(len(yTest)) # OOS RMSE
        errorList.append(rmsError)
        attTemp = []

    iBest = np.argmin(errorList)
    attributeList.append(attTry[iBest])
    oosError.append(errorList[iBest])

print("OOS error vs. attribute set size: ")
print(oosError)
print("\nBest attribute indices: ")
print(attributeList)
namesList = [names[i] for i in attributeList]
print("\nBest attribute names: ")
print(namesList)

# plotting
x = range(len(oosError))
plt.plot(x, oosError, 'k')
plt.xlabel("Number of attributes")
plt.ylabel("RMS Error")
plt.show()

# with best number of attributes, more analysis
indexBest = oosError.index(min(oosError))
print(indexBest)
attributesBest = attributeList[:(indexBest+1)] # why 1: ? should it not be 0: ?
print(attributesBest)

# now retrain for all data with these columns
xTrainTemp = xattrSelect(xListTrain, attributesBest)
xTestTemp = xattrSelect(xListTest, attributesBest)
xTrain = np.array(xTrainTemp)
xTest = np.array(xTestTemp)

wineQModel = linear_model.LinearRegression()
wineQModel.fit(xTrain, yTrain)
errorVector = yTest - wineQModel.predict(xTest)
print("RMSE: ", (np.linalg.norm(errorVector, 2)/sqrt(len(yTest))))
plt.hist(errorVector)
# sometimes, the error histogram will have two or more distinct peaks. Sometimes, it could have a small peak on the far right or far left of the graph. In that case, it might be possible to understand why this happens and we could also add in an attribute that explains the membership in one or the other of the groups of points.
plt.xlabel("Bin boundaries")
plt.ylabel("Counts")
plt.show()

# scatter plot of actual vs. predicted
plt.scatter(wineQModel.predict(xTest), yTest, s=100, alpha = 1.00)
# When the true values take on a small number of values like in this case, it is useful to make the data points partially transparent so that the darkness can indicate the accumulation of many points in one area of the graph. Here, actual taste scores at 5 and 6 are predicted well, but not at the edges. Usually, ML algos don't perform well near edges of data set.
plt.xlabel('Predicted Taste Score')
plt.ylabel('Actual Taste Score')
plt.show()

'''
This method trains a family of models (parameterized, since these depend on the number of columns/ attributes and which all - complexity parameter). Models with more complexity parameter - more likely to overfit.
Important ML method: Have the attributes come out of the process in descending order of their usefulness like in this method. Desirable to understand the importance of attributes relative to each other as well.
Here, from multiple iterations, we have picked the one with the least RMSE OOS error. If you notice carefully here, the 9th model (best) and 10th model has only a small change in the 4th significant digit of the RMSE OOS error. But, adding in a 10th attribute will increase model complexity. Hence, it is better to be conservative and remove those attributes.
'''
