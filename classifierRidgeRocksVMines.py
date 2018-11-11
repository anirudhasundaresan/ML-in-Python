#!/usr/bin/python2

from __future__ import print_function
__author__ = 'anirudhasundaresan'

import numpy as np
import urllib2
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
from math import sqrt
import matplotlib.pyplot as plt

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data = urllib2.urlopen(target_url)

labels = []
xList = []

for line in data:
    row = line.strip().split(',')
    if row[-1]=='M':
        labels.append(1.0)
    else:
        labels.append(0.0)
    row.pop()
    xList.append([float(i) for i in row])

index = range(len(xList))

# usually, scikit learn algorithms require numpy arrays
xTest = np.array([xList[i] for i in index if i%3==0])
yTest = np.array([labels[i] for i in index if i%3==0])
xTrain = np.array([xList[i] for i in index if i%3!=0])
yTrain = np.array([labels[i] for i in index if i%3!=0])

alphaList = [0.1**i for i in [-3, -2, -1, 0, 1, 2, 3, 4, 5]]
aucList = []

for alph in alphaList:
    rocksVMinesRidgeModel = linear_model.Ridge(alpha=alph)
    rocksVMinesRidgeModel.fit(xTrain, yTrain)
    # like before, we calculate roc characteristics;
    fpr, tpr, thresholds = roc_curve(yTest, rocksVMinesRidgeModel.predict(xTest)) # what's the meaning of 'thresholds' here?
    aucList.append(auc(fpr, tpr))

print("AUC      alpha")
for i in range(len(alphaList)):
    print(aucList[i], alphaList[i]) # seems like alpha=1.0 gives best auc

# plot auc values vs. alpha values
plt.plot([-3, -2, -1, 0, 1, 2, 3, 4, 5], aucList)
plt.xlabel('-log(alpha)')
plt.ylabel('AUC')
plt.show()
# gives a visual demonstration of the value of reducing the complexity of the ordinary LS solution by imposing a constraint on the Euclidean length of the coefficient vector.

# check out performance of best classifier now:
indexBest = aucList.index(max(aucList))
alph = alphaList[indexBest]
rocksVMinesRidgeModel = linear_model.Ridge(alpha=alph)
rocksVMinesRidgeModel.fit(xTrain, yTrain)

plt.scatter(rocksVMinesRidgeModel.predict(xTest), yTest, s=100, alpha=0.25)
plt.xlabel('Predicted Value')
plt.ylabel('Actual value')
plt.show()
# Since there are discrete labels, two rows of points can be seen.

'''
We have not applied threshold to the predicted values. This is because we did not want a confusion matrix here. Also, roc_curve is a plot between the TPR and FPR. These are calculated for a number of thresholds that also get returned from the roc_curve function.
As alpha gets smaller, the RR problem becomes an OLS normal regression problem (unconstrained). In error_.py, we saw that AUC was 0.85 for OOS set, here it is so for smaller alphas indicating the previous statements. RR results in significant improvement in performance.
Is it overfitting here?
The test set has 70 examples - that's the only holdout set here. The training set has 138 rows BUT! - there are 60 attributes! The number of rows for training > 2x number of attributes, but it may stil overfit.
Better solution: Use 10-fold CV: 20 examples for a single pass holdout set. Probably will get better performance. See in ch4.
'''




