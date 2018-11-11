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

