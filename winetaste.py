#!/usr/bin/python2

from __future__ import print_function
__author__ = 'anirudha sundaresan'

import urllib2
import sys
import pprint
import pdb
import numpy as np
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plt

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
wine = pd.read_csv(target_url, header=0, sep=";")

print(wine.head())
summary = wine.describe()
print(summary)
tasteCol = len(summary.columns)

wine_N = wine.copy() # normalized
# initially, did not have the .copy() --> it is necessary; else, wine_N when changed will change wine df.

ncols = len(wine_N.columns)
nrows = len(wine.index)

for i in range(ncols):
    mean = summary.iloc[1,i]
    sd = summary.iloc[2,i]
    wine_N.iloc[:, i:(i+1)] = (wine_N.iloc[:, i:(i+1)] - mean)/sd

array = wine_N.values # always get df normalized and into an array before boxplot
boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges - Normalized")
plt.show()

# To use boxplots, we need to normalize them first because we need them all in scale for comparison.
# But, for getting the parallel coordinates plot, this is not a necessity, but here we see that it is necessary since the data is not scaled well.

meanTaste = summary.iloc[1, tasteCol - 1]
sdTaste = summary.iloc[2, tasteCol - 1]
nDataCol = len(wine.columns) - 1

for i in range(nrows):
    dataRow = wine.iloc[i,1:nDataCol]
    normTarget = (wine.iloc[i,nDataCol] - meanTaste)/sdTaste
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

for i in range(nrows):
    dataRow = wine_N.iloc[i,1:nDataCol]
    normTarget = wine_N.iloc[i,nDataCol] # - meanTaste)/sdTaste not needed as they are already normalized.
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

# dark blue lines - high taste scores --> High values of alcohol indicates good taste.
# dark red lines - low taste scores --> High values of volatility indicates poor taste.
# this is opposite to heat map

corMat = DataFrame(wine.corr())
plt.pcolor(corMat)
plt.show()

# shows relatively high correlation between last column (taste) and alcohol (next-to-last column). Also, volatile acidity has low correlation with taste.
# Note: the wine tastes are integers from 3 to 8. This is however, not a multi-class classification problem because there is an order between the tastes. 3 means poor taste and 8 means great taste.
# But, in the case of multi-class classification, so such order exists.

