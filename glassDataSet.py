#!/usr/bin/python2

# This is a multi-class classification problem; determine the type of glass in a forensic scene: whether it is from windows/ glass containers, etc.
# This is not a regression problem because there is no order relation between the different target labels.

from __future__ import print_function
__author__ = 'anirudhasundaresan'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pprint
from pylab import *
from pandas import DataFrame

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")
glass = pd.read_csv(target_url, header=None, prefix='V')

print("Glass df before adding columns: ")
print(glass.head())
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

print("Glass df after adding columns: ")
print(glass.tail())
summary = glass.describe()
print("Summary of Glass df: ")
print(summary)

ncol1 = len(glass.columns)
glass_N = glass.iloc[:,1:ncol1] # since 1st column is Id column - we don't need that/ not useful.
ncol2 = len(glass_N.columns)
print("Glass_N after getting from Glass df: ")
print(glass_N.head())

for i in range(ncol2):
    mean = summary.iloc[1,i+1]
    sd = summary.iloc[2,i+1]
    glass_N.iloc[:, i:(i+1)] = (glass_N.iloc[:, i:(i+1)] - mean)/sd

print("Normalized glass: ")
print(glass_N.head())

# Like always, boxplots need normalized values and needs to be converted to array before plotting.

array = glass_N.values
boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges - Normalized")
plt.show()

# Seems like there are a lot of outliers. Check for class imbalance?
print(pd.value_counts(glass['Type'].values)) # we see that there is some imbalance. So, there can be outliers. Also, we should definitely not expect all the data to be in a single group, else this won't be a multi-class classification problem. 76 for most populous to 9 for the least populous.
# NO reason to expect proximity across classes - that's what makes this a multi-class classification problem. The average stats are dominated by the members of the most populous classes.
# Ensemble methods will perform better here than PLM because ensemble methods can create complex decision boundaries.

# Try the parralel coordinates plot? --> like in rocksVMines and using only normalized values.
nrows = len(glass.index)
for i in range(nrows):
    dataRow = glass_N.iloc[i, 1:ncol1-1]
    # in the book, they have not printed the final labels, but it is better to graphically show the labels as well, gives more accurate information.
    labelColor = glass.iloc[i, ncol1-1]/7.0
    dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

plt.xlabel("Attribute Index")
plt.ylabel(("Attribute values"))
plt.show()

corMat = DataFrame(glass.iloc[:,1:-1].corr())
# 1st column is id column, hence neglecting that
# last column is target label column, so neglect that
plt.pcolor(corMat)
plt.show()

# Mostly blue in this heat map, hence low correlation. Attributes are thus independent of each other, which is good. Targets are not included in the heat map since they take discrete values and they rob the heat map of their explanatory powers.

# From heat maps and parallel coordinates plot, we see that there seems to be some complicated boundary that separates these classes; ensemble methods will work better with more data here.
