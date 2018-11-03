# Find out the age of an abalone by using factors other than rings in their shells (that is too expensive)

#!/usr/bin/python2
from __future__ import print_function
__author__ = 'anirudha sundaresan'

import urllib2
import sys
import pprint
import pdb
import numpy as np
import pandas as pd

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
data = urllib2.urlopen(target_url)

''' No need of this below as read_csv takes in the link itself.
with open("abalone.csv", 'w') as ifile:
    for line in data:
        ifile.write(line)
# data is in csv format; we need it in DataFrame format.
abalone = pd.read_csv("abalone.csv")
'''

'''
abalone = pd.read_csv(target_url, header=None, prefix="V")
The previous rocksVMines had generic column names and hence were not important, but here, getting to know them will give us information and will aid our intuition while experimenting different methods.

Also, first column is categorical variables. M/F/I. Some algorithms only deal with real-valued attributes, eg: SVM, KNN, PLM. Chapter 4 - how to transform categorical to real-valued variables.
'''

abalone = pd.read_csv(target_url, header=None, prefix="V")
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight','Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

print(abalone.head())
print(abalone.tail())

summary = abalone.describe()
print(summary) # summary does not include the 'Sex' column. It automatically neglects categorical entries.

# Use boxplots:
from pylab import *
import matplotlib.pyplot as plt
from pandas import DataFrame

# abalone.iloc[:,1:9] is still a dataframe.
array = abalone.iloc[:,1:9].values # returns a Numpy representation of the df.
boxplot(array)
plt.xlabel("Attribute index")
plt.ylabel("Quartile Ranges")
plt.show()

# Since the last column has values much larger in range than the ones in other columns, we can remove it and plot again.
array = abalone.iloc[:, 1:8].values
boxplot(array)
plt.show()

# This got us to see data in relative knowledge, but instead of removing the column, we can also normalize each column by centering the values around the mean of the column and subtracting the sd from it.
# Thus, renormalize columns to 0 mean and unit SD (important for Kmeans clustering/ KNN).

abalone_N = abalone.iloc[:,1:9]
for i in range(8):
    mean = summary.iloc[1,i] # summary is also df; and it did not have M/F/I summary column.
    sd = summary.iloc[2,i]
    abalone_N.iloc[:,i] = (abalone_N.iloc[:, i] - mean)/sd
    # centering and scaling

print(abalone_N)
array3 = abalone_N.values
boxplot(array3)
plt.xlabel("Attribute index")
plt.ylabel("Quartile Ranges: Normalized")
plt.show()

''' How to read box plots?
median line - represented inside the rectangle
25th percentile - bottom of rectangle
75th percentile - top of rectangle
2 small horizontal lines on either side of rectangle at 1.4 times the interquartile spacing (distance between 25th and 75th percentile - height of rectangle) from the ends of the rectangle. This is adjustable. Box plots are good for viewing outliers (values beyond the 2 small horizontal lines).
'''

# Get some relations among attributes and between attributes and labels. In rocksVMines case, we used color-coded parralel coordinates plot. That was for classification. Ours is a regression problem. Here, plot all numerical variables except Rings, since that is what we need to do regression on.
# In the rocksVMines case, we assigned blue and red to 2 classes, here we assign different shades of colors to different rings.

# Plotting with coarse color division:
minRings = summary.iloc[3,7]
maxRings = summary.iloc[7,7]
nrows = len(abalone.index)
'''
for i in range(nrows):
    print(i)
    dataRow = abalone.iloc[i,1:8] # 0 not used and normalized abalone is not used. Why? (because it has -ve values? )
    # plot them as if they were series data
    labelColor = (abalone.iloc[i,8] - minRings)/ (maxRings - minRings)
    dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

plt.xlabel("Attribute index")
plt.ylabel("Attribute values")
plt.show()
'''
''' takes too much time:
# Plotting with finer color division by using logit finction
from math import exp
mean_rings = summary.iloc[1,7]
sd_rings = summary.iloc[2,7]

for i in range(nrows):
    dataRow = abalone.iloc[i, 1:8]
    normTarget = (abalone.iloc[i,8] - mean_rings)/sd_rings
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)
plt.xlabel("Attribute index")
plt.ylabel("Attribute values")
plt.show()
'''

# Logit function compresses any input to a value between 0.0 and 1.0. This utilizes the full range of colors possible.

# Heat maps for correlation between the attributes: Regression case.
corMat = DataFrame(abalone.corr())
# corr will ignore categorical column, I think. In textbook, it is abalone.iloc[:,1:9]
# Note: Here, in contrast to classification problems, we include the variable to be predicted (rings) also for correlation as it is numeric.
print(corMat)
plt.pcolor(corMat)
plt.show()

# From heat map, red indicates high correlation and blue - weak correlation. Looks like the targets are weakly correlated with all attributes.

