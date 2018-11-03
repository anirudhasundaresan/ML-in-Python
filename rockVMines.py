#!/usr/bin/python2

from __future__ import print_function

__author__ = 'anirudha_sundaresan'
import urllib2
import sys

# read data from the UCI repo
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
        "databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = urllib2.urlopen(target_url)

xList, labels = [], []

for line in data:
    row = line.strip().split(',')
    xList.append(row)

print("Number of rows: ", len(xList))
print("Number of columns: ", len(xList[0]))

# Till now, basic exploration; next, assess which columns are categorical/ numerical

for col in range(len(xList[0])):
    type_ = [0, 0, 0]
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type_[0] += 1
        except ValueError:
            if len(row[col]) > 0: # smart way of checking for string
                type_[1] += 1
            else:
                type_[2] += 1
    print("Column: ", col, "and counts: ", type_)

# Summary stats for the attributes and labels (to get some intuition about the dataset)

# Starting with column 3:
# Basic mean and std

import numpy as np
col = 3
colData = []
for row in xList:
    colData.append(float(row[col])) # don't forget the float

colArray = np.array(colData) # always better to work with Numpy arrays; all the Numpy functions act on Numpy arrays and are faster.
colMean = np.mean(colArray)
colsd = np.std(colArray)

print("For column 3: Mean = ", colMean, " and Standard Deviation = ", colsd)

# Note: Be careful about splicing list of lists in Python. Slicing is actually meant for lists only, not when axes are involved. Better to convert to numpy and deal with slicing.

# Calculate quantile boundaries: this is to detect and visualize outliers. Range each quartile will reveal this information.
ntiles = 4 # running with 4 equal intervals
percentBdry = []

for i in range(ntiles+1): #[0,1,2,3,4]
    percentBdry.append(np.percentile(colArray, (i*100/ntiles)))

print("Boundaries for 4 equal percentiles: ", percentBdry)

# Visualize the outliers of column 3 (values held by colData)
import scipy
import pylab
import scipy.stats as stats
stats.probplot(colData, dist='norm', plot=pylab)
pylab.show() # quantile-quantile plot
# this gives an idea about which outliers need to be kept in mind while building/ training models.

# Find the distribution of the categorical variables:
cat_ = []
for row in xList:
    cat_.append(row[-1])

from collections import Counter
print("Distribution of labels: ", Counter(cat_)) # can be used to check if there is a balanced set or not.


# using Pandas to get a feel for the data; much simpler
import pandas as pd
from pandas import DataFrame
rocksVMines = pd.read_csv(target_url, header=None, prefix="V") # good for giving generic column names

# print(rocksVMines) # will display all the values
print(rocksVMines.head())
# this shows that the 'R' categories are in the head and the 'M' are in the tail. While sampling the data, extra care must be needed to take into account the structure of the data in the dataset.
print(rocksVMines.tail())
print(rocksVMines.describe()) # the summary variable is also a pandas df object. Hence, easy to manipulate later on.

# From the quantile data in the summary of pandas; observe differences between the values in the quantiles within each attribute; if there is a drastic difference, there is chance of outliers.

# visualizing with Parallel coordinate plots: Here, each row can be plotted for a certain set of attributes
import matplotlib.pyplot as plt

for i in range(208):
    if rocksVMines.iat[i, 60] == 'M': # use 'iat' to get a single value from a DataFrame or Series.
        pcolor = 'red'
    else:
        pcolor = 'blue'

    dataRow = rocksVMines.iloc[i, 0:60] # plot them as if they were series data
    dataRow.plot(color=pcolor) # notice the plot attribute for the series data.

# iat vs. iloc: 'i' stands for integer based lookups. In loc and at, it is label-based lookup.
# Use iloc instead of iat for slicing and extracting more than a row.

plt.xlabel("Attribute Index")
plt.ylabel("Attribute values")
plt.show() # index 35 looks like it differentiates between 'M' and 'R' more easily than the others.

# We could also visualize how the attributes are related to one another.
# Note: SONAR data is stored here in the dataset in the way it was recorded. Adjacent rows have similar frequencies. These frequencies rise over time. Thus, graphically, we should expect strong correlation between adjacent rows than row1 and row20.
dataRow2 = rocksVMines.iloc[1, 0:60]
dataRow3 = rocksVMines.iloc[2, 0:60]
dataRow21 = rocksVMines.iloc[20, 0:60]

plt.scatter(dataRow2, dataRow3)
plt.xlabel("2nd attribute")
plt.ylabel("3rd attribute")
plt.show()

plt.scatter(dataRow2, dataRow21)
plt.xlabel("2nd attribute")
plt.ylabel("21st attribute")
plt.show()

# Index 35 to be compared with the targets.
dataRow = rocksVMines.iloc[:,35]
target = [1.0 if rocksVMines.iat[i, 60]=='M' else 0.0 for i in range(208)] # remember if...else...for
plt.scatter(dataRow, target)
plt.xlabel("Attribute value")
plt.ylabel("Target value")
plt.show()

# Visualization with more information, but with same points. It all comes down to what you know and how to use colors and shades. Try to make the previous graph more informative by adding dither and semi-opacity.
from random import uniform
target = [(1.0+uniform(-0.1, 0.1)) if rocksVMines.iat[i, 60]=='M' else (0.0+uniform(-0.1, 0.1)) for i in range(208)] # remember if...else...for
plt.scatter(dataRow, target, alpha=0.5, s=120)
plt.xlabel("Attribute value")
plt.ylabel("Target value")
plt.show() # see that the density is more on the left of the upper band. This attribute can be used as a classifier with the line drawn at maybe 0.5? It might seem that only 1 attribute (here, index 35) is enough to classify them, but the PLM and ensemble methods will use all, but arrive at giving more importance to the main attributes.

# Explore correlation values between attributes and between rows
print("Correlation between V1 and V2", rocksVMines['V1'].corr(rocksVMines['V2']))
print("Correlation between V1 and V20", rocksVMines['V1'].corr(rocksVMines['V20']), "\n")

import pprint
# these correlations use Pearson's by default
pprint.pprint(rocksVMines.corr()) # between all the columns in a dataframe
pprint.pprint(rocksVMines.iloc[:,0:60].T.corr()) # between all rows in the df, excluding the label row as they are characters.
# In a df, iloc is used to get the row and column and the indexing is different when compared to indexing of lists.
# Note that the col2 and 3 are more correlated than col 2 and 21

# Visualizing attribute and label correlations using Heat maps
# Difficult to see corrs for 100 attributes at a time. Best thing would be to get cross-correlations into matrix format and see heatmap.
corMat = DataFrame(rocksVMines.corr())
plt.pcolor(corMat)
plt.show()

# Perfect correlation = 1; if you include the same column twice. Check again.
# Very high correlation (> 0.7) between two attributes --> multi-collinearity --> unstable estimates. Be careful of these.
# If an attribute correlates with the target; then very helpful for prediction.
