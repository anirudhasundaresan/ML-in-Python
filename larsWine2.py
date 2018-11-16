__author__ = 'anirudhasundaresan'

import requests
import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt
import pdb

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = (requests.get(target_url)).text

# pulling data into labels and lists of lists.
xList, labels = [], []
firstLine = True # for collecting headers

for line in data.splitlines():
    if firstLine:
        names = line.strip().split(';')
        firstLine = False
    else:
        row = line.strip().split(';')
        labels.append(float(row.pop()))
        xList.append([float(num) for num in row])

# Normalizing columns is always a good step in PLRM.
# Normalize columns in x and labels:

nrows = len(xList)
ncols = len(xList[0])

# calculate means and variances for each column of XList.
xMeans = []
xSD = []

for i in range(ncols):
    mean = sum([xList[j][i] for j in range(nrows)])/nrows
    xMeans.append(mean)
    colDiff = [xList[j][i]-mean for j in range(nrows)]
    sd = sqrt(sum([colDiff[i]*colDiff[i] for i in range(nrows)])/nrows)
    xSD.append(sd)

# normalize xList using means and SD calculated for each column.
xNormalized = [[(xList[i][j]-xMeans[j])/xSD[j] for j in range(ncols)] for i in range(nrows)]

# you need to normalize the labels also
meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([(labels[i]-meanLabel)*(labels[i]-meanLabel) for i in range(nrows)])/nrows)
labelNormalized = [(labels[i] - meanLabel)/sdLabel for i in range(nrows)]

# init betas, one for each column
beta = [0.0] * ncols

# for each step, you will get a beta coefficient vector, so store these in a beta matrix.
betaMat = []
betaMat.append(list(beta))

# define number of steps and stepsize
nSteps = 350
stepSize = 0.004

for i in range(nSteps):
    # calculate residuals
    residuals = [0.0] * nrows
    for j in range(nrows):
        # for each row, find the residual
        labelsHat = sum([beta[k]*xNormalized[j][k] for k in range(ncols)])
        residuals[j] = labelNormalized[j] - labelsHat

    # get correlation between attribute cols from normalized wine and residual
    corr = [0.0] * ncols
    for j in range(ncols):
        corr[j] = sum([xNormalized[k][j] * residuals[k] for k in range(nrows)])/nrows
        # actually, correlation btw 2 vars is the product of their variations from their means normalized by their standard deviations. Here, we did not do that since the attributes are already normalized and because their SD = 1 and also because the resulting values are going to be used to find the biggest correlation and multiplying all the values by a constant won't change that otder.

    iStar, corrStar = 0, corr[0]
    for j in range(1, ncols):
        if abs(corrStar) < abs(corr[j]):
            iStar = j; corrStar = corr[j] # to see which attribute affects labels/ residuals the most

    beta[iStar] += stepSize * corrStar/abs(corrStar) # changing the coefficient of the attribute that affects residual the most by a stepsize in +/- direction.
    # each of the steps in LARS increment one of the betas by a fixed account. Thus, each beta would mean differently if the scales of the columns are all different; hence they're normalized.
    # And because the algo is running on normalized data, there is no need for beta0. Since all the attributes are normed to 0 mean, they don't have any offset.
    # betaMat will show how for each step, beta changes.
    # print(i)
    # print(beta)
    betaMat.append(list(beta))
    # Note that this is also another version of RR with lambda parameter; As we populate the beta vector, the sum of squares of betas also increase, this is equivalent to increasing lambda.

for i in range(ncols):
    # plot range of beta values for each attribute
    coefCurve = [betaMat[k][i] for k in range(nSteps)]
    plt.plot(range(nSteps), coefCurve) # holds and plots for all cols

plt.xlabel("Steps taken")
plt.ylabel("Coefficient values")
plt.show()

'''
The plot shows how LARS is Lasso in some sense. As we see, iniitally, for the first 25 steps or so, most of the coefficients are 0. This is a sparse solution of the LARS, just like Lasso. As the no. of steps increases, more variables will get non-zero betas. The order in which betas become non-zero correspond to the importance of the variables. If you want to leave out variables, leave out the ones for which beta becomes non-zero later.
PLRM - important property is the ability to indicate the importance of variables. - Feature engineering. Not all ML models give this information.
'''
