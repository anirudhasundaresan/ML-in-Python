# Normal LARS will give you a list of betas for each step, and how do you reliably take the best set of betas? Use CV.
# Using CV, find the betas for which error is minimum. One of the concepts that bugged me: https://stats.stackexchange.com/questions/27627/normalization-prior-to-cross-validation

__author__ = 'anirudhasundaresan'

import requests
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

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

# Now, build CV loops to determine best coefficient values.
nxval = 10
# why 10? if smaller no. of folds --> we will be training on less data; but more folds --> more calculations

nSteps = 350
stepSize = 0.004 # even this is a variable.

# init list for storing errors for each of the CV loops
errors = []
for i in range(nSteps):
    b = []
    errors.append(b)

for ixval in range(nxval):
    # define the training and test set for each CV runi
    # modulus function to split the data - sometimes when the data is unbalanced (when there is more data for 1 class and less for other), use stratified sampling; we would want the training sets to be representative of the full data set. Make sure patterns don't show up; weekly data -> don't use 5 fold CV since you will get each day's data in their respective set.
    idxTest = [a for a in range(nrows) if a%nxval == ixval*nxval] # test always has lesser elements than train, hence check for '=='
    idxTrain = [a for a in range(nrows) if a%nxval != ixval*nxval]

    # define test and training attribute and label sets
    xTrain = [xNormalized[r] for r in idxTrain]
    xTest = [xNormalized[r] for r in idxTest]
    labelTrain = [labelNormalized[r] for r in idxTrain]
    labelTest = [labelNormalized[r] for r in idxTest]

    # train LARS regression on the defined training data
    nrowsTrain = len(idxTrain)
    nrowsTest = len(idxTest)

    # init beta
    beta = [0.0] * ncols

    # init beta matrix that gets updated with beta vectors after each step
    betaMat = []
    betaMat.append(list(beta))

    for iStep in range(nSteps):
        # calculate residuals
        residuals = [0.0] * nrows
        for j in range(nrowsTrain):
            labelsHat = sum([xTrain[j][k] * beta[k] for k in range(ncols)])
            residuals[j] = labelTrain[j] - labelsHat # true-pred

        # calculate correlation between attribute columns from normalized wine and residuals
        corr = [0.0] * ncols
        for j in range(ncols):
            corr[j] = sum([xTrain[k][j] * residuals[k] for k in range(nrowsTrain)])/nrowsTrain

        iStar = 0
        corrStar = corr[0]

        # find the attribute with the max correlation for this step
        for j in range(1, ncols):
            if abs(corrStar) < abs(corr[j]):
                iStar = j; corrStar = corr[j]

        beta[iStar] += stepSize*corrStar/abs(corrStar)
        betaMat.append(list(beta))

        # use this beta to get OOS error for this step; last time without CV code, we did not even log the errors. This time, we get MSE for all the steps in a CV run and use that to find best beta
        for j in range(nrowsTest):
            labelsHat = sum([xTest[j][k]*beta[k] for k in range(ncols)])
            err = labelTest[j] - labelsHat
            errors[iStep].append(err) # will accumulate errors for each step for all the CV folds

cvCurve = [] # stores mse for each step
for errVect in errors:
    mse = sum([x*x for x in errVect])/len(errVect)
    cvCurve.append(mse)

minMse = min(cvCurve)
minPt = [i for i in range(len(cvCurve)) if cvCurve[i]==minMse][0]
print("Min MSE: ", minMse)
print("Index of min. MSE: ", minPt)

xaxis = range(len(cvCurve))
plt.plot(xaxis, cvCurve)
plt.xlabel("Steps taken")
plt.ylabel("MSE")
plt.show()

'''
The graph shows the minimum MSE at 311, but it is not a strong min. We can go ahead with using the betas of 311.
But, in the case of ambiguities, we should use the conservative approach --> more conservative for PLRMs mean the models with smaller coefficients.
CV can determind the best model complexity for the model you will deploy.
'''


