__author__ = 'anirudha-sundaresan'
import urllib2
import matplotlib.pyplot as plt
from math import sqrt, cos, log

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = urllib2.urlopen(target_url)

xList = []
labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line.strip().split(";") # has alcohol in the penultimate col and quality in the last col.
        firstLine = False
    else:
        #split on semi-colon
        row = line.strip().split(";")
        #put labels in separate array
        labels.append(float(row[-1]))
        #remove label from row
        row.pop()
        #convert row to floats
        floatRow = [float(num) for num in row]
        xList.append(floatRow)

# extend the alcohol variable (last column in the xList matrix)
xExtended = []
alchCol = len(xList[1])

for row in xList:
    newRow = list(row)
    alch = row[alchCol - 1]
    # random variations to alcohol
    newRow.append((alch - 7)*(alch - 7)/10) # appending all to row
    # 7 and 10 used above so that they could fit into one plot
    newRow.append(5 * log(alch - 7))
    newRow.append(cos(alch))
    xExtended.append(newRow)

nrow = len(xList)
v1 = [xExtended[j][alchCol - 1] for j in range(nrow)] # the old alchohol values

for i in range(4):
    v2 = [xExtended[j][alchCol - 1 + i] for j in range(nrow)]
    plt.scatter(v1, v2) # comparing how just alcohol factor looks with modified-by-polynomial-factors alcohol

plt.xlabel("Alcohol")
plt.ylabel("Extension functions of alcohol") # squared, logarithmic and sinusoidal behaviour of alcohol factor
plt.show()

# nest step would be to just run models on these two: xList and xExtended - thus, one has usual factors and one has basis expanded factors.
