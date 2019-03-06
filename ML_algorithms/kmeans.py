#  https://stats.stackexchange.com/questions/188087/proof-of-convergence-of-k-means
import copy
import random
import numpy as np
from matplotlib import pyplot as plt

random.seed(42)

# create clusterwise set of points
'''
data = np.array([[ 1.23571758,  1.47666206],
                 [ 0.40451215, -0.01062741],
                 [ 1.71635348,  0.83296132],
                 [ 0.84367405,  1.00105918],
                 [ 0.63970563,  1.20272671],
                 [-1.05641853, -1.35545403],
                 [-1.07020579, -0.8394209 ],
                 [-1.81826175, -2.27345278],
                 [-1.99607591,  1.89867684],
                 [-2.56067124,  1.67201533],
                 [-1.71249107,  2.09671069],
                 [-1.75201349,  2.27671946]])

'''
'''
plt.scatter(data[:, 0], data[:, 1])
plt.show()
'''

data1 = np.random.normal(0, 1, 8).reshape(4, 2)
data2 = np.random.normal(1, 1, 8).reshape(4, 2)
data3 = np.random.normal(2, 1, 8).reshape(4, 2)

data = np.vstack((data1, data2, data3))
plt.scatter(data[:, 0], data[:, 1])
plt.show()

def kmeans(data, k=3):
    rows, cols = data.shape
    colmin = np.amin(data, axis=0)
    colmax = np.amax(data, axis=0)
    centers = np.random.uniform(low=colmin, high=colmax, size=(k, cols))
    assignments = np.zeros(data.shape[0])
    orig = assignments.copy()
    while True:
        for ind, pt in enumerate(data):
            assignments[ind] = np.argmin([np.linalg.norm(pt-cent) for cent in centers])
        print(assignments)
        for ind, cent in enumerate(centers):
            print(centers)
            centers[ind] = sum(data[assignments==float(ind)])/len(data[assignments==float(ind)])
        if np.all(orig == assignments):
            return assignments
        else:
            orig = assignments.copy()
        plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(centers[:, 0], centers[:, 1], color='red')
        plt.show()

print(kmeans(data, 3))
