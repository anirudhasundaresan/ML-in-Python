# KNN is an instance based classification/ regression method - lazy learning mechanism
from collections import Counter
import numpy as np
X = np.random.normal(0, 1, 200).reshape(100, 2)
y = np.random.randint(2, size=100)

# print(X.shape, y.shape)

# do cross-val to decide on the value of k
inds = list(range(100))
np.random.shuffle(inds)

# doing simple validation - not crossval here
X_train, X_test = X[inds[:80]], X[inds[80:]]
y_train, y_test = y[inds[:80]], y[inds[80:]]
final_res = []

def knn(k=3):
    res = []
    for ind, x in enumerate(X_test):
        dist = [np.linalg.norm(x-xtrain) for xtrain in X_train]
        corr_class = 0
        for i in np.argsort(dist)[:k]:
            if y_test[ind] == y_train[i]:
                corr_class += 1
            else:
                corr_class -= 1
        if corr_class < 0 :
            res.append(False) # misclassified
        else:
            res.append(True)
    final_res.append(res.count(False)) # checking loss
    return final_res

for i in range(1, 20, 2):
    knn(i)
    print(i, final_res)
