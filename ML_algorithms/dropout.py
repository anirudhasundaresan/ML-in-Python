#  https://iamtrask.github.io/2015/07/28/dropout/
#  Good crash course - https://www.youtube.com/watch?v=kK8-jCCR4is

import numpy as np
X = np.array([[0,1,1],[0,1,0],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T
alpha, hidden_dim, dropout_percent, do_dropout = 0.05, 5, 0.4, True
keep_percent = 1 - dropout_percent
w0 = 2*np.random.random((3,hidden_dim)) - 1 #  so that both +ve and -ve numbers are represented
w1 = 2*np.random.random((hidden_dim,1)) - 1
maxiter = 60000
for j in range(maxiter):
    layer_1 = (1/(1+np.exp(-(np.dot(X, w0)))))
    if(do_dropout):
        layer_1 *= np.random.binomial([np.ones((len(X),hidden_dim))],1-dropout_percent)[0] * (1.0/(keep_percent))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,w1))))
    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
    layer_1_delta = layer_2_delta.dot(w1.T) * (layer_1 * (1-layer_1))
    w1 -= (alpha * layer_1.T.dot(layer_2_delta))
    w0 -= (alpha * X.T.dot(layer_1_delta))
    # print(w0, '\n')
    # print(w1, '\n')
print (w0, w1)
