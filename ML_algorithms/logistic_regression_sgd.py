import numpy as np
import random

np.random.seed(0)
# prepare datasets
m = 1000
X = np.random.normal(0, 1, 2*m).reshape(m, 2)
X = np.c_[np.ones(m), X]
# print(X.shape) # (1000, 2)

y = np.random.randint(2, size=m).reshape(m, 1)
# print(y.shape) # (1000, 1)

theta = np.zeros(3).reshape(3,1)

def logistic(theta, x):
    t = x.dot(theta)
    g = np.zeros(t.shape)
    g[t>0.0] = 1/(1.0 + np.exp(-t[t>0.0]))
    g[t<0.0] = np.exp(t[t<0.0])/(1.0 + np.exp(t[t<0.0]))
    return g

thr = 0.5
def neg_log_like(theta, x, y):
    g = logistic(theta, x)
    eps = 0.000000000001
    return -sum(np.log(g[y>thr]+eps)) -sum(np.log(1-g[y<thr]+eps))

def log_grad(theta, x, y):
    g = logistic(theta, x)
    return -x.T.dot(y-g)

def grad_desc(theta, x, y, maxiter, alpha, tol):
    nll_vec = []
    iter = 0
    nll_vec.append(neg_log_like(theta, x, y))
    nll_del = 2.0*tol
    inds = list(range(m))
    num_mb = m # m mini-batches
    while (iter < maxiter) and (nll_del > tol):
        np.random.shuffle(inds) # this is in-place operation
        for i in range(m):
            theta = theta - alpha*log_grad(theta, x[inds[i]].reshape(1,3), y[inds[i]].reshape(1,1))
            nll_vec.append(neg_log_like(theta, x, y))
            nll_del = abs(nll_vec[-2] - nll_vec[-1])
        iter += 1
    return theta, nll_vec

# to predidct from a new output
def lr_predict(theta, x):
    shape = x.shape
    xtilde = np.zeros((shape[0], shape[1]+1))
    xtilde[:,1:] = x
    xtilde[:,0] = np.ones(shape[0])
    return logistic(theta, x)

alpha = 0.0001
tol = 0.000001
maxiter = 10000
theta, cost = grad_desc(theta, X, y, maxiter, alpha, tol)
print(theta)
