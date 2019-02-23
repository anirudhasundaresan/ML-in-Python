import numpy as np
import random

random.seed(42)
# prepare datasets
m = 1000
X = np.random.normal(0, 1, 2*m).reshape(m, 2)
X = np.c_[X, np.ones(m)]
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

def neg_log_like(theta, x, y):
    g = logistic(theta, x)
    return -sum(g[y>0.5]) -sum(1-g[y<0.5])

def log_grad(theta, x, y):
    g = logistic(theta, x)
    return -x.T.dot(y-g)

def grad_desc(theta, x, y, maxiter, alpha, tol):
    nll_vec = []
    iter = 0
    nll_vec.append(neg_log_like(theta, x, y))
    nll_del = 2.0*tol
    inds = list(range(m))
    np.random.shuffle(inds) # this is in-place operation
    num_mb = 10 # 10 mini-batches
    while (iter < maxiter) and (nll_del > tol):
        k = 0
        for i in range(num_mb):
            theta = theta - alpha*log_grad(theta, x[inds[k:k+100]], y[inds[k:k+100]])
            k += 1
            nll_vec.append(neg_log_like(theta, x, y))
            nll_del = nll_vec[-2] - nll_vec[-1]
        iter += 1
    return theta, nll_vec

# to predidct from a new output
def lr_predict(theta, x):
    shape = x.shape
    xtilde = np.zeros((shape[0], shape[1]+1))
    xtilde[:,0] = x
    xtilde[:,1:] = np.ones(shape[0])
    return logistic(theta, x)

alpha = 0.05
tol = 0.001
maxiter = 1000
theta, cost = grad_desc(theta, X, y, alpha, tol, maxiter)
