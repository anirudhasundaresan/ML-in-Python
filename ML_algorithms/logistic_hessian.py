# https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
# https://www.quora.com/What-is-the-Hessian-matrix-What-is-it-used-for-and-for-what-reason

# instead of taking the derivative wrt components, we will work directly with vectors

# logistic regression with Batch Gradient Descent

import numpy as np
import random
np.random.seed(0)

# prepare X matrix
m = 1000 # number of training instances
X = np.random.normal(0, 1, 2*m) # mean=0, sigma=1, m training instances
X = X.reshape((m, 2)) # just making sure of dims, otherwise it is (1000,) - it should be a 2D matrix
print(X.shape)
# we need an X concatenated with 1s as well for the bias term
X = np.c_[np.ones(m), X] # note that this is not a callable
print(X.shape)

# prepare Y matrix
y = np.random.randint(2, size=m)
y = y.reshape((m, 1))
print(y.shape)

# prepare weight matrix
theta = np.zeros(3).reshape((3, 1))

# define the sigmoid/ logistic function
def logistic(theta, x):
    t = x.dot(theta)
    g = np.zeros(t.shape)
    g[t>=0.0] = 1.0/(1.0 + np.exp(-t[t>=0.0]))
    g[t<0.0] = np.exp(t[t<0.0])/(1.0 + np.exp(t[t<0.0]))
    return g

def hessian(theta, x, n):
    tot = np.empty((3,3))
    tot.fill(0)
    for i in range(n):
        tot += ((np.outer(x[i],((x[i]).T)))*(logistic(theta,x[i]))*(1 - logistic(theta,x[i])))
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.outer.html
    return tot

thr = 0.5
# function to compute the cost/ log-likelihood
def neg_log_like(theta, x, y):
    g = logistic(theta, x)
    eps = 0.00000000000001
    return -sum(np.log(g[y>thr] + eps)) -sum(np.log(1-g[y<thr]+eps))

# function to compute gradient of neg-log-likelihood
def log_grad(theta, x, y):
    g = logistic(theta, x)
    return -x.T.dot(y-g)

# implementation of BGD
def grad_desc(theta, x, y, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        hss = np.linalg.inv(hessian(theta, x, m)) # new statement - also passing number of instances
        theta = theta - (hss.dot(log_grad(theta, x, y))) # theta = theta - (alpha*log_grad(theta, x, y))
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = abs(nll_vec[-2] - nll_vec[-1])
        iter += 1
    return theta, np.array(nll_vec)

# to predidct from a new output
def lr_predict(theta, x): # x is a 2d vector here without appending it with 1s
    shape = x.shape
    xtilde = np.zeros((shape[0], shape[1]+1))
    xtilde[:,1:] = x
    xtilde[:,0] = np.ones(shape[0])
    return logistic(theta, x)

tol = 0.000001
maxiter = 10000
theta, cost = grad_desc(theta, X, y, tol, maxiter)
print(theta)
