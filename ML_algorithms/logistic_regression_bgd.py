# logistic regression with Batch Gradient Descent

import numpy as np

# prepare X matrix
m = 1000 # number of training instances
X = np.random.normal(0, 1, 2*m) # mean=0, sigma=1, m training instances
X = X.reshape((m, 2)) # just making sure of dims, otherwise it is (1000,) - it should be a 2D matrix
print(X.shape)
# we need an X concatenated with 1s as well for the bias term
X = np.c_[X, np.ones(m)] # note that this is not a callable
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

thr = 0.5
# function to compute the cost/ log-likelihood
def neg_log_like(theta, x, y):
    g = logistic(theta, x)
    return -sum(g[y>thr]) -sum(1-g[y<thr])

# function to compute gradient of neg-log-likelihood
def log_grad(theta, x, y):
    g = logistic(theta, x)
    return -x.T.dot(y-g)

# implementation of BGD
def grad_desc(theta, x, y, alpha, tol, maxiter):
    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (nll_delta > tol) and (iter < maxiter):
        theta = theta - (alpha*log_grad(theta, x, y))
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2] - nll_vec[-1]
        iter += 1
    return theta, np.array(nll_vec)

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
