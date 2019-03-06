import numpy as np

X = np.random.normal(1, 2, 100).reshape((10, 10))
print(X)

print('\n')
X = X - X.mean(0)
print(X)

X = X/ X.std(0)

E, V = np.linalg.eig(np.dot(X.T, X))

print('\n')
print(E)
print(V)

indx = np.argsort(E)[::-1]
print(indx)
V = V[indx][:, :2]
E = E[indx]

U, sig, Vt = np.linalg.svd(X)
print('Sigma: ', sig)

print('\n', E)
print('\n V from cov', V)

print('V from svd', Vt.T[:, :2])

new_X = np.dot(X, V)
print('\n x from cov', new_X)

print('x from svd', np.dot(X, Vt.T[:, :2]))
