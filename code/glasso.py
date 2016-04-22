import numpy as np
import scipy.linalg as spl
from copy import copy

### One norm
# x is a n x 1 matrix
def onenorm(x):
    return sum(np.apply_along_axis(lambda x: np.absolute(x),1,x))[0]

### Two norm
# x is any numpy.matrix
def twonorm(x):
    return np.sqrt(np.sum(np.square(x)))

### LASSO Objective (admm form)
# X is an N x p matrix
# y is an N x 1 matrix
# beta is a p x 1 matrix
# z is a p x 1 matrix
# lambduh is a positive constant
# rho is a positive constant
#a is a p x 1 matrix
# output is a real Number

def admmObj(X,y,beta,z,lambduh,rho,a):
    return twonorm(y-X*beta) + lambduh*onenorm(z) + (a.T*(beta - z))[0,0] + rho/2*twonorm(beta-z)
    
### ADMM Solver
# X is an N x p matrix
# Y is an N x 1 matrix
# lambda is a positive regularization constant
# rho is a positive number
# output is a p x 1 matrix
def admm(X,y,lambduh,rho):
    tol = 1e-10
    p = X.shape[1]
    xtxpi = (X.T * X + rho * np.matrix(np.identity(p))).I
    oldbeta = np.matrix([[0.0]]*p)
    beta = np.matrix([[1.0]]*p)
    z = np.matrix([[1.0]]*p)
    a = np.matrix([[1.0]]*p)
    iteration = 0
    while iteration < 100 and twonorm(oldbeta - beta) > tol:
        oldbeta = copy(beta)
        beta = xtxpi * X.T * y + rho * xtxpi  * z -  xtxpi *a
        for i in range(p):
            z[i,0] = np.sign(beta[i,0] + 1.0/rho*a[i,0])*max(0.0,(np.absolute(beta[i,0] + 1.0/rho*a[i,0])-lambduh[0,i]/rho))
        a += rho*(beta - z)
        iteration += 1
    return beta
  
### Matrix One Norm
def matrixonenorm(M):
    return sum(np.apply_along_axis(lambda z: reduce(lambda x,y: x + np.absolute(y),
                                                z,0), 1, M))
        
### Drops ith row / column as in the matrix M
# M is n x n matrix
# returns n-1 x n-1 matrix
def dropi(M,i):
    return np.delete(np.delete(M,i,0),i,1)

### Glasso Objective
# Omega is a d x d matrix
# S is a d x d matrix

def glassObj(Omega,S,lambduh):
    return -np.log(np.linalg.det(Omega)) + np.trace(S * Omega) + lambduh * matrixonenorm(Omega)

### GLasso solver
# S is the empirical covariance matrix of X
def glassoSolve(S,lambduh,rho):
    tol = 1e-16
    d = S.shape[0]
    oldSigma = np.matrix(np.identity(d))
    Sigma = S + np.multiply(lambduh, np.matrix(np.identity(d)))
    iteration = 0
    while iteration < 50 and twonorm(Sigma - oldSigma) > tol:
        oldSigma = copy(Sigma)
        for i in range(d):
            SigmaMinusI = dropi(Sigma,i)
            X = np.matrix(spl.sqrtm(SigmaMinusI))
            y = np.matrix(spl.sqrtm(SigmaMinusI)).I * np.delete(S[0:,i],i).T
            beta = admm(X,y,np.delete(lambduh[0:,i],i),rho)
            sigmadot = SigmaMinusI * beta
            Sigma[0:,i] = np.insert(sigmadot,i,S[i,i] + lambduh[i,i],0)
        iteration += 1
    return Sigma.I
    


if __name__ == "__main__":
### Loading Glasso Data
    data = np.loadtxt("glasso_data.csv",delimiter=",")
    means = np.array([0.0]*5)
    for i in range(1000):
        means += data[i]
    means = means/1000.0
    centerData = []
    for i in range(1000):
        centerData.append(data[i] - means)

    centerData = np.matrix(centerData)
    Cov = centerData.T * centerData /1000

    lambduh= np.matrix(np.array([[0.1]*5]*5))

    out = glassoSolve(Cov,lambduh,1.0)

    print np.around(out,3)




