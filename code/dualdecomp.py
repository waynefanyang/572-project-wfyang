__author__ = 'wayne'

import numpy as np
from numpy.linalg import eigh
from subproblem import solveSub

# Matrix Two Norm
def norm(X):
    return sum(sum(X * X))

# Likelihood Prox Step
def likelihoodStep(Y,U,C,ro):
    temp = ro * (Y - U) - C
    L, Q = eigh(temp)
    for i in range(len(L)):
        li = L[i]
        L[i] = (li + np.sqrt(li*li + 4.0*ro))/(2.0*ro)
    return Q.dot(np.diag(L).dot(Q.transpose()))

# Z is an n x n numpy array (row sliced)
# alpha, ro are positive constants
def dualDecomp(Z,alpha,ro,eta,h):
    X = Z
    n = len(Z)
    V = np.zeros((n,n))
    for i in range(n):
        X[i] = solveSub(Z[i],V[i],i,alpha,ro,h)
    V += eta * (X - X.transpose())
    error = norm(X-X.transpose())
    counter = 1.0
    while(error > 0.0001):

        print counter, error
        for i in range(n):
            X[i] = solveSub(Z[i],V[i],i,alpha,ro,h)
        V += eta * (X - X.transpose())
        error = norm(X - X.transpose())
        counter += 1.0
    X = 0.5 * (X + X.transpose())
    return X

# Main ADMM Loop
def admm(C,alpha,ro,eta,h,eps):
    n = len(C)
    Y = np.eye(n)
    U = np.zeros((n,n))
    print Y
    X = likelihoodStep(Y,U,C,ro)
    print X
    Y = dualDecomp(X-U, alpha,ro,eta,h)
    U = U + ro*(X - Y)

    Y_prev = np.eye(n)

    iter = 0
    while(norm(X-Y) > eps or norm(Y-Y_prev) > eps):
        Y_prev = Y
        iter += 1
        print iter
        print X-U
        print Y
        X = likelihoodStep(Y,U,C,ro)
        Y = dualDecomp(U-X, alpha,ro,eta,h)
        U = U + ro*(X - Y)

    return Y


# if __name__ == "__main__":

from numpy.random import multivariate_normal as mvn
from numpy.linalg import inv
Theta = np.diag([1.5]*5)
for i in range(1,5):
    Theta[0][i] = -0.2
    Theta[i][0] = -0.2
Sigma = inv(Theta)

sample = mvn(np.zeros((5)), Sigma, 1000)

C = np.cov(sample.transpose())

admm(C,0.1,0.5,0.9,lambda x: abs(x),0.00001)