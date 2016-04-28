__author__ = 'wayne'

import numpy as np
from numpy.linalg import eigh
from numpy.linalg import inv
from subproblem import solveSub
from copy import copy
# Matrix Two Norm
from numpy.linalg import norm
from scipy.linalg import blas

# Likelihood Prox Step
def likelihoodStep(Y,U,C,ro):
    temp = ro * (Y - U) - C
    L, Q = eigh(temp)
    # print "Smallest Eigenvalue", np.min(L)
    newEigs = (L + np.sqrt(L*L + 4*ro)) / (2*ro)
    # print "Smallest Eigenvalue", np.min(L)

    return blas.dgemm(alpha=1.0, a=(Q*newEigs), b=Q, trans_b=True),L,Q,newEigs

# Z is an n x n numpy array (row sliced)
# alpha, ro are positive constants
def dualDecomp(Z,V,alpha,ro,eta,h):
    n = len(Z)
    X = np.empty((n,n))
    temp = copy(V)
    for iteration in range(800):
        for i in range(n):
            X[i] = solveSub(Z[i], temp[i], i, alpha, ro, h)
        diff = X - X.T
        temp += eta * (X - X.T)
        error = norm(X - X.T)
        if error < 1e-10:
            print 'dual decomp finished after ' + str(iteration) + " iterations"
            break

    Xfin = 0.5 * (X + X.transpose())
    for i in range(n):
        for j in range(n):
            if abs(X[i,j]) < 1e-10 or abs(X[j,i]) < 1e-10:
                Xfin[i,j] = 0.0
    return Xfin, temp



# Main ADMM Loop
def admm(Cov, alpha, mu, eta, h, eps, normal, maxiter):

    # Normalizing The Covariance Matrix

    C = None
    if normal:
        covNormalizer = np.sqrt(np.diag(Cov))
        C = Cov / np.outer(covNormalizer, covNormalizer)
        maxOffDiag = np.max(np.abs(np.tril(C, -1)))
        C = np.array(C / maxOffDiag)
    else:
        C = Cov

    # Setting up initialization

    ro = mu
    n = len(C)
    Y = np.eye(n)
    U = np.eye(n)
    V = np.zeros((n,n))

    iter = 0
    error1 = 1
    error2 = 1
    while(iter < maxiter and (error1 > eps or error2 > eps)):
        print iter, error1, error2, ro

        Y_prev = copy(Y)
        iter += 1

        X,L,Q,newEigs = likelihoodStep(Y,U,C,ro)
        print X[1,1]
        Y, V = dualDecomp(U+X, V, alpha, ro, eta, h)
        U += X - Y
        error1 = norm(X-Y)
        error2 = norm(Y-Y_prev)

        differenceMargin = 10
        if error1 > error2*differenceMargin:
            ro *= 1.1
            U /= 1.1
        elif error2 > error1*differenceMargin:
            ro *= 0.9
            U /= 0.9
        print "***"
        print "nonzeros", sum(Y != 0.0)
        print "***"

    return Y


if __name__ == "__main__":

    from numpy.random import multivariate_normal as mvn
    from numpy.linalg import inv
    Theta = np.diag([1.5]*5)
    for i in range(1,5):
        Theta[0][i] = -0.2
        Theta[i][0] = -0.2
    Sigma = inv(Theta)

    sample = mvn(np.zeros((5)), Sigma, 1000)
    center = sample - np.mean(sample)
    for i in range(5):
        center[:,i] /= norm(center[:,i])
    # print Sigma
    C = center.transpose().dot(center)
    for alpha in [0.01,0.05,0.1,0.5]:
        print admm(C,alpha,0.5,0.9,lambda x: abs(x),0.00001,True,1000)
