__author__ = 'wayne'

import numpy as np
from numpy.linalg import eigh
from numpy.linalg import inv
from subproblem import solveSub
from copy import copy
# Matrix Two Norm
from numpy.linalg import norm

# Likelihood Prox Step
def likelihoodStep(Y,U,C,ro):
    temp = ro * (Y - U) - C
    L, Q = eigh(temp)
    # print "Smallest Eigenvalue", np.min(L)
    for i in range(len(L)):
        li = L[i]
        L[i] = (li + np.sqrt(li*li + 4.0*ro))/(2.0*ro)
    # print "Smallest Eigenvalue", np.min(L)

    return Q.dot(np.diag(L).dot(Q.transpose()))

# Z is an n x n numpy array (row sliced)
# alpha, ro are positive constants
def dualDecomp(Z,V,alpha,ro,eta,h):
    X = Z
    n = len(Z)
    temp = V
    for i in range(n):
        X[i] = solveSub(Z[i], V[i], i, alpha, ro, h)
    temp += eta * (X - X.transpose())
    error = norm(X - X.transpose())
    counter = 0.0
    while(error > 0.00001):
        for i in range(n):
            X[i] = solveSub(Z[i], V[i], i, alpha, ro, h)
        temp += eta * (X - X.transpose())
        error = norm(X - X.transpose())
        counter += 1.0
        # print counter

    X = np.around(X,15)
    X = 0.5 * (X + X.transpose())

    return X, temp

# Main ADMM Loop
def admm(Cov, alpha, mu, eta, h, eps):

    covNormalizer = np.sqrt(np.diag(Cov))
    C = Cov / np.outer(covNormalizer, covNormalizer)
    # maxOffDiag = np.max(np.abs(np.tril(C, -1)))
    # C = np.array(C / maxOffDiag)
    ro = mu
    n = len(C)
    Y = inv(alpha*np.eye(n) + C)

    U = np.zeros((n,n))
    V = np.zeros((n,n))
    # Y, V = dualDecomp(U+X, np.zeros((n,n)), alpha, ro, eta, h)
    # U = U + X - Y

    iter = 0
    error1 = 1
    error2 = 1
    while(error1 > eps or error2 > eps):
        print iter, error1, error2
        Y_prev = copy(Y)
        iter += 1
        # X = likelihoodStep(Y,U,C,ro)
        L, Q = eigh(ro * (Y - U) - C)
        # while np.min(L) < 0:
        #     ro *= 0.5
        #     L, Q = eigh(ro * (Y - U) - C)
        #     print np.min(L), np.max(L), np.linalg.cond(ro*(Y-U) - C)
        L = (L + np.sqrt(L*L + 4.0*ro))/(2.0*ro)
        X = Q.dot(np.diag(L).dot(Q.transpose()))
        Y, V = dualDecomp(U+X, V, alpha, ro, eta, h)
        U = U + X - Y
        error1 = norm(X-Y)
        error2 = norm(Y-Y_prev)

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
        print admm(C,alpha,0.5,0.9,lambda x: abs(x),0.00001)
