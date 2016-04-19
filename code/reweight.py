__author__ = 'wayne'

import numpy as np
from glasso import glassoSolve
from glasso import twonorm
from copy import copy

vabs = np.vectorize(np.abs)

## computeLambduh
# Compute the lambdas in the "Expectation Step" of the Liu and Ihler method.
# Theta is a p x p matrix (estimated precision matrix from previous round)
# Epsilon is a list of length p
# alpha is a scalar tuning parameter

def computeLambduh(Theta,Epsilon,alpha,beta):
    p = np.shape(Theta)[0]
    Lambduh = np.matrix(np.zeros((p,p)))
    norms = []
    for i in range(p):
        norms.append(np.sum(np.abs(np.delete(Theta[i,:],i,1))))
        Lambduh[i,i] = beta
    for i in range(p):
        for j in range(p):
            if i != j:
                Lambduh[i,j] = 1.0/(norms[i] + Epsilon[i]) + 1.0/(norms[j] + Epsilon[j])
                Lambduh[i,j] *= alpha
    return Lambduh

def computeEpsilon(Theta):
    E = []
    for i in range(np.shape(Theta)[0]):
        E.append(Theta[i,i])
    return E

## Estimate Graph
def reweightedl1(C,alpha,beta):
    p = np.shape(C)[0]
    tol = 1e-10

    Theta = np.matrix(np.eye(p))
    Epsilon = computeEpsilon(Theta)
    Lambduh = computeLambduh(Theta,Epsilon,alpha,beta)
    oldTheta = np.matrix(np.zeros((p,p)))

    iteration = 0

    while iteration < 100 and twonorm(Theta - oldTheta) > tol:
        print iteration
        oldTheta = copy(Theta)
        Theta = glassoSolve(C,Lambduh,0.5)
        iteration += 1

    return Theta


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

    out = np.around(reweightedl1(Cov,0.1,0.1),3)
    print out