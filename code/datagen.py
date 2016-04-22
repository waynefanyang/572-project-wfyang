__author__ = 'wayne'

import numpy as np
import networkx as nx
from numpy.linalg import inv
from numpy.random import multivariate_normal as mvn
from numpy.linalg import norm
from networkx_viewer import Viewer

def genPrecisionPA(p):

    M = np.zeros((p,p))

    ba=nx.barabasi_albert_graph(p,2)

    for e in ba.edges():


        M[e[0]][e[1]] = -0.2
        M[e[1]][e[0]] = -0.2

    for i in range(p):
        M[i][i] = 1.5 - np.sum(M[i])

    return M

def sampleObservations(n,p,Theta):

    Sigma = inv(Theta)
    sample = mvn(np.zeros((p)), Sigma, n)

    return sample

def sampleCovariance(sample):
    center = sample - np.mean(sample)
    return center.transpose().dot(center)



