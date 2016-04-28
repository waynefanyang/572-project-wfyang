__author__ = 'wayne'

import numpy as np
import networkx as nx
from numpy.linalg import inv
from numpy.random import multivariate_normal as mvn
from numpy.linalg import norm
from networkx_viewer import Viewer

def genPrecisionPA(p,m):

    M = np.zeros((p,p))

    ba=nx.barabasi_albert_graph(p,m)

    for e in ba.edges():


        M[e[0]][e[1]] = -0.2
        M[e[1]][e[0]] = -0.2

    for i in range(p):
        M[i][i] = 0.5 - np.sum(M[i])

    return M

def sampleObservations(n,p,Theta):

    Sigma = inv(Theta)
    sample = mvn(np.zeros((p)), Sigma, n)

    return sample

def sampleCovariance(sample):
    m = np.apply_along_axis(np.mean,0,sample)
    center = np.apply_along_axis(lambda x: x-m, 1, sample)
    v = np.apply_along_axis(norm,0,center)
    center = np.apply_along_axis(lambda x: x/v, 1, center)
    return center.transpose().dot(center)

def sampleCovariance1(sample):
    m = np.apply_along_axis(np.mean,0,sample)
    center = np.apply_along_axis(lambda x: x-m, 1, sample)
    C = center.transpose().dot(center)
    covNormalizer = np.sqrt(np.diag(C))
    C = C / np.outer(covNormalizer, covNormalizer)
    return C



