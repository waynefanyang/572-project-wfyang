__author__ = 'wayne'

from dualdecomp import *
from reweight import *
from datagen import *
from numpy.linalg import norm as no

def normalize(s):
    return np.apply_along_axis(lambda x: (x-np.mean(x))/no(x-np.mean(x)) ,0,s)

if __name__ == "__main__":
    p = 50
    Theta = genPrecisionPA(p)
    print Theta

    sample = sampleObservations(500,p,Theta)
    C = sampleCovariance(sample)
    print C
    print "running"
    # a = reweightedl1(np.matrix(C),0.5,0.5)
    a = admm(C,0.1,0.5,0.9,lambda x: np.abs(x),0.000001)
    b = admm(C,0.5,0.5,0.9,lambda x: np.abs(x),0.000001)
    c = admm(C,0.7,0.5,0.9,lambda x: np.abs(x),0.000001)
    print Theta
    print np.around(a,2)
    print np.around(b,2)
    print np.around(c,2)
    # #
    # data = np.loadtxt("glasso_data.csv",delimiter=",")
    # means = np.array([0.0]*5)
    # for i in range(1000):
    #     means += data[i]
    # means = means/1000.0
    # centerData = []
    # for i in range(1000):
    #     centerData.append(data[i] - means)
    #
    # centerData = np.matrix(centerData)
    # Cov = centerData.T * centerData /1000
    # Cov = np.array(Cov)
    #
    # a = admm(Cov,0.4,0.5,0.9,lambda x: np.log(x+1), 0.000001)
    # print np.around(a,3)
    #
    # lambduh = np.matrix(np.array([[0.1]*5]*5))
    # for i in range(5):
    #     lambduh[i,i] = 0.0
    #
    # print lambduh
    # b = glassoSolve(np.matrix(Cov),lambduh,0.5)
    # print np.around(b,3)
    #
    # print np.around(reweightedl1(np.matrix(Cov),0.1,0.1),3)