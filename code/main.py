__author__ = 'wayne'

from dualdecomp import *
from reweight import *
from datagen import *
from roc import *
from testingUtils import *
# if __name__ == "__main__":

p = 20
m = 2
Theta = genPrecisionPA(p,m)
print Theta

sample = sampleObservations(200,p,Theta)
C = sampleCovariance(sample)
print C
print "running"
# a = reweightedl1(np.matrix(C),0.5,0.5)

alphas = [0.5,0.7,1.2,2.0,2.5]

a = admm(C,alpha,0.5,0.9,lambda x: np.log(x+1) + 0.1*x, 1e-6, True, 1000)
b = admm(C,alpha,0.5,0.9,lambda x: np.log(x+1) + 0.1*x, 1e-6, False, 1000)
#
# estimates = [admm(C,alpha,0.5,0.9,lambda x: np.log(x+1)+0.1*x, 0.000001, False, 1000) for alpha in alphas]
#
#
#
# for i in range(len(alphas)):
#     print alphas[i], edgeCount(estimates[i])
#
# rocs = [roc(est,Theta) for est in estimates]
# names = ["Alpha = " + str(alpha) for alpha in alphas]
# fname = "test.png"
#
# plot_roc(rocs,names,fname)
#








    # # #
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
