__author__ = 'wayne'

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

def roc(est,true):
    n = np.shape(est)[0]
    trueLabels = []
    estScores = []
    for i in range(n-1):
        for j in range(i+1,n):
            if true[i,j] != 0:
                trueLabels.append(1)
            else:
                trueLabels.append(0)
            estScores.append(np.abs(est[i,j]))

    return roc_curve(np.array(trueLabels), np.array(estScores))



def plot_roc(rocs,names,filename):
    plt.ioff()
    lines = []
    for i in range(len(rocs)):
        r = rocs[i]
        temp, = plt.plot(r[0],r[1], label=names[i])
        lines.append(temp)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(lines, names)
    plt.savefig(filename)

