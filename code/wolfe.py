__author__ = 'wayne'

import numpy as np
from copy import copy

def AffineMinimizer():
    # TO BE IMPLEMENTED
    return "IMPLEMENT ME"

def innerprod(x,y):
    output = 0.0
    for key in x.keys():
        if key in y:
            output += x[key]*y[key]
    return output

def f(eSet,p):
    deg = np.zeros((p))
    for edge in eSet:
        deg[edge[0]] += 1
        deg[edge[1]] += 1
    return np.sum(np.log(deg+1))



def LO(w,p):

    ordering = sorted(w, key=w.get)
    print ordering

    X = {}
    eSet = []
    eSetOld = []
    for edge in ordering:
        print eSet
        eSet.append(edge)
        print eSet
        X[edge] = f(eSet,p) - f(eSetOld,p)
        eSetOld = copy(eSet)

    return X

def wolfe(initq,p,eps):
    x = initq
    y = None
    lambduhs = [1.0]
    S = [initq]
    while True:
        q = LO(x,p)
        if innerprod(x,x) <= innerprod(x,q) + eps:
            break
        S.append(q)
        while True:
            y, alpha = AffineMinimizer(S)
            if all(alpha >= 0):
                break
            else:
                temp = np.zeros((len(S)))
                for i in range(len(S)):
                    if alpha[i] < 0:
                        temp[i] = lambduhs[i]/(lambduhs[i] - alpha[i])
                    else:
                        temp[i] = np.inf
                i = np.argmin(temp)
                theta = temp[i]
                x = plus(mult(theta,y) , mult(1-theta,x))
                lambduhs = theta * alpha + (1- theta)*lambduhs
                S = S[lambduhs > 0]
                lambduhs = lambduhs[lambduhs > 0]
        x = y
    return x










if __name__ == "__main__":

    ## Testing Linear Oracle

    w = {(0,1):3, (0,2):2, (1,2):1}
    p = 3
    print LO(w,p)

    w = {(0,1):1.0, (0,2):2.0, (1,2):3.0}
    p = 3
    print LO(w,p)