__author__ = 'wayne'
import numpy as np

def falsepositives(est,truth):
    n = np.shape(truth)[0]
    total = 0.0
    incorrect = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                if est[i,j] != 0:
                    total += 1.0
                    if truth[i,j] == 0.0:
                        incorrect += 1.0
    return incorrect/total

def truenegatives(est,truth):
    n = np.shape(truth)[0]
    total = 0.0
    correct = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                if truth[i,j] == 0.0:
                    total += 1.0
                    if est[i,j] == 0:
                        correct += 1.0
    return correct/total

def truepositives(est,truth):
    n = np.shape(truth)[0]
    total = 0.0
    correct = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                if truth[i,j] != 0.0:
                    total += 1.0
                    if est[i,j] != 0:
                        correct += 1.0
    return correct/total

def edgeCount(mat):
    n = np.shape(mat)[0]
    count = 0
    for i in range(n-1):
        for j in range(i+1,n):
            if mat[i,j] != 0:
                count += 1
    return count

