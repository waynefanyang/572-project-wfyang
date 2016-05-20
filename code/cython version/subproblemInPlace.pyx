import numpy as np
cimport numpy as np
cimport cython
from numpy.linalg import norm
from time import time
from copy import copy

ctypedef np.float32_t dtype_t



## Helper Function for solveSub
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[double, ndim=1] _solveSub(np.ndarray[double, ndim=1]  w,
              double alpha,
              double ro,
              np.ndarray[double, ndim=1] weights):
    cdef int n = np.shape(w)[0]

    cdef np.ndarray[double, ndim=1] values = np.abs(w) - (alpha/ro)*weights
    cdef np.ndarray[Py_ssize_t, ndim = 1] parents
    cdef np.ndarray[double, ndim = 1] counts, bounds
    cdef double ub = np.inf

    parents = np.arange(n)
    counts = np.ones((n))
    bounds = np.empty((n))

    cdef Py_ssize_t r,pos, k, cRoot, newRoot, j
    pos = 0

    while pos < n:
        r = pos
        bounds[pos] =  ub
        if values[pos] >= bounds[pos]:
            cRoot = pos

            while values[cRoot] >= bounds[cRoot]:
                ub_pos = cRoot - 1
                k = ub_pos
                while parents[k] != k:
                    k = parents[k]
                newRoot = k
                parents[cRoot] = newRoot

                values[newRoot] = (values[newRoot] * counts[newRoot] + values[cRoot]*counts[cRoot])/(counts[cRoot] + counts[newRoot])
                counts[newRoot] += counts[cRoot]

                cRoot = newRoot
                ub = values[newRoot]
        else:
            ub = values[pos]

        pos += 1

    # Assign values to output, denoted x
    cdef np.ndarray[double, ndim=1] x = np.zeros((n))
    k = 0

    while k < n:
        j = k
        while parents[j] != j:
            j = parents[j]
        x[k] = values[j]
        k += 1

    return x


## Main function for solving subproblem
cdef void solveSubInPlace(np.ndarray[double, ndim=2] X,
             np.ndarray[double, ndim=1] z, 
             np.ndarray[double, ndim=1] v, 
             Py_ssize_t l, 
             double alpha,
             double ro,
             np.ndarray[double, ndim=1] weights):
    # In Defazio and Caetano, the description of the algorithm maintains
    # a sorted to original function throughout the algorithm.
    # I found it is much easier to parse under the assumption that the subproblem
    # receives a presorted array and then revert the orderings after the problem is solved.

    # Variable Declarations
    cdef int n = z.shape[0] - 1
    cdef np.ndarray[double, ndim=1] w, x
    cdef np.ndarray[Py_ssize_t, ndim=1] indices, sortedToOriginal
    cdef Py_ssize_t k = 0


    # Computing target
    w = z - v
    w = np.delete(w,l)

    # Sort order and sorted to Original
    indices = np.argsort(np.abs(w))[::-1]
    sortedToOriginal = np.argsort(indices)

    x = _solveSub(np.abs(w)[indices], alpha, ro, weights)

    k = 0

    while k < n:
        if x[k] < 0:
            x[k] = 0.0
        if w[k] < 0:
            x[k] *= -1.0
        if k < l:
            X[l,k] = x[k]
        else:
            X[l,k+1] = x[k]
        k += 1

    X[l,l] = z[l]

    return

def dualDecomp(np.ndarray[double, ndim=2] Z,
               np.ndarray[double, ndim=2] V,
               double alpha,
               double ro,
               double eta,
               np.ndarray[double, ndim=1] weights):

    # variable declarations
    cdef int n
    cdef np.ndarray[double, ndim=2] X, temp, Xfin
    cdef double error
    cdef double startTime, averageIteration, iterStartTime, iterStopTime, stopTime

    n = len(Z)
    X = np.empty((n,n))
    temp = copy(V)

    startTime = time()
    averageIteration = 0.0
    for iteration in range(800):
        iterStartTime = time()
        for i in range(n):
            solveSubInPlace(X, Z[i], temp[i], i, alpha, ro, weights)
        diff = X - X.T
        temp += eta * (X - X.T)
        error = norm(X - X.T)
        iterStopTime = time()
        averageIteration += iterStopTime - iterStartTime
        if error < 1e-10:
            stopTime = time()
            print 'dual decomp finished after ' + str(iteration) + " iterations"
            print "Dual Decomposition Time: ", stopTime - startTime
            print "Average Iteration Time: ", averageIteration / (iteration + 1.0)
            break

    Xfin = 0.5 * (X + X.transpose())
    for i in range(n):
        for j in range(n):
            if abs(X[i,j]) < 1e-10 or abs(X[j,i]) < 1e-10:
                Xfin[i,j] = 0.0
    return Xfin, temp