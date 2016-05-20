__author__ = 'wayne'

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float32_t dtype_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t find(Py_ssize_t v, 
         np.ndarray[double, ndim=1] values,
         np.ndarray[Py_ssize_t, ndim=1] parents,
         np.ndarray[double, ndim=1] counts):
        # Based on union strategy, this is equivalent to finding the minimum value
        if not v == parents[v]:
            parents[v] = find(parents[v], values, parents, counts)
        return parents[v]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void union(Py_ssize_t x, 
           Py_ssize_t y, 
           np.ndarray[double, ndim=1] values,
           np.ndarray[Py_ssize_t, ndim=1] parents,
           np.ndarray[double, ndim=1] counts):

    cdef Py_ssize_t xRoot = find(x,values,parents,counts)
    cdef Py_ssize_t yRoot = find(y,values,parents,counts)

    if xRoot == yRoot:
        return


    cdef double xVal = values[xRoot]
    cdef double yVal = values[yRoot]
    cdef double xCount = counts[xRoot]
    cdef double yCount = counts[yRoot]
    cdef double newVal = (xVal * xCount + yVal * yCount) / (xCount + yCount)
    values[xRoot] = newVal
    values[yRoot] = newVal

    counts[xRoot] += yCount
    counts[yRoot] = counts[xRoot]

    if xRoot <= yRoot:
        parents[yRoot] = xRoot

    else:
        parents[xRoot] = yRoot

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double findValue(Py_ssize_t v,
                      np.ndarray[double, ndim=1] values,
                      np.ndarray[Py_ssize_t, ndim=1] parents,
                      np.ndarray[double, ndim=1] counts):
    return values[find(v,values,parents,counts)]


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


# z,v are 1d numpy arrays
# l is the row number
@cython.boundscheck(False)
@cython.wraparound(False)
def solveSub(np.ndarray[double, ndim=1] z, 
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


    # Computing target
    w = z - v
    w = np.delete(w,l)

    # Sort order and sorted to Original
    indices = np.argsort(np.abs(w))[::-1]
    sortedToOriginal = np.argsort(indices)

    x = _solveSub(np.abs(w)[indices], alpha, ro, weights)
    x = x[sortedToOriginal]

    for k in range(n):
        if x[k] < 0:
            x[k] = 0.0
        if w[k] < 0:
            x[k] *= -1.0

    x = np.insert(x,l,z[l])
    return x




