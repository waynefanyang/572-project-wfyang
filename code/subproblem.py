__author__ = 'wayne'

import numpy as np
from disjointset import disjointSet

# z,v are 1d numpy arrays
# l is
def solveSub(z,v,l,alpha,ro,h):
    # In Defazio and Caetano, the description of the algorithm maintains
    # a sorted to original function throughout the algorithm.
    # I found it is much easier to parse under the assumption that the subproblem
    # receives a presorted array and then revert the orderings after the problem is solved.

    n = len(z) - 1
    w = z - v
    w = np.delete(w,l)
    indices = np.argsort(np.abs(w))
    sortedToOriginal = {indices[::-1][i]:i for i in range(len(indices))}

    x = _solveSub(np.abs(w)[indices[::-1]], alpha, ro, h)
    x = np.array([x[sortedToOriginal[k]] for k in range(n)])

    for k in range(len(x)):
        if x[k] < 0:
            x[k] = 0.0
        if w[k] < 0:
            x[k] *= -1.0

    x = np.insert(x,l,z[l])
    return x



def _solveSub(w,alpha,ro,h):
    n = len(w)
    values = [0.0] * n

    for k in range(len(values)):
        values[k] = np.abs(w[k]) - alpha/ro*(h(k+1) - h(k))

    set = disjointSet(n,values)

    for k in range(len(values)):
        r = k
        while r > 1 and set.findValue(r) >= set.findValue(r-1):
            set.union(r,r-1)
            # Note that the averaging of the values is built into the union method
            r = set.find(r)

    x = np.array([0.0] * n)
    for k in range(len(values)):
        x[k] = set.findValue(k)

    return x


if __name__ == "__main__":
    z = np.array([0,-1,2,-3,4,-5,6])
    v = np.array([0]*7)
    l = 0
    alpha = 1.0
    ro = 1.0
    def h(x):
        return abs(x)
    import cProfile
    # cProfile.run("""for i in range(100):
    #     solveSub(z,v,l,alpha,ro,lambda x: np.log(x+1))
    # """)

    import numpy.random as rnd
    w = rnd.random_integers(0,100,size=(60))

    indices = np.argsort(w)

    def h(x):
        return np.abs(x+1)
    #
    # cProfile.run("""for i in range(6000):
    #     _solveSub(w[indices[::-1]],1.0,1.0,h)
    #     """)

    print np.around(solveSub(z,v,l,alpha,ro,lambda x: np.log(x+1)),2)
    print z
