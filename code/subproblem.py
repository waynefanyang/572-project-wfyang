__author__ = 'wayne'

import numpy as np
from disjointset import disjointSet

# z,v are 1d numpy arrays
def solveSub(z,v,l,alpha,ro,h):

    n = len(z) - 1
    w = z - v
    w = np.delete(w,l)
    mu = np.argsort(abs(w))
    values = [0.0] * n

    for k in range(len(values)):
        j = mu[k]
        values[j] = abs(w[j]) - alpha/ro*(h(j+1) - h(j))

    set = disjointSet(n,values)

    for k in range(len(values)):
        j = mu[k]
        r = k
        while r > 1 and set.findValue(r) >= set.findValue(r-1):
            set.union(r,r-1)
            # Note that the averaging of the values is built into the union method
            r = set.find(r)

    x = np.array([0.0] * n)
    for k in range(len(values)):
        x[k] = set.findValue(k)
        if x[k] < 0:
            x[k] = 0.0
        if w[k] < 0:
            x[k] *= -1.0
    x = np.insert(x,l,z[l])
    return x

if __name__ == "__main__":
    z = np.array([1,1,1])
    v = np.array([0,0,0])
    l = 0
    alpha = 0.5
    ro = 0.9
    def h(x):
        return abs(x)
    import cProfile
    cProfile.run("""for i in range(10000):
        solveSub(z,v,l,alpha,ro,lambda x: np.log(x+1))
    """)
    print solveSub(z,v,l,alpha,ro,lambda x: np.log(x+1))
