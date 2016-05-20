__author__ = 'wayne'

cdef class disjointSet:
    

    
    def __init__(self,n,values):
        self.parent = list(range(n))
        self.values = values
        self.counts = [1.0]*n

    def find(self, v):
        # Based on union strategy, this is equivalent to finding the minimum value
        if not v == self.parent[v]:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, x, y):
        xRoot = self.find(x)
        yRoot = self.find(y)

        if xRoot == yRoot:
            return

        xVal = self.values[xRoot]
        yVal = self.values[yRoot]
        xCount = self.counts[xRoot]
        yCount = self.counts[yRoot]
        newVal = (xVal * xCount + yVal * yCount) / (xCount + yCount)
        self.values[xRoot] = newVal
        self.values[yRoot] = newVal

        self.counts[xRoot] += yCount
        self.counts[yRoot] = self.counts[xRoot]

        if xRoot <= yRoot:
            self.parent[yRoot] = xRoot

        else:
            self.parent[xRoot] = yRoot

    def findValue(self,v):
        return self.values[self.find(v)]

if __name__ == "__main__":
    test = disjointSet(10,range(10))
    for i in range(10):
        print "Element", i, "Parent", test.find(i), "value", test.findValue(i)

    test.union(0,1)
    print test.findValue(0) == 0.5
    print test.findValue(1) == 0.5

    print test.find(1) == 0

    test.union(1,9)
    test.union(9,3)

    print test.findValue(1) == 3.25
    print test.findValue(9) == 3.25

    print test.find(9) == 0
    print test.find(3) == 0

    test.union(8,9)
    print test.find(8) == 0
    test.union(4,5)
    test.union(6,5)
    print test.find(6) == 4
    print test.find(5) == 4