import numpy


x = numpy.array([1,2,3])
y = numpy.array([1,2,3])

print x.dot(y)


y = numpy.matrix( ((1,2), (5, -1)) )
x = numpy.array( ((2,3), (3, 5)) )


# 2 x 2  . 2 x 2

print x.dot(y)

print y[:,0]


t = numpy.matrix([1, 1, 1]).transpose()
x = numpy.matrix([2, 2, 2]).transpose()
print numpy.multiply(t, x)


from scipy.stats import logistic

print logistic.cdf(numpy.multiply(t, x))
