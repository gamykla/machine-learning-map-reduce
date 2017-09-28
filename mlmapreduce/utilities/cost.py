import numpy
from mlmapreduce.utilities import hypothesis


def _cost(theta, X, y, h):
    m = len(y)
    theta = numpy.matrix(theta)

    h_of_x = h(theta, X)
    h_of_x_minus_y = numpy.subtract(h_of_x, y)
    h_of_x_minus_y_squared = numpy.multiply(h_of_x_minus_y, h_of_x_minus_y)
    return (1.0/(2*m)) * h_of_x_minus_y_squared.sum()


def linear_regression_cost(theta, X, y):
    """ vectorized linear regression cost function"""
    return _cost(theta, X, y, hypothesis.h_linear_regression)

def logistic_regression_cost(theta, X, y):
    """ vectorized linear regression cost function"""
    return _cost(theta, X, y, hypothesis.h_logistic_regression)