import numpy
from scipy.stats import logistic


def h(theta, x):
    """ logistic regression hypothesis function"""
    # sigmoid function (1 / (1 + e^-(theta' * x)))
    return numpy.asscalar(logistic.cdf(theta.dot(x)))


def gradient_descent(X, y, initial_theta, alpha, total_iterations):
    """ Serial gradient descent"""
    len_theta = len(initial_theta)

    m = len(y)
    one_over_m = (1.0 / float(m))

    theta = initial_theta.copy()

    # repeat for total_iterations
    for it in range(0, total_iterations):
        temp_theta = numpy.zeros(len_theta)

        for j in range(0, len_theta):
            sum_j = 0
            for i, x_i in X.iterrows():
                sum_j += (h(theta, x_i) - y[i]) * x_i[j]
            derivative_j =  one_over_m*sum_j
            temp_theta[j] = theta[j] - alpha*derivative_j

        theta = temp_theta

    return theta