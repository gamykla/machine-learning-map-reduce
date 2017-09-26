import numpy

def gradient_descent(X, y, theta, alpha, total_iterations, hypothesis_function):
    """
    perform a gradient descent serially in vectorized form
    """
    len_theta = len(theta)
    m = len(y)
    y = numpy.matrix(y).transpose()
    X = numpy.matrix(X.as_matrix())
    one_over_m = (1.0 / float(m))

    for _ in range(0, total_iterations):
        temp_theta = numpy.zeros(len_theta)

        for j in range(0, len_theta):
            derivative_j = one_over_m * numpy.multiply((hypothesis_function(theta, X) - y), X[:,j]).sum()
            temp_theta[j] = theta[j] - alpha*derivative_j

        theta = temp_theta

    return theta