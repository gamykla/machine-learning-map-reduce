import numpy

def h(theta, x):
    """ linear regression hypothesis parameterizsed by model parameters theta """
    return numpy.asscalar(theta.dot(x))


def gradient_descent(X, y, theta, alpha, total_iterations):
    """
    perform a gradient descent serially
    """
    len_theta = len(theta)
    m = len(y)

    for _ in range(0, total_iterations):
        temp_theta = numpy.zeros(len_theta)

        # for each entry in theta, calculate new theta
        for j in range(0, len_theta):
            sum_j = 0
            # iterating over each example in the training data
            i = 0
            for _, x_i in X.iterrows():
                sum_j += (h(theta, x_i) - y[i]) * x_i[j]
                i += 1

            derivative_j = (1.0 / float(m)) * sum_j
            temp_theta[j] = theta[j] - alpha*derivative_j

        theta = temp_theta

    return theta