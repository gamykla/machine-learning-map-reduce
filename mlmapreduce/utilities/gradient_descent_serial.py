import numpy


def gradient_descent(X, y, theta, alpha, total_iterations, hypothesis):
    """ Efficient vectorized gradient descent algorithm"""
    len_theta = len(theta)
    m = len(y)
    one_over_m = (1.0 / float(m))

    for _ in range(0, total_iterations):
        temp_theta = numpy.zeros(len_theta)

        X_by_theta_minus_y = (hypothesis(numpy.matrix(theta), X) - y)

        for j in range(0, len_theta):
            jth_column_of_X = X[:,j]
            derivative_j = one_over_m * numpy.multiply(X_by_theta_minus_y, jth_column_of_X).sum()
            temp_theta[j] = theta[j] - alpha*derivative_j

        theta = temp_theta

    return theta