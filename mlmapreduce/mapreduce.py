import numpy
# see: # https://stackoverflow.com/questions/10857250/python-name-space-issues-with-ipython-parallel
from ipyparallel import interactive

@interactive
def distributed_sum(theta, j, h):
    """
    one part of a sum that is distributred across the cluster.
    The result will then be reduced and multiplied by 1/m to form a derivative term.

    Expects X, y, to be set through view.scatter()
    """
    i = 0
    sum_j = 0
    for _, x_i in X.iterrows():
        sum_j += (h(theta, x_i) - y[i]) * x_i[j]
        i += 1
    return sum_j


def gradient_descent(dview, theta, alpha, total_iterations, training_set_size, hypothesis_function):
    """
    Gradient descent - this is the algorithm that finds optimal parameters. Compute with map-reduce pattern.
    Can be used for logistic regression and linear regression.
    """

    len_theta = len(theta)
    # repeat for total_iterations
    for _ in range(0, total_iterations):
        temp_theta = numpy.zeros(len_theta)

        for j in range(0, len_theta):
            async_result = dview.apply_async(distributed_sum, theta, j, hypothesis_function)
            dview.wait(async_result)
            total_sum = reduce((lambda x, y: x + y), async_result.get())
            derivative_j = (1.0 / float(training_set_size)) * total_sum
            temp_theta[j] = theta[j] - alpha*derivative_j
        theta = temp_theta
    return theta
