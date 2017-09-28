import numpy
# see: # https://stackoverflow.com/questions/10857250/python-name-space-issues-with-ipython-parallel
from ipyparallel import interactive

@interactive
def distributed_sum(theta):
    """
    one part of a sum that is distributred across the cluster.
    The result will then be reduced and multiplied by 1/m to form a derivative term.

    Expects:
        - X, y: to be set through view.scatter()
        - h: hypothesis function is set
    """
    h_of_x_minus_y = numpy.subtract(h(numpy.matrix(theta), X) , y)

    result = []
    for j in range(0, len(theta)):
        result.append((numpy.multiply(h_of_x_minus_y, X[:,j])).sum())

    return result


def gradient_descent(dview, theta, alpha, total_iterations, training_set_size, hypothesis_function):
    """
    Gradient descent by mapreduce.
    dview - ipyparallel DirectView http://ipyparallel.readthedocs.io/en/5.0.0/multiengine.html#creating-a-directview-instance
    theta - initial settings for theta, a numpy array https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.array.html
    alpha - gradient descent parameter, how fast should we adjust thetas
    total_iterations - number of gradient descent iterations to perform
    training_set_size - total number of examples in the training set
    hypothesis_function - function from utilities.hypothesis.

    Note: before running this function, X and y must be set.
    X, y are numpy matrices see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matrix.html.
    They must be set with ipyparallel scatter (http://ipyparallel.readthedocs.io/en/latest/details.html#scatter-and-gather)
    """
    dview.push({"h": hypothesis_function})

    len_theta = len(theta)

    for _ in range(0, total_iterations):
        temp_theta = numpy.zeros(len_theta)

        async_result = dview.apply_async(distributed_sum, theta)
        dview.wait(async_result)

        for j in range(0, len_theta):
            results = async_result.get()
            total_j = 0
            for entry in results:
                total_j += entry[j]
            derivative_j = total_j / training_set_size
            temp_theta[j] = theta[j] - alpha*derivative_j

        theta = temp_theta
    return numpy.matrix(theta)
