import ipyparallel as ipp

client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    # this way, all of these imports are available to workers
    import pandas
    import numpy
    import math
    from sklearn.model_selection import train_test_split


def h(theta, x):
    """ Linear regression hypothesis function"""
    return numpy.asscalar(theta.dot(x))


def get_labels(data_frame):
    return numpy.asarray(data_frame['y'])


def get_features(data_frame):
    return data_frame.drop('y', axis=1)


def load_data_frame():
    data_frame = pandas.read_csv('data/linear-regression.txt', delimiter=",")
    # add intercept x=1
    data_frame.insert(0, 'i', 1)
    return data_frame


def compute_cost_function(theta, X, y):
    """ for given model parameters 'theta' compute the model
    cost for features 'X' and labels 'y' """
    m = len(y)
    total_sum = 0

    for i, x in X.iterrows():
        x = x.as_matrix()
        total_sum += math.pow((h(theta, x) - y[i]), 2)

    return (1.0/(2*m)) * total_sum


def distributed_sum():
    """
    one part of a sum that is distributred across the cluster.
    The result will then be reduced and multiplied by 1/m to form a derivative term.
    must be pushed: j, theta
    scattered: X, y
    """
    i = 0
    sum_j = 0
    for _, x_i in X.iterrows():
        sum_j += (h(theta, x_i) - y[i]) * x_i[j]
        i += 1

    return sum_j


def gradient_descent(theta, alpha, total_iterations, training_set_size, hypothesis_function):
    """
    Gradient descent - this is the algorithm that finds optimal parameters. Compute with map-reduce pattern.
    """
    len_theta = len(theta)

    # repeat for total_iterations
    for _ in range(0, total_iterations):
        temp_theta = numpy.zeros(len_theta)

        for j in range(0, len_theta):
            dview.push({"theta": theta, "j": j, "h": hypothesis_function})
            async_result = dview.apply_async(distributed_sum)
            dview.wait(async_result)
            total_sum = reduce((lambda x, y: x + y), async_result.get())
            derivative_j = (1.0 / float(training_set_size)) * total_sum
            temp_theta[j] = theta[j] - alpha*derivative_j
        theta = temp_theta
    return theta


RANDOM_SEED = 42
TEST_SET_SIZE = 0.3


def main():
    data_frame = load_data_frame()
    y = get_labels(data_frame)
    X = get_features(data_frame)
    m, feature_vector_size = X.shape # feature vector size includes intercept term (1)

    # partition data set into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_SEED)

    print "Features: {}".format(feature_vector_size-1)
    print "X Size: {}".format(len(X))
    print "X train size: {}".format(len(X_train))
    print "y train size: {}".format(len(y_train))
    print "X test size: {}".format(len(X_test))
    print "y test size: {}".format(len(y_test))

    # Distribute training set across the cluster
    dview.scatter('X', X_train)
    dview.scatter('y', y_train)

    theta = numpy.zeros(feature_vector_size)

    # tunable gradient descent parameters
    alpha = 0.01
    iterations = 500

    # NB: for 500 iterastions, alpha=0.01 theta should be computed to be [-2.61862792  1.07368604]  with cost 4.62852531029
    theta = gradient_descent(theta, alpha, iterations, len(y_train), h)
    print "trained theta: {}".format(theta)


if __name__ == "__main__":
    main()

