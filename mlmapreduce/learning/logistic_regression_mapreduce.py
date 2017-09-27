import time

import ipyparallel as ipp

from sklearn.model_selection import train_test_split

from mlmapreduce.kernel import mapreduce
from mlmapreduce.utilities import utils, hypothesis

client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    import numpy


"""
TODO: vectorize
def cost(theta, X, y):
    m = len(y)
    total_sum = 0

    for i, x in X.iterrows():
        x = x.as_matrix()
        estimate = h(theta, x)
        total_sum += (-1 * y[i]) * math.log(estimate) - (1.0 - y[i]) * math.log(1.0 - estimate)

    return (1.0/m) * total_sum
"""

RANDOM_SEED = 42
TEST_SET_SIZE = 0.2


def main():
    """
    Use mapReduce to train a logistic regression model.
    """
    data_frame = utils.mean_normalization(utils.load_data_frame('../../data/logistic-regression.txt'))
    # add intercept x=1
    data_frame.insert(0, 'x0', 1)

    y = utils.get_labels(data_frame)
    X = utils.get_features(data_frame)
    m, feature_vector_size = X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_SEED)

    print "Features: {}".format(feature_vector_size-1)
    print "X Size: {}".format(len(X))
    print "X train size: {}".format(len(X_train))
    print "y train size: {}".format(len(y_train))
    print "X test size: {}".format(len(X_test))
    print "y test size: {}".format(len(y_test))

    initial_theta = numpy.zeros(feature_vector_size)

    alpha = 0.1
    iterations = 500

    # distribute training data across the cluster
    async_result = dview.scatter('X', numpy.matrix(X_train.as_matrix()))
    dview.wait(async_result)
    async_result = dview.scatter('y', numpy.matrix(y_train).transpose())
    dview.wait(async_result)

    start = time.time()
    optimized_theta = mapreduce.gradient_descent2(
        dview, initial_theta, alpha, iterations, len(y_train), hypothesis.h_logistic_regression)

    end = time.time()
    print "Total: {}".format(end-start)
    # nb near perfect cost would be 0.203
    # for alpha = 0.1 and 500 iterations theta should be [-1.49128326  2.21833174  1.76958357]
    print "Optimized Theta: {}".format(optimized_theta)
    #print "Cost with optimized theta: {}".format(cost(optimized_theta, X, y))

if __name__ == "__main__":
    main()
