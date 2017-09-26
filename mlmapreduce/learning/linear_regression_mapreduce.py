import ipyparallel as ipp
from sklearn.model_selection import train_test_split

from mlmapreduce.kernel import mapreduce
from mlmapreduce.utilities import utils, hypothesis

client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    # make sure workers have access to imports
    # http://ipyparallel.readthedocs.io/en/5.0.0/multiengine.html#remote-imports
    import numpy

"""
TODO: vectorize
def compute_cost_function(theta, X, y):
    m = len(y)
    total_sum = 0

    for i, x in X.iterrows():
        x = x.as_matrix()
        total_sum += math.pow((h(theta, x) - y[i]), 2)

    return (1.0/(2*m)) * total_sum
"""

RANDOM_SEED = 42
TEST_SET_SIZE = 0.3


def main():
    data_frame = utils.load_data_frame('data/linear-regression.txt')
    # add intercept x0=1
    data_frame.insert(0, 'x0', 1)

    y = utils.get_labels(data_frame)
    X = utils.get_features(data_frame)
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
    async_result = dview.scatter('X', numpy.matrix(X_train.as_matrix()))
    dview.wait(async_result)
    async_result = dview.scatter('y', numpy.matrix(y_train).transpose())
    dview.wait(async_result)

    initial_theta = numpy.zeros(feature_vector_size)

    # tunable gradient descent parameters
    alpha = 0.01
    iterations = 500

    import time
    start = time.time()
    # NB: for 500 iterastions, alpha=0.01 theta should be computed to be [-2.61862792  1.07368604]  with cost 4.62852531029
    theta = mapreduce.gradient_descent(dview, initial_theta, alpha, iterations, len(y_train), hypothesis.h_linear_regression)
    print "trained theta: {}".format(theta)
    end = time.time()

    print "Time: {}".format(end-start)


if __name__ == "__main__":
    main()
