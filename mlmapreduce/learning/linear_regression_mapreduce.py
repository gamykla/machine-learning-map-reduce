import ipyparallel as ipp
from sklearn.model_selection import train_test_split

from mlmapreduce.kernel import mapreduce

client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    # make sure workers have access to imports http://ipyparallel.readthedocs.io/en/5.0.0/multiengine.html#remote-imports
    import pandas
    import numpy
    import math


def h(theta, x):
    """ Linear regression hypothesis function"""
    return numpy.asscalar(theta.dot(x))


def get_labels(data_frame):
    return numpy.asarray(data_frame['y'])


def get_features(data_frame):
    return data_frame.drop('y', axis=1)


def load_data_frame():
    data_frame = pandas.read_csv('data/linear-regression.txt', delimiter=",")
    # add intercept x0=1
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
    async_result = dview.scatter('X', X_train)
    dview.wait(async_result)
    async_result = dview.scatter('y', y_train)
    dview.wait(async_result)

    initial_theta = numpy.zeros(feature_vector_size)

    # tunable gradient descent parameters
    alpha = 0.01
    iterations = 500

    # NB: for 500 iterastions, alpha=0.01 theta should be computed to be [-2.61862792  1.07368604]  with cost 4.62852531029
    theta = mapreduce.gradient_descent(dview, initial_theta, alpha, iterations, len(y_train), h)
    print "trained theta: {}".format(theta)


if __name__ == "__main__":
    main()

