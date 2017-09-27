import ipyparallel as ipp
import os
from sklearn.model_selection import train_test_split

from nose.tools import eq_
from mlmapreduce.kernel import mapreduce
from mlmapreduce.utilities import utils, hypothesis, gradient_descent_serial

client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    import numpy


RANDOM_SEED = 42
TEST_SET_SIZE = 0.3


def test_linear_regression_results_match():
    datafile = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/linear-regression.txt')
    data_frame = utils.load_data_frame(datafile)
    data_frame.insert(0, 'x0', 1)

    y = utils.get_labels(data_frame)
    X = utils.get_features(data_frame)
    m, feature_vector_size = X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_SEED)

    X_train_matrix = numpy.matrix(X_train.as_matrix())
    y_train_matrix = numpy.matrix(y_train).transpose()

    # Distribute training set across the cluster
    async_result = dview.scatter('X', X_train_matrix)
    dview.wait(async_result)
    async_result = dview.scatter('y', y_train_matrix)
    dview.wait(async_result)

    initial_theta = numpy.zeros(feature_vector_size)

    alpha = 0.01
    iterations = 25

    # NB: for 500 iterastions, alpha=0.01 theta should be computed to be [-2.61862792  1.07368604]  with cost 4.62852531029
    theta = mapreduce.gradient_descent(dview, initial_theta, alpha, iterations, len(y_train), hypothesis.h_linear_regression)
    theta2 = gradient_descent_serial.gradient_descent(X_train_matrix, y_train_matrix, numpy.zeros(feature_vector_size), alpha, iterations, hypothesis.h_linear_regression)
    eq_(theta[0], -0.11521271754557487)
    eq_(theta[1], 0.82279250122643288)
    eq_(theta.all(), theta2.all())
