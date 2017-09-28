import os
import time

import ipyparallel as ipp
from sklearn.model_selection import train_test_split
from nose.tools import eq_

from mlmapreduce.kernel import gradient_descent_mapreduce, gradient_descent_serial
from mlmapreduce.utilities import utils, hypothesis


client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    import numpy


RANDOM_SEED = 42
TEST_SET_SIZE = 0.2


def def_test_logistic_regression():
    datafile = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/logistic-regression.txt')

    data_frame = utils.mean_normalization(utils.load_data_frame(datafile))
    data_frame.insert(0, 'x0', 1)

    y = utils.get_labels(data_frame)
    X = utils.get_features(data_frame)
    m, feature_vector_size = X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_SEED)

    alpha = 0.1
    iterations = 25

    X_train_matrix = numpy.matrix(X_train.as_matrix())
    y_train_matrix = numpy.matrix(y_train).transpose()

    async_result = dview.scatter('X', X_train_matrix)
    dview.wait(async_result)
    async_result = dview.scatter('y', y_train_matrix)
    dview.wait(async_result)

    start = time.time()
    theta = gradient_descent_mapreduce.gradient_descent(
        dview, numpy.zeros(feature_vector_size), alpha, iterations, len(y_train), hypothesis.h_logistic_regression)
    end = time.time()
    print "time parallel: {}".format(end-start)
    time_parallel = end-start

    start = time.time()
    theta2 = gradient_descent_serial.gradient_descent(X_train_matrix, y_train_matrix, numpy.zeros(feature_vector_size), alpha, iterations, hypothesis.h_logistic_regression)
    end = time.time()
    time_serial = end-start
    print "time serial: {}".format(end-start)

    # parallel ~ 60 times slower
    print "parallel / serial factor = {}".format(time_parallel / time_serial)

    eq_(theta.tolist()[0][0], 0.11953961910411977)
    eq_(theta.tolist()[0][1], 0.25853238042750853)
    eq_(theta.tolist()[0][2], 0.22779121094153235)

    eq_(theta.all(), theta2.all())
