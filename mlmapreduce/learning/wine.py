from sklearn.model_selection import train_test_split
import time
import numpy

from mlmapreduce.utilities import utils, hypothesis, cost
from mlmapreduce.kernel import gradient_descent_mapreduce, gradient_descent_serial


RANDOM_SEED = 42
TEST_SET_SIZE = 0.2


def main():
    data_frame = utils.mean_normalization(utils.load_data_frame('../../data/wine/winequality-white.csv'))

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

    alpha = 0.01
    iterations = 2500

    X_train_matrix = numpy.matrix(X_train.as_matrix())
    y_train_matrix = numpy.matrix(y_train).transpose()

    start = time.time()
    theta = gradient_descent_serial.gradient_descent(X_train_matrix, y_train_matrix, numpy.zeros(feature_vector_size), alpha, iterations, hypothesis.h_linear_regression)
    end = time.time()
    time_serial = end-start
    print "Theta: {}".format(theta)
    print "Cost: {}".format(cost.linear_regression_cost(theta, X_train_matrix, y_train_matrix))
    print "time serial: {}".format(time_serial)

    X_test_matrix = numpy.matrix(X_test.as_matrix())
    y_test_matrix = numpy.matrix(y_test).transpose()

    #theta = numpy.matrix(theta)
    #predictions = hypothesis.h_linear_regression(theta, X_test_matrix)
    #print theta.shape
    #print X_test_matrix.shape
    #print predictions.shape

if __name__ == "__main__":
    main()