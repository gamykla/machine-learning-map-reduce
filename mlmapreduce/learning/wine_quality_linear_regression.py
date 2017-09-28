#
# Predict the quality of wine with a linear regression
#
import ipyparallel as ipp

from sklearn.model_selection import train_test_split
import time

from mlmapreduce.utilities import utils, hypothesis, cost
from mlmapreduce.kernel import gradient_descent_mapreduce, gradient_descent_serial

client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    import numpy

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
    iterations = 1000

    X_train_matrix = numpy.matrix(X_train.as_matrix())
    y_train_matrix = numpy.matrix(y_train).transpose()

    # Serial processing
    start = time.time()
    theta = gradient_descent_serial.gradient_descent(X_train_matrix, y_train_matrix, numpy.zeros(feature_vector_size), alpha, iterations, hypothesis.h_linear_regression)
    end = time.time()
    time_serial = end-start
    print "Theta: {}".format(theta)
    print "time serial: {}".format(time_serial)

    print "Cost: {}".format(cost.linear_regression_cost(theta, X_train_matrix, y_train_matrix))

    # map reduce
    async_result = dview.scatter('X', X_train_matrix)
    dview.wait(async_result)
    async_result = dview.scatter('y', y_train_matrix)
    dview.wait(async_result)

    start = time.time()
    theta = gradient_descent_mapreduce.gradient_descent(dview, numpy.zeros(feature_vector_size), alpha, iterations, len(y_train), hypothesis.h_linear_regression)
    print "Theta: {}".format(theta)
    end = time.time()
    time_parallel = end-start
    print "time parallel: {}".format(time_parallel)


    # Analyze results on test set
    X_test_matrix = numpy.matrix(X_test.as_matrix())
    y_test_matrix = numpy.matrix(y_test).transpose()
    print "Cost: {}".format(cost.linear_regression_cost(theta, X_test_matrix, y_test_matrix))

    if True:
        print "\n"
        print "predictions vs actuals:"
        predictions = hypothesis.h_linear_regression(theta, X_test_matrix)
        predictions_list = predictions.tolist()
        y_test_list = y_test_matrix.tolist()

        percent_diff_total = 0
        for i in range(0, len(predictions_list)):
            prediction_i = predictions_list[i][0]
            actual_i = y_test_list[i][0]

            if actual_i:
                percent_diff = abs(100.0 * ((prediction_i - actual_i) / actual_i))
            else:
                percent_diff = abs(100.0 * ((prediction_i - actual_i) / 1))

            percent_diff_total += percent_diff
            print "pred: {} actual: {}, difference: {}".format(prediction_i, actual_i, percent_diff)

        print "Average % diff: {}".format(percent_diff_total / len(y_test_list))


if __name__ == "__main__":
    main()