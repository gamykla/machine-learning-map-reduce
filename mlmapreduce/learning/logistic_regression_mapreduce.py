import ipyparallel as ipp

from sklearn.model_selection import train_test_split

from mlmapreduce.kernel import mapreduce

client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    import pandas
    import numpy
    from scipy.stats import logistic
    import math


def h(theta, x):
    """ logistic regression hypothesis function"""
    # sigmoid function (1 / (1 + e^-(theta' * x)))
    return numpy.asscalar(logistic.cdf(theta.dot(x)))


def load_data_frame():
    data_frame = pandas.read_csv('data/logistic-regression.txt', delimiter=",")
    return data_frame


def get_labels(data_frame):
    return numpy.asarray(data_frame['y'])


def get_features(data_frame):
    return data_frame.drop('y', axis=1)


def cost_i(theta, x_i, y_i):
    """ calculate the cost at x_i, y_i given theta """


def cost(theta, X, y):
    """ for given model parameters 'theta' compute the model cost for features 'X' and labels 'y' """
    m = len(y)
    total_sum = 0

    for i, x in X.iterrows():
        x = x.as_matrix()
        estimate = h(theta, x)
        total_sum += (-1 * y[i]) * math.log(estimate) - (1.0 - y[i]) * math.log(1.0 - estimate)

    return (1.0/m) * total_sum


def distributed_sum(theta, j, h):
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
            async_result = dview.apply_async(distributed_sum, theta, j, hypothesis_function)
            dview.wait(async_result)
            total_sum = reduce((lambda x, y: x + y), async_result.get())
            derivative_j = (1.0 / float(training_set_size)) * total_sum
            temp_theta[j] = theta[j] - alpha*derivative_j
        theta = temp_theta
    return theta



def mean_normalization(df):
    """ normalize dataset,
    see https://en.wikipedia.org/wiki/Normalization_(statistics) """
    from sklearn import preprocessing
    x = df.values #returns a numpy array
    column_names = df.columns.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pandas.DataFrame(x_scaled, columns=column_names)
    return df


RANDOM_SEED = 42
TEST_SET_SIZE = 0.2


def main():
    """
    Use mapReduce to train a logistic regression model.
    """
    data_frame = mean_normalization(load_data_frame())
    # add intercept x=1
    data_frame.insert(0, 'x0', 1)

    y = get_labels(data_frame)
    X = get_features(data_frame)
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
    async_result = dview.scatter('X', X_train)
    dview.wait(async_result)
    async_result = dview.scatter('y', y_train)
    dview.wait(async_result)

    optimized_theta = mapreduce.gradient_descent(dview, initial_theta, alpha, iterations, len(y_train), h)

    # nb near perfect cost would be 0.203
    # for alpha = 0.1 and 500 iterations theta should be [-1.49128326  2.21833174  1.76958357]
    print "Optimized Theta: {}".format(optimized_theta)
    print "Cost with optimized theta: {}".format(cost(optimized_theta, X, y))

if __name__ == "__main__":
    main()
