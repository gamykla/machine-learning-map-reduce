import ipyparallel as ipp

client = ipp.Client()
dview = client[:]

with dview.sync_imports():
    import pandas
    import numpy
    from scipy.stats import logistic
    import math
    from sklearn.model_selection import train_test_split


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


def distributed_sum():
    """
    one part of a sum that is distributred across the cluster.
    Compute: sum( (H(x_i) - y_i)*x_ij )
    the result will then be reduced and multiplied by 1/m to form a derivative term.
    must be pushed: j, theta
    scattered: X, y
    """
    i = 0
    sum_j = 0
    for _, x_i in X.iterrows():
        sum_j += (h(x_i) - y[i]) * x_i[j]
        i += 1

    return sum_j


def gradient_descent_mapreduce(theta, alpha, total_iterations, training_set_size):
    len_theta = len(theta)

    # repeat for total_iterations
    for _ in range(0, total_iterations):
        temp_theta = numpy.zeros(len_theta)

        for j in range(0, len_theta):
            dview.push({"theta": theta, "j": j})
            async_result = dview.apply_async(distributed_sum)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_SEED)

    initial_theta = numpy.array([0, 0, 0])

    alpha = 0.1
    iterations = 100

    # distribute training data across the cluster
    dview.scatter('X', X_train)
    dview.scatter('y', y_train)

    optimized_theta = gradient_descent_mapreduce(initial_theta, alpha, iterations, len(y_train))

    # nb near perfect cost would be 0.203
    print "Optimized Theta: {}".format(optimized_theta)
    print "Cost with optimized theta: {}".format(cost(optimized_theta, X, y))

if __name__ == "__main__":
    main()
