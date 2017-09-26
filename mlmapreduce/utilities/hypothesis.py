from scipy.stats import logistic


def h_linear_regression(theta, X):
    """ Hypothesis function for linear regression"""
    return X.dot(theta.transpose())


def h_logistic_regression(theta, X):
    """Hypothesis function for logistic regresssion"""
    return logistic.cdf(X.dot(theta.transpose()))