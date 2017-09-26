import numpy
import pandas
from sklearn import preprocessing


def get_labels(data_frame):
    return numpy.asarray(data_frame['y'])


def get_features(data_frame):
    return data_frame.drop('y', axis=1)


def load_data_frame(data_file):
    data_frame = pandas.read_csv(data_file, delimiter=",")
    return data_frame


def mean_normalization(df):
    """ normalize dataset,
    see https://en.wikipedia.org/wiki/Normalization_(statistics) """
    x = df.values #returns a numpy array
    column_names = df.columns.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pandas.DataFrame(x_scaled, columns=column_names)
    return df
