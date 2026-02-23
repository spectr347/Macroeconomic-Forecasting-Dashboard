import numpy as np


def train_test_split_ts(series, test_size):
    train = series[:-test_size]
    test = series[-test_size:]
    return train, test


def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))