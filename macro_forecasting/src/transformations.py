import numpy as np


def transform_series(series, log_transform=False):
    if log_transform:
        return np.log(series)
    return series