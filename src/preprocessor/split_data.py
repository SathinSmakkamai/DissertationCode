"""
File: split_data.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: data seperation, into traning set, validation set, and test set
"""

import numpy as np

def split_data_percentage(X_scaled, y_scaled, y_weights, train_ratio = 0.7, val_ratio = 0.1):

    num_samples = X_scaled.shape[0]
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size

    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]
    y_weights_train = y_weights[:train_size]

    X_val = X_scaled[train_size:train_size + val_size]
    y_val = y_scaled[train_size:train_size + val_size]
    y_weights_val = y_weights[train_size:train_size + val_size]

    X_test = X_scaled[train_size + val_size:]
    y_test = y_scaled[train_size + val_size:]
    y_weights_test = y_weights[train_size + val_size:]

    return (X_train, y_train, y_weights_train, X_val, y_val, y_weights_val,
            X_test, y_test, y_weights_test, train_size, val_size)


def split_data_by_date(X_scaled, y_scaled, y_weights, dates, test_start_date, val_ratio = 0.1):

    test_start_idx = np.where(dates >= np.datetime64(test_start_date))[0][0]

    num_samples = test_start_idx
    val_size = int(num_samples * val_ratio)
    train_size = num_samples - val_size

    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]
    y_weights_train = y_weights[:train_size]

    X_val = X_scaled[train_size:test_start_idx]
    y_val = y_scaled[train_size:test_start_idx]
    y_weights_val = y_weights[train_size:test_start_idx]

    X_test = X_scaled[test_start_idx:]
    y_test = y_scaled[test_start_idx:]
    y_weights_test = y_weights[test_start_idx:]

    return (X_train, y_train, y_weights_train, X_val, y_val, y_weights_val,
            X_test, y_test, y_weights_test, train_size, val_size)


def split_data_by_date_no_val(X_scaled, y_scaled, y_weights, dates, test_start_date):

    test_start_idx = np.where(dates >= np.datetime64(test_start_date))[0][0]

    num_samples = test_start_idx
    train_size = num_samples

    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]

    X_test = X_scaled[test_start_idx:]
    y_test = y_scaled[test_start_idx:]

    return X_train, X_test, y_train, y_test


