"""
File: preprocessor.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: preprocess step (stacking and scaling)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_residuals(residuals, sequence_length, stack):

    data = residuals.dropna().values
    X, y, y_weights = [], [], []

    for i in range(len(data) - sequence_length):

        if stack:
            X.append(np.cumsum(data[i:i + sequence_length], axis=0))
        else:
            X.append(data[i:i + sequence_length])

        y.append(data[i + sequence_length])
        y_weights.append(np.ones(data.shape[1]) / data.shape[1])

    X, y, y_weights = np.array(X), np.array(y), np.array(y_weights)

    return X, y, y_weights

def preprocess_return(daily_return, sequence_length):

    data = daily_return.dropna().values
    X, y, y_weights = [], [], []

    for i in range(len(data) - sequence_length):

        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
        y_weights.append(np.ones(data.shape[1]) / data.shape[1])

    X, y, y_weights = np.array(X), np.array(y), np.array(y_weights)

    return X, y, y_weights

def preprocess_scales_data(X, y, scale):

    if scale:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_processed = scaler.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

    else:
        X_processed = X.reshape(-1, X.shape[-1]).reshape(X.shape).astype(np.float32)
        y_processed = y.reshape(-1, y.shape[-1]).reshape(y.shape).astype(np.float32)

    return X_processed, y_processed