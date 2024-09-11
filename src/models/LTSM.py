"""
File: LSTM.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: predictive model: long short-term memory model
"""

import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, Flatten, GlobalAveragePooling1D
from keras.models import Model

# Layer to normalize weights
class NormalizeWeightsLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(NormalizeWeightsLayer, self).__init__(**kwargs)

    def call(self, inputs):
        sum_abs_weights = tf.reduce_sum(tf.abs(inputs), axis=-1, keepdims=True)
        normalized_weights = inputs / (sum_abs_weights + 1e-6)
        return normalized_weights

class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class LSTM_Model(Model):

    def __init__(self, input_shape, num_assets, num_lstm_units=64, **kwargs):
        super(LSTM_Model, self).__init__(**kwargs)
        self.lstm1 = LSTM(num_lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        self.lstm2 = LSTM(num_lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        self.global_avg_pool = GlobalAveragePooling1D()
        self.expand_dims = ExpandDimsLayer(axis=1)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(64, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(num_assets)
        self.normalize_weights = NormalizeWeightsLayer()

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.global_avg_pool(x)
        x = self.expand_dims(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        outputs = self.normalize_weights(x)
        return outputs

def LSTM_model(input_shape, num_assets, num_lstm_units=64):
    model = LSTM_Model(input_shape=input_shape, num_assets=num_assets, num_lstm_units=num_lstm_units)
    return model
