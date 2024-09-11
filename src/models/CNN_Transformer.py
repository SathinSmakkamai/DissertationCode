"""
File: CNN_Transformer.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: predictive model: Transformer with convolution layers
"""

import tensorflow as tf
from keras.layers import (Input, Conv1D, Dense, Dropout, Flatten,
                          GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention)
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


class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1):

        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation='relu'),Dense(key_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):

        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)

class CNN_Transformer_Model(Model):

    def __init__(self, input_shape, num_assets, num_heads=10, key_dim=64, ff_dim=128, **kwargs):

        super(CNN_Transformer_Model, self).__init__(**kwargs)
        self.conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
        self.global_avg_pool = GlobalAveragePooling1D()
        self.expand_dims = ExpandDimsLayer(axis=1)
        self.transformer_block = TransformerBlock(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(64, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(num_assets)
        self.normalize_weights = NormalizeWeightsLayer()

    def call(self, inputs, training=False):

        x = self.conv1(inputs)
        x = self.global_avg_pool(x)
        x = self.expand_dims(x)
        x = self.transformer_block(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        outputs = self.normalize_weights(x)
        return outputs

def CNNTFM_model(input_shape, num_assets, num_heads = 10, key_dim=64, ff_dim=128):
    model = CNN_Transformer_Model(input_shape=input_shape, num_assets=num_assets, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim)
    return model
