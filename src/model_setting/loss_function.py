"""
File: loss_function.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: Loss function for model optimization.
             This file contains options for Sharpe ratio loss and return loss.
"""

import tensorflow as tf

# sharp ratio loss
def max_sharpe_ratio_loss(weights, returns):
    weights = tf.convert_to_tensor(weights, dtype = tf.float32)
    returns = tf.convert_to_tensor(returns, dtype = tf.float32)

    weights = tf.expand_dims(weights, axis = 0)
    returns = tf.expand_dims(returns, axis = 0)

    portfolio_returns = tf.reduce_sum(tf.multiply(returns, weights), axis = 2)

    mean_return = tf.reduce_mean(portfolio_returns, axis = 1)
    std_dev = tf.math.reduce_std(portfolio_returns, axis = 1)

    epsilon = 1e-6
    adjusted_std_dev = tf.where(std_dev > epsilon, std_dev, epsilon)

    sharpe_ratio = mean_return / adjusted_std_dev

    return -tf.reduce_mean(sharpe_ratio)

# return loss
def max_return_loss(weights, returns):
    weights = tf.convert_to_tensor(weights, dtype = tf.float32)
    returns = tf.convert_to_tensor(returns, dtype = tf.float32)

    weights = tf.expand_dims(weights, axis = 0)
    returns = tf.expand_dims(returns, axis = 0)

    portfolio_returns = tf.reduce_sum(tf.multiply(returns, weights), axis = 2)

    mean_return = tf.reduce_mean(portfolio_returns, axis = 1)
    mean_return = mean_return * 100

    return -tf.reduce_mean(mean_return)



