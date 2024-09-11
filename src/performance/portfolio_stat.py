"""
File: portfolio_stat.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: calcuate portfolio return
"""

import numpy as np
import tensorflow as tf

def calculate_cumulative_returns(returns, initial_investment = 100):

    cumulative_returns = np.cumprod(1 + returns, axis=0) - 1
    cumulative_returns = initial_investment * (1 + cumulative_returns)
    return cumulative_returns

def calculate_sharpe_ratio_from_loss(weights, returns):

    weights = np.array(weights)
    returns = np.array(returns)

    weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
    weights_tensor = tf.expand_dims(weights_tensor, axis=0)
    returns_tensor = tf.expand_dims(returns_tensor, axis=0)

    portfolio_returns = tf.reduce_sum(tf.multiply(returns_tensor, weights_tensor), axis=2)

    mean_return = tf.reduce_mean(portfolio_returns, axis=1)
    std_dev = tf.math.reduce_std(portfolio_returns, axis=1)

    epsilon = 1e-6
    adjusted_std_dev = tf.where(std_dev > epsilon, std_dev, epsilon)

    sharpe_ratio = mean_return / adjusted_std_dev
    loss = -tf.reduce_mean(sharpe_ratio)
    mean_return = tf.reduce_mean(mean_return).numpy()
    std_dev = tf.reduce_mean(std_dev).numpy()

    return mean_return, std_dev, -loss
