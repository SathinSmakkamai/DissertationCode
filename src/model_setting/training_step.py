"""
File: training_step.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: A custom training set, taking into account sequential input to calculate Sharpe ratio loss.
"""

import tensorflow as tf

def train_step(X_batch, y_weights_batch, actual_returns, model, optimizer, lookback, loss_function):

    with tf.GradientTape() as tape:
        predicted_weights = model(X_batch, training = True)
        normalized_weights = predicted_weights

        # Define the risk_free_rate
        risk_free_rate = 0.00

        # loss for each batch
        batch_losses = []

        for i in range(X_batch.shape[0]):

            returns_slice = actual_returns[i:i + lookback, :X_batch.shape[2]]
            loss = loss_function(normalized_weights[i], returns_slice)
            batch_losses.append(loss)

        loss = tf.reduce_mean(batch_losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
