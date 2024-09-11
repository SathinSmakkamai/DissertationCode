"""
File: main_time_series_learning.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: Relative Value Investment Strategy with time-series learning method
"""

# import general library
import warnings
import time
import numpy as np
import yfinance as yf
import random
import tensorflow as tf
import inspect
import matplotlib.pyplot as plt
from tabulate import tabulate
from keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D

# import models library
from src.models.CNN_Transformer import CNNTFM_model
from src.models.LTSM import LSTM_model
from src.models.PCA import apply_rolling_pca_and_calculate_residuals
from src.model_setting.training_step import train_step
from src.model_setting.loss_function import max_return_loss, max_sharpe_ratio_loss
from src.preprocessor.preprocessor import preprocess_residuals, preprocess_scales_data
from src.preprocessor.split_data import split_data_by_date
from src.performance.portfolio_stat import calculate_cumulative_returns, calculate_sharpe_ratio_from_loss
from src.performance.chart import chart

# Suppress TensorFlow warnings and informational messages
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)


# Dataset
# top 50 Thai equities (SET50)
TH_50 = ['PTT.BK', 'ADVANC.BK', 'AOT.BK', 'DELTA.BK', 'PTTEP.BK', 'CPALL.BK', 'BDMS.BK',
       'KBANK.BK', 'BBL.BK', 'TISCO.BK', 'BAY.BK', 'KTB.BK', 'SCC.BK', 'TRUE.BK',
       'INTUCH.BK', 'LH.BK', 'TOP.BK', 'IRPC.BK', 'BANPU.BK', 'TU.BK', 'HMPRO.BK',
       'EGCO.BK', 'RATCH.BK', 'BCP.BK', 'CPF.BK', 'AP.BK', 'CENTEL.BK', 'MINT.BK',
       'BH.BK', 'BTS.BK', 'MAJOR.BK', 'CPN.BK', 'BJC.BK', 'STEC.BK', 'ITD.BK',
       'CK.BK', 'ROJNA.BK', 'AMATA.BK', 'SPALI.BK', 'KCE.BK', 'SAMART.BK', 'JAS.BK',
       'LPN.BK', 'SYNTEC.BK', 'BCH.BK', 'EGCO.BK', 'TTA.BK', 'HANA.BK', 'THANI.BK', 'STA.BK']

# top 30 Thai equities
TH_30 = ['PTT.BK', 'ADVANC.BK', 'DELTA.BK', 'PTTEP.BK', 'BDMS.BK', 'KBANK.BK',
         'BBL.BK', 'TISCO.BK', 'BAY.BK', 'KTB.BK', 'SCC.BK', 'TRUE.BK',
         'INTUCH.BK', 'LH.BK', 'IRPC.BK', 'BANPU.BK', 'TU.BK', 'HMPRO.BK',
         'EGCO.BK', 'RATCH.BK', 'BCP.BK', 'CPF.BK', 'AP.BK', 'CENTEL.BK',
         'MINT.BK', 'BH.BK', 'BTS.BK', 'MAJOR.BK', 'CPN.BK', 'BJC.BK']

# top 10 Thai equities
TH_10 = ['PTT.BK', 'ADVANC.BK', 'AOT.BK', 'DELTA.BK', 'PTTEP.BK', 'CPALL.BK', 'BDMS.BK',
        'KBANK.BK', 'BBL.BK', 'TISCO.BK']

# top 10 US equities
US_10 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'BRK-B', 'TSM', 'LLY', 'NVO', 'JPM', 'WMT']

# top 10 World equities, having historical price data dating back to 1975
SNC_1975 = ['JNJ', 'PG', 'DIS', 'KO', 'PEP', 'XOM', 'WMT', 'PFE', 'MRK', 'BAC']

# top 10 oil and petrol equities
OIL_10 = ['XOM', 'CVX', 'BP', 'COP', 'HES', 'OXY', 'SU', 'VLO', 'MRO', 'RRC']

# top 10 largest japan equities
JP_10 = ['7203.T', '6758.T', '9984.T', '8306.T', '9432.T', '6861.T', '7974.T', '7733.T', '4502.T', '8316.T']


def plot_3d_validation_loss1(validation_losses):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    epochs = len(validation_losses)
    batches = len(validation_losses[0]) if epochs > 0 else 0
    X, Y = np.meshgrid(range(batches), range(epochs))
    Z = np.array(validation_losses)

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_title('3D Validation Loss Across Epochs and Batches')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('Validation Loss')
    plt.show()


# select dataset
tickers = TH_10
loss_function = max_sharpe_ratio_loss
save_option = 1

# purpose of the model
stack_residual = True
scale_option = True

# factor model variables and preprocessor
factor_rolling_window = 250
covarience_rolling_window = 100
no_components = 5
sequence_length = 15

# traning variables
num_epochs = 1000
learning_rate = 0.00005
early_stop = 1
patience = 10

# time-series learning variables
batch_size = 500
validation_size = 100
rolling_increment = 50

# risk free rate
risk_free_rate = 0.0

# other variables
loss_data = sequence_length + max(factor_rolling_window,covarience_rolling_window) - 1

sharp_ratio_scaling_factor = np.sqrt(252 / sequence_length)
dataset_name = [name for name, value in inspect.currentframe().f_locals.items() if value is tickers][0]
loss_function_name = [name for name, value in inspect.currentframe().f_locals.items() if value is loss_function][0]

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def main():

    # Fetch and preprocess data
    price_data = yf.download(tickers, period = "max")['Adj Close']
    returns = price_data.pct_change().dropna()
    dates = returns.index

    # residual generation, factor model
    residuals, explained_variance_df = apply_rolling_pca_and_calculate_residuals(returns,
        factor_window = factor_rolling_window, covariance_window = covarience_rolling_window, n_components = no_components)

    # preprocess residual
    X, y, y_weights = preprocess_residuals(residuals, sequence_length = sequence_length, stack = stack_residual)

    # scaling residual
    X_scaled, y_scaled = preprocess_scales_data(X, y, scale = scale_option)

    # split the data to traning set and test set (out-of-sample)
    X_train, X_test, y_train, y_test = (
        split_data_by_date_no_val(X_scaled, y_scaled, y_weights, dates, test_start_date = '2020-01-01'))

    # model variables
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_assets = X_train.shape[2]

    # prediction model: transformerwith concolution layer or LSTM
    model = CNNTFM_model(input_shape, num_assets)

    optimizer = Adam(learning_rate=learning_rate)

    # Path for saving the best weights
    best_weights_path = f'weights/weight.weights.h5'

    # Custom training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    epoch_start_time_total = time.time()

    # variable storing traning loss
    train_losses = []
    val_losses = []

    # loop over each epochs
    for epoch in range(num_epochs):

        epoch_val_losses = []
        epoch_train_losses = []

        epoch_start_time = time.time()

        num_batches = (len(X_train) - batch_size) // rolling_increment + 1

        for i in range(num_batches):
            start_idx = i * rolling_increment
            end_idx = start_idx + batch_size

            if end_idx > len(X_train):
                break

            X_batch = X_train[start_idx:end_idx]
            y_weights_batch = y_train[start_idx:end_idx]
            actual_returns_batch = y[start_idx:end_idx]

            train_step(X_batch, y_weights_batch, actual_returns_batch, model, optimizer,
                       lookback = sequence_length, loss_function = loss_function)

            train_predictions = model.predict(X_batch, verbose = 0)
            train_loss = loss_function(train_predictions, actual_returns_batch)
            epoch_train_losses.append(train_loss.numpy())

            val_start_idx = end_idx
            val_end_idx = val_start_idx + validation_size

            if val_end_idx <= len(X_train):
                X_val = X_train[val_start_idx:val_end_idx]
                actual_returns_val = returns.iloc[val_start_idx:val_end_idx].values

                val_predictions = model.predict(X_val, verbose = 0)
                val_loss = loss_function(val_predictions, actual_returns_val)
                epoch_val_losses.append(val_loss.numpy())

        # Calculate test loss
        test_predictions = model.predict(X_test, verbose = 0)
        test_loss = loss_function(test_predictions, y_test)

        validation_losses.append(epoch_val_losses)
        training_losses.append(epoch_train_losses)

        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        avg_test_loss = test_loss.numpy()

        # Early stop machanism
        if early_stop == 1:

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

                # Save the best weights
                model.save_weights(best_weights_path)

            else:
                epochs_without_improvement += 1
                if early_stop and epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # logging for each epoch trained
        print(f"Epoch {epoch + 1} - "
              f"Average Training Loss: {-(avg_train_loss * sharp_ratio_scaling_factor):.4f} - "
              f"Average Validation Loss: {-(avg_val_loss * sharp_ratio_scaling_factor):.4f} - "
              f"Test Loss: {-(avg_test_loss * sharp_ratio_scaling_factor):.4f} - "
              f"Patience: {epochs_without_improvement} - Duration {epoch_duration:.2f} seconds")

    # Restore the best weights
    print('\n============================== training completed ==============================')
    if early_stop == 1 and epochs_without_improvement != 0:  # Only restore if early stopping was enabled
        model.load_weights(best_weights_path)
        best_weight_epoch = epoch + 1 - epochs_without_improvement
        print(f"Loaded best weights stored from epoach {best_weight_epoch}")
    else:
        best_weight_epoch = num_epochs
        print("Proceeding without restored weights.")

    epoch_end_time_total = time.time()
    epoch_duration_total = epoch_end_time_total - epoch_start_time_total

    hours, remainder = divmod(epoch_duration_total, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f'Total Duration: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds')
    print('================================================================================\n')

    # predict test set weights
    predicted_weights = model.predict(X_test, verbose = 1)

    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------

    recent_returns = returns.iloc[len(X_train) + loss_data:].values
    dates = returns.index[len(X_train) + loss_data:]

    predicted_portfolio_returns = []

    for i in range(predicted_weights.shape[0]):
        daily_weights = predicted_weights[i, :]
        daily_returns = recent_returns[i, :]
        daily_portfolio_return = np.sum(daily_weights * daily_returns)
        predicted_portfolio_returns.append(daily_portfolio_return)

    predicted_portfolio_returns = np.array(predicted_portfolio_returns)

    equal_weight_weights = np.ones(num_assets) / num_assets
    equal_weight_portfolio_returns = np.dot(recent_returns, equal_weight_weights)

    predicted_portfolio_cumulative_returns = (
        calculate_cumulative_returns(predicted_portfolio_returns, initial_investment = 100))

    equal_weight_portfolio_cumulative_returns = (
        calculate_cumulative_returns(equal_weight_portfolio_returns, initial_investment = 100))

    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------

    predicted_weights_return, predicted_weights_std, predicted_weights_sharpe = (
        calculate_sharpe_ratio_from_loss(predicted_weights, recent_returns))

    equal_weight_return, equal_weight_std, equal_weight_sharpe = (
        calculate_sharpe_ratio_from_loss(equal_weight_weights, recent_returns))

    data = [
        ["", "Return", "Standard Deviation", "Sharpe Ratio"],
        ["Predicted Weights Portfolio", f"{predicted_weights_return * 100:.5f}",
         f"{predicted_weights_std * sharp_ratio_scaling_factor:.5f}",
         f"{(predicted_weights_sharpe * sharp_ratio_scaling_factor):.5f}"],
        ["Equal Weight Portfolio", f"{equal_weight_return * 100:.5f}",
         f"{equal_weight_std * sharp_ratio_scaling_factor:.5f}",
         f"{equal_weight_sharpe * sharp_ratio_scaling_factor:.5f}"]]

    print(tabulate(data, headers = "firstrow", tablefmt = "grid"))

    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------

    plot_3d_validation_loss1(validation_losses)

    result_path = (f'results/TimeSeriesLearning/{loss_function_name}/{dataset_name}/'
                   f'SQ{sequence_length}_LR{learning_rate}_E({num_epochs})')

    cht = chart(save_option = save_option, result_path = result_path)

    # Call plot_all with the necessary arguments
    cht.plot_all(explained_variance_df, train_loss, val_loss, dates, predicted_weights, tickers,
                 predicted_portfolio_cumulative_returns, equal_weight_portfolio_cumulative_returns,
                 no_components, price_data, residuals, X_scaled, y_scaled, X, y)

    avg_training_losses = [np.mean(losses) for losses in training_losses]
    avg_validation_losses = [np.mean(losses) for losses in validation_losses]
    cht.plot_avg_training_vs_validation_loss(avg_training_losses, avg_validation_losses)

    cht.plot_3d_validation_loss(validation_losses)


if __name__ == "__main__":
    main()
