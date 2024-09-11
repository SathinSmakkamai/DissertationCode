"""
File: chart.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: chart for strategy performance
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import pandas as pd
import seaborn as sns

class chart:

    def __init__(self, figsize = (10, 5), fontsize = 13, save_option = 1, result_path = 'results'):

        self.figsize = figsize
        self.fontsize = fontsize
        self.save_option = save_option
        self.result_path = result_path
        plt.rcParams.update({'font.size': self.fontsize})

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def save_and_show(self, filename):
        if self.save_option == 1:
            filepath = os.path.join(self.result_path, filename)
            plt.savefig(filepath)
        plt.show()

    def plot_training_progress(self, train_losses, val_losses):

        plt.figure(figsize=self.figsize)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        self.save_and_show('training_progress.png')

    def plot_predicted_weights(self, dates, predicted_weights, tickers):

        date_nums = np.arange(len(dates))
        abs_weights = np.abs(predicted_weights)
        normalized_weights = abs_weights / np.sum(abs_weights, axis=1, keepdims=True) * 100

        plt.figure(figsize=self.figsize)
        interp_func = interp1d(date_nums, normalized_weights, kind='linear', fill_value='extrapolate', axis=0)
        finer_date_nums = np.linspace(0, len(dates) - 1, num=1000)
        smoothed_weights = interp_func(finer_date_nums)
        finer_dates = [dates[int(round(x))] for x in finer_date_nums]
        plt.stackplot(finer_dates, smoothed_weights.T, labels=tickers, alpha=1)

        plt.xlabel('Date')
        plt.ylabel('Predicted Weights (%)')
        plt.title('Predicted Asset Weights Over Time')
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
        self.save_and_show('predicted_weights.png')

    def plot_explained_variance(self, explained_variance_df, no_components):

        explained_variance_df_percentage = explained_variance_df * 100
        avg_explained_variance = explained_variance_df_percentage.mean().mean()
        explained_variance = avg_explained_variance * no_components

        plt.figure(figsize=self.figsize)
        plt.stackplot(explained_variance_df_percentage.index,
                      explained_variance_df_percentage.T,
                      labels=['PC1', 'PC2', 'PC3'])

        plt.axhline(explained_variance, color='black', linestyle='--', linewidth=1,
                    label=f'Avg Explained Variance: {explained_variance:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Explained Variance (%)')
        plt.title('Explained Variance Ratio of Principal Components')
        plt.legend(loc='upper left')
        plt.grid()
        self.save_and_show('explained_variance.png')

    def plot_long_short_proportions(self, dates, predicted_weights):

        proportions = (predicted_weights > 0).mean(axis=1)
        long_proportions = proportions * 100
        short_proportions = (1 - proportions) * 100

        plt.figure(figsize=self.figsize)
        plt.stackplot(dates, long_proportions, short_proportions,
                      labels=['Long Proportion', 'Short Proportion'], colors=['g', 'r'], alpha=0.7)
        plt.axhline(50, color='black', linestyle='--', linewidth=1)

        plt.xlabel('Date')
        plt.ylabel('Proportion (%)')
        plt.title('Proportion of Long and Short Positions Over Time')
        plt.ylim(0, 100)
        plt.legend(loc='upper left')
        plt.grid()
        self.save_and_show('long_short_proportions.png')

    def plot_cumulative_return(self, dates, portfolio_cumulative_returns, benchmark_cumulative_returns):

        plt.figure(figsize=self.figsize)
        plt.plot(dates, portfolio_cumulative_returns, label='Predicted Weights Portfolio', color='b')
        plt.plot(dates, benchmark_cumulative_returns, label='Equal Weight Portfolio', color='r')

        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns ($)')
        plt.title('Cumulative Returns of the Predicted Weights Portfolio vs Equal Weight Portfolio')
        plt.legend(loc='upper left')
        plt.grid()
        self.save_and_show('cumulative_return.png')

    def plot_asset_contribution(self, dates, predicted_weights, tickers):

        avg_weights = np.mean(predicted_weights, axis=0)
        max_contrib_asset_idx = np.argmax(avg_weights)
        max_contrib_asset = tickers[max_contrib_asset_idx]

        plt.figure(figsize=self.figsize)
        plt.plot(dates, predicted_weights[:, max_contrib_asset_idx] * 100, label=max_contrib_asset, color='blue')

        avg_contrib_over_time = np.mean(predicted_weights[:, max_contrib_asset_idx]) * 100
        plt.axhline(avg_contrib_over_time, color='black', linestyle='--', linewidth=2, label='Average Contribution')
        plt.axhline(0, color='black', linestyle='--', linewidth = 1)

        plt.xlabel('Date')
        plt.ylabel(f'Predicted Weight (%) for {max_contrib_asset}')
        plt.title(f'Predicted Weight of the Asset with the Highest Contribution: {max_contrib_asset}')
        plt.legend(loc='upper left')
        plt.grid()
        self.save_and_show(f'{max_contrib_asset}_predicted_weight.png')

    def plot_normalized_price(self, price_data):

        start_dates = {ticker: price_data[ticker].dropna().index.min() for ticker in price_data.columns}
        latest_start_date = max(start_dates.values())
        filtered_price_data = price_data.loc[latest_start_date:]
        normalized_price_data = filtered_price_data / filtered_price_data.iloc[0] * 100

        plt.figure(figsize=self.figsize)

        for ticker in normalized_price_data.columns:
            plt.plot(normalized_price_data.index, normalized_price_data[ticker], label=ticker)

        plt.title('Normalized Price of Each Stock (Log Scale)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price (Log Scale)')
        plt.yscale('log')
        # plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.save_and_show('normalized_price.png')

    def plot_residuals_distribution(self, residuals):

        residuals = residuals.apply(pd.to_numeric, errors='coerce')
        residuals_flattened = residuals.values.flatten()
        residuals_flattened = residuals_flattened[~np.isnan(residuals_flattened)]

        mean_residual = residuals_flattened.mean()
        print(f"Mean of Residuals: {mean_residual}")

        plt.figure(figsize=self.figsize)
        sns.histplot(residuals_flattened, bins = 75, kde=True)
        plt.title('Distribution of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid()
        self.save_and_show('residuals_distribution.png')

    def plot_distribution_X_scaled(self, X_scaled, y_scaled):

        X_scaled = X_scaled.flatten()
        y_scaled = y_scaled.flatten()

        data = pd.DataFrame({
            'Value': np.concatenate([X_scaled, y_scaled]),
            'Type': ['X_processed'] * len(X_scaled) + ['y_processed'] * len(y_scaled)
        })

        plt.figure(figsize=self.figsize)
        sns.histplot(data, bins=75, kde=True)

        plt.title('Distribution of X_scaled')
        plt.xlabel('Scaled Values')
        plt.ylabel('Frequency')
        plt.grid()

        self.save_and_show('plot_distributionX_scaled.png')

    def plot_avg_training_vs_validation_loss(self, avg_training_losses, avg_validation_losses):

        epochs = len(avg_training_losses)
        plt.figure(figsize=self.figsize)
        plt.plot(range(epochs), avg_training_losses, label='Average Training Loss')
        plt.plot(range(epochs), avg_validation_losses, label='Average Validation Loss')
        plt.title('Average Training Loss vs Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        self.save_and_show('avg_training_vs_validation_loss.png')

    def plot_3d_validation_loss(self, validation_losses):

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        epochs = len(validation_losses)
        batches = len(validation_losses[0]) if epochs > 0 else 0
        X, Y = np.meshgrid(range(batches), range(epochs))
        Z = np.array(validation_losses)

        surf = ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.set_title('3D Validation Loss Across Epochs and Batches')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Epoch')
        ax.set_zlabel('Validation Loss')

        x_min, x_max = 0, batches - 1
        y_min, y_max = 0, epochs - 1
        z_min, z_max = np.min(Z), np.max(Z)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        fig_width, fig_height = fig.get_size_inches()
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        aspect_ratio = fig_width / fig_height
        scale_x = 1
        scale_y = 1 / aspect_ratio
        scale_z = (scale_x * x_range) / (scale_y * y_range)

        ax.set_xlim(x_min, x_min + scale_x * x_range)
        ax.set_ylim(y_min, y_min + scale_y * y_range)
        ax.set_zlim(z_min, z_min + scale_z * z_range)

        self.save_and_show('plot_3d_validation_loss.png')

    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------

    def plot_all(self, explained_variance_df, train_losses, val_losses, dates, predicted_weights, tickers,
                 portfolio_cumulative_returns, benchmark_cumulative_returns, no_components,
                 price_data, residuals, X_scaled, y_scaled, X, y):

        # self.plot_normalized_price(price_data)
        # self.plot_residuals_distribution(residuals)
        # self.plot_distribution_X_scaled(X_scaled, y_scaled)
        self.plot_explained_variance(explained_variance_df, no_components)
        self.plot_training_progress(train_losses, val_losses)
        self.plot_predicted_weights(dates, predicted_weights, tickers)
        self.plot_long_short_proportions(dates, predicted_weights)
        self.plot_asset_contribution(dates, predicted_weights, tickers)
        self.plot_cumulative_return(dates, portfolio_cumulative_returns, benchmark_cumulative_returns)







