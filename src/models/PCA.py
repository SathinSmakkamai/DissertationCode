"""
File: PCA.py
Author: Sathin Smakkamai
Date: 11/09/2024
Description: predictive model: Principal component analysis (PCA) model
"""

import pandas as pd
from sklearn.decomposition import PCA

def apply_rolling_pca_and_calculate_residuals(returns, factor_window=50, covariance_window=21, n_components=3):

    residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
    explained_variance_ratios = []

    for start in range(len(returns) - max(factor_window, covariance_window) + 1):
        factor_end = start + factor_window
        covariance_end = start + covariance_window

        if factor_end <= len(returns):
            factor_window_data = returns.iloc[start:factor_end]
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(factor_window_data)
            explained_variance_ratios.append(pca.explained_variance_ratio_)

            if covariance_end <= len(returns):
                covariance_window_data = returns.iloc[start:covariance_end]
                reconstructed_returns = pca.inverse_transform(pca.transform(covariance_window_data))
                residuals.iloc[covariance_end - 1] = returns.iloc[covariance_end - 1] - reconstructed_returns[-1]

    explained_variance_df = pd.DataFrame(explained_variance_ratios, index=returns.index[factor_window - 1:len(
        explained_variance_ratios) + factor_window - 1])

    return residuals, explained_variance_df
