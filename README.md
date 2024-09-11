# Relative-Value Investment Strategy with PCA and Transformer on SET50 Thai Exchange Equities

### Dissertation Project for University College London
**Author**: Sathin Smakkamai  
**Supervisor**: Larrain Alexandre

---

## Introduction

This repository contains the code implementation of a **Relative-Value Investment Strategy** (RVP) developed as part of a dissertation for University College London. The study introduces a hybrid approach that combines traditional financial modeling with cutting-edge machine learning techniques to optimize portfolio performance.

The core strategy integrates:

- **Principal Component Analysis (PCA)**: A factor model that reduces dimensionality and captures key market factors.
- **Convolutional Neural Networks (CNN)**: For detecting local dependencies in time-series financial data.
- **Transformer Architectures**: To capture sequential patterns and global relationships within the data.

The model is trained with the objective of maximizing the **Sharpe Ratio** and applied to a selection of equities from the **SET50 Thai Index**. The aim is to explore how combining PCA with modern predictive models can improve portfolio optimization and deliver superior risk-adjusted returns.

---

## Features
- **PCA Factor Model**: For dimensionality reduction and identifying underlying market factors.
- **CNN and Transformer Models**: To capture both local and global dependencies in financial time series.
- **Sharpe Ratio Optimization**: The model is designed to maximize risk-adjusted returns.
- **Application to Thai Equities**: The focus of this study is on the SET50 Index.

---

## Repository Contents
- `main.py`: The main code for executing the investment strategy.
- `loss_functions.py`: Contains custom loss functions, including Sharpe Ratio loss and return loss.
- `training_set.py`: Defines the custom training set, taking into account sequential inputs for Sharpe Ratio loss calculation.
- `README.md`: This document.

---

## Acknowledgments
This project was completed as part of a dissertation for **University College London**, under the supervision of **Larrain Alexandre**.
