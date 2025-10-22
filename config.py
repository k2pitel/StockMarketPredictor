"""
Configuration file for Stock Market Predictor.

Customize these settings for your needs.
"""

# Default stock ticker
DEFAULT_TICKER = 'AAPL'

# Data settings
DEFAULT_PERIOD = '5y'  # Options: 1y, 2y, 5y, 10y, max
TEST_SIZE = 0.2  # 20% of data for testing

# Feature engineering
RSI_WINDOW = 14
MACD_SLOW = 26
MACD_FAST = 12
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# Moving averages
MA_WINDOWS = [5, 10, 20, 50, 200]
EMA_WINDOWS = [12, 26]

# Model settings
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'lbfgs'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

NEURAL_NETWORK_PARAMS = {
    'hidden_layers': [64, 32, 16],
    'dropout_rate': 0.3,
    'epochs': 50,
    'batch_size': 32
}

# Scaling method
SCALER_TYPE = 'standard'  # Options: 'standard', 'minmax'

# Time-series cross-validation
N_SPLITS = 5

# Random seed for reproducibility
RANDOM_SEED = 42
