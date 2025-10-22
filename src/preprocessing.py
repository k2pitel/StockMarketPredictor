"""
Data preprocessing module for scaling and preparation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle


class DataPreprocessor:
    """
    Preprocessor for stock market data with time-series aware scaling.
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type (str): 'standard' or 'minmax'
        """
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    def fit_transform(self, X_train):
        """
        Fit scaler on training data and transform it.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
        
        Returns:
            np.ndarray: Scaled training features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        return X_train_scaled
    
    def transform(self, X):
        """
        Transform data using fitted scaler.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features to transform
        
        Returns:
            np.ndarray: Scaled features
        """
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def save_scaler(self, filepath):
        """
        Save scaler to file.
        
        Args:
            filepath (str): Path to save scaler
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, filepath):
        """
        Load scaler from file.
        
        Args:
            filepath (str): Path to load scaler from
        """
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)


def create_time_series_splits(n_samples, n_splits=5):
    """
    Create time-series cross-validation splits.
    
    Args:
        n_samples (int): Number of samples
        n_splits (int): Number of splits
    
    Returns:
        TimeSeriesSplit: Time series split object
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv


def train_test_split_time_series(X, y, test_size=0.2):
    """
    Split data into train and test sets preserving time order.
    
    Args:
        X (pd.DataFrame or np.ndarray): Features
        y (pd.Series or np.ndarray): Target
        test_size (float): Proportion of data for testing
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test
