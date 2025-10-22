"""
Feature engineering module for technical indicators.
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Ensure we have the necessary columns
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    # RSI (Relative Strength Index)
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    df['BB_mid'] = bollinger.bollinger_mavg()
    df['BB_width'] = df['BB_high'] - df['BB_low']
    df['BB_position'] = (df['Close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'])
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Price momentum
    df['momentum_1'] = df['Close'].pct_change(1)
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    
    # Volume features
    if 'Volume' in df.columns:
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
    
    return df


def create_target_variable(df, horizon=1):
    """
    Create binary target variable: 1 if stock rises tomorrow, 0 otherwise.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
        horizon (int): Number of days ahead to predict
    
    Returns:
        pd.DataFrame: DataFrame with target variable
    """
    df = df.copy()
    
    # Calculate future return
    df['future_return'] = df['Close'].shift(-horizon) - df['Close']
    
    # Binary classification: 1 if price rises, 0 otherwise
    df['target'] = (df['future_return'] > 0).astype(int)
    
    # Remove the last row(s) since we don't have future data
    df = df.iloc[:-horizon]
    
    return df


def prepare_features(df):
    """
    Prepare feature columns for model training.
    
    Args:
        df (pd.DataFrame): DataFrame with all indicators
    
    Returns:
        tuple: (feature_columns, df_clean)
    """
    # Define feature columns (excluding OHLCV base columns and target)
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'BB_high', 'BB_low', 'BB_mid', 'BB_width', 'BB_position',
        'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_200',
        'EMA_12', 'EMA_26',
        'momentum_1', 'momentum_5', 'momentum_10',
        'volume_change', 'volume_ma_5', 'volume_ma_20'
    ]
    
    # Add VIX if available
    if 'VIX' in df.columns:
        feature_columns.append('VIX')
    
    # Remove columns that don't exist
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Drop rows with NaN values (from rolling windows)
    df_clean = df.dropna()
    
    return feature_columns, df_clean
