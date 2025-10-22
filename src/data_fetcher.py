"""
Data fetching module for stock market data from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_data(ticker, start_date=None, end_date=None, period='5y'):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        period (str): Period to fetch if dates not specified (e.g., '5y', '10y')
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    if start_date and end_date:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    else:
        data = yf.download(ticker, period=period, progress=False)
    
    return data


def fetch_vix_data(start_date=None, end_date=None, period='5y'):
    """
    Fetch VIX (Volatility Index) data from Yahoo Finance.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        period (str): Period to fetch if dates not specified
    
    Returns:
        pd.DataFrame: DataFrame with VIX data
    """
    if start_date and end_date:
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    else:
        vix = yf.download('^VIX', period=period, progress=False)
    
    return vix


def combine_stock_and_vix(stock_data, vix_data):
    """
    Combine stock data with VIX data.
    
    Args:
        stock_data (pd.DataFrame): Stock OHLCV data
        vix_data (pd.DataFrame): VIX data
    
    Returns:
        pd.DataFrame: Combined data
    """
    # Rename VIX close column to avoid conflicts
    vix_close = vix_data[['Close']].rename(columns={'Close': 'VIX'})
    
    # Merge on date index
    combined = stock_data.join(vix_close, how='left')
    
    # Forward fill VIX values for any missing dates
    combined['VIX'] = combined['VIX'].ffill()
    
    return combined
