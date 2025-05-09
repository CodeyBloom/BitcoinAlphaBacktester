"""
Strategies module for Bitcoin investment strategies.

This module provides the main strategy interfaces that are called by the application.
It uses pure functions from the domain module to implement the strategies.
"""

import polars as pl
import numpy as np
from domain import (
    apply_dca_strategy,
    apply_value_averaging_strategy,
    apply_maco_strategy,
    apply_rsi_strategy,
    apply_volatility_strategy,
    apply_xgboost_ml_strategy
)

def dca_strategy(df, weekly_investment, exchange_id=None, use_discount=False):
    """
    Implement Dollar Cost Averaging strategy (buying a fixed amount on Sundays)
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        exchange_id (str, optional): Exchange identifier for fee calculation
        use_discount (bool, optional): Whether to apply exchange discounts
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    return apply_dca_strategy(df, weekly_investment, exchange_id, use_discount)

def value_averaging_strategy(df, weekly_base_investment, target_growth_rate):
    """
    Implement Value Averaging strategy (adjusting investment to achieve target portfolio growth)
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_base_investment (float): Base amount to invest weekly
        target_growth_rate (float): Target monthly growth rate (decimal)
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    return apply_value_averaging_strategy(df, weekly_base_investment, target_growth_rate)

def maco_strategy(df, weekly_investment, short_window=20, long_window=100):
    """
    Implement Moving Average Crossover strategy
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        short_window (int): Short-term moving average window in days
        long_window (int): Long-term moving average window in days
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    return apply_maco_strategy(df, weekly_investment, short_window, long_window)

def rsi_strategy(df, weekly_investment, rsi_period=14, oversold_threshold=30, overbought_threshold=70):
    """
    Implement RSI-based investment strategy
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        rsi_period (int): Period for calculating RSI
        oversold_threshold (int): RSI threshold for oversold condition
        overbought_threshold (int): RSI threshold for overbought condition
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    return apply_rsi_strategy(df, weekly_investment, rsi_period, oversold_threshold, overbought_threshold)

def volatility_strategy(df, weekly_investment, vol_window=14, vol_threshold=1.5):
    """
    Implement volatility-based investment strategy
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        vol_window (int): Window for volatility calculation
        vol_threshold (float): Threshold multiplier for increased investment
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    return apply_volatility_strategy(df, weekly_investment, vol_window, vol_threshold)

def xgboost_ml_strategy(df, weekly_investment, training_window=14, prediction_threshold=0.55, features=None):
    """
    Implement XGBoost machine learning strategy for Bitcoin investment
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday', 'returns' columns
        weekly_investment (float): Amount to invest weekly
        training_window (int): Number of days to use for initial training
        prediction_threshold (float): Confidence threshold for making investments 
        features (list, optional): List of column names to use as features
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    return apply_xgboost_ml_strategy(df, weekly_investment, training_window, prediction_threshold, features)