import pandas as pd
import numpy as np

def calculate_max_drawdown(df):
    """
    Calculate maximum drawdown in BTC terms
    
    Args:
        df (pandas.DataFrame): DataFrame with 'cumulative_btc' column
        
    Returns:
        float: Maximum drawdown as a decimal (0.0 to 1.0)
    """
    # Get cumulative BTC series
    cumulative_btc = df["cumulative_btc"]
    
    # Calculate running maximum
    running_max = cumulative_btc.cummax()
    
    # Calculate drawdown
    drawdown = (running_max - cumulative_btc) / running_max
    
    # Get maximum drawdown
    max_drawdown = drawdown.max()
    
    return max_drawdown if not pd.isna(max_drawdown) else 0.0

def calculate_sortino_ratio(df, risk_free_rate=0.0, minimum_acceptable_return=0.0, annualization_factor=252):
    """
    Calculate Sortino ratio in BTC terms
    
    Args:
        df (pandas.DataFrame): DataFrame with 'cumulative_btc' column
        risk_free_rate (float): Risk-free rate (default: 0)
        minimum_acceptable_return (float): Minimum acceptable return (default: 0)
        annualization_factor (int): Annualization factor (default: 252 trading days)
        
    Returns:
        float: Sortino ratio
    """
    # Calculate daily returns of BTC holdings
    btc_returns = df["cumulative_btc"].pct_change().dropna()
    
    if len(btc_returns) < 2:
        return 0.0
    
    # Calculate excess returns over minimum acceptable return
    excess_returns = btc_returns - minimum_acceptable_return
    
    # Calculate average excess return
    avg_excess_return = excess_returns.mean()
    
    # Calculate downside deviation (only consider returns below minimum acceptable return)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if avg_excess_return > 0 else 0.0
    
    downside_deviation = np.sqrt(np.sum(downside_returns ** 2) / len(downside_returns))
    
    # Calculate annualized downside deviation
    annualized_downside_deviation = downside_deviation * np.sqrt(annualization_factor)
    
    # Calculate annualized excess return
    annualized_excess_return = (1 + avg_excess_return) ** annualization_factor - 1
    
    # Calculate Sortino ratio
    sortino_ratio = (annualized_excess_return - risk_free_rate) / annualized_downside_deviation if annualized_downside_deviation > 0 else 0.0
    
    return sortino_ratio

def calculate_drawdown_over_time(df):
    """
    Calculate drawdown over time in BTC terms
    
    Args:
        df (pandas.DataFrame): DataFrame with 'cumulative_btc' column
        
    Returns:
        pandas.Series: Drawdown over time
    """
    # Get cumulative BTC series
    cumulative_btc = df["cumulative_btc"]
    
    # Calculate running maximum
    running_max = cumulative_btc.cummax()
    
    # Calculate drawdown
    drawdown = (running_max - cumulative_btc) / running_max
    
    return drawdown
