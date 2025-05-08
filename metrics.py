import polars as pl
import numpy as np

def calculate_max_drawdown(df):
    """
    Calculate maximum drawdown in BTC terms
    
    Args:
        df (polars.DataFrame): DataFrame with 'cumulative_btc' column
        
    Returns:
        float: Maximum drawdown as a decimal (0.0 to 1.0)
    """
    # Create a running maximum expression
    cummax_expr = pl.col("cumulative_btc").cum_max()
    
    # Calculate drawdown with a window expression
    with_drawdown = df.with_columns([
        cummax_expr.alias("running_max"),
        ((cummax_expr - pl.col("cumulative_btc")) / cummax_expr).alias("drawdown")
    ])
    
    # Get maximum drawdown
    max_drawdown = with_drawdown.select(pl.max("drawdown")).item()
    
    # Handle null values
    return max_drawdown if max_drawdown is not None else 0.0

def calculate_sortino_ratio(df, risk_free_rate=0.0, minimum_acceptable_return=0.0, annualization_factor=252):
    """
    Calculate Sortino ratio in BTC terms
    
    Args:
        df (polars.DataFrame): DataFrame with 'cumulative_btc' column
        risk_free_rate (float): Risk-free rate (default: 0)
        minimum_acceptable_return (float): Minimum acceptable return (default: 0)
        annualization_factor (int): Annualization factor (default: 252 trading days)
        
    Returns:
        float: Sortino ratio
    """
    # Calculate daily returns of BTC holdings
    with_returns = df.with_columns(
        pl.col("cumulative_btc").pct_change().alias("btc_returns")
    )
    
    # Drop NaN values in returns
    returns_df = with_returns.filter(~pl.col("btc_returns").is_null())
    
    if len(returns_df) < 2:
        return 0.0
    
    # Calculate excess returns
    returns_df = returns_df.with_columns(
        (pl.col("btc_returns") - minimum_acceptable_return).alias("excess_returns")
    )
    
    # Calculate average excess return
    avg_excess_return = returns_df.select(pl.mean("excess_returns")).item()
    
    # Get downside returns (returns below minimum acceptable return)
    downside_returns = returns_df.filter(pl.col("excess_returns") < 0)
    
    if len(downside_returns) == 0:
        return float('inf') if avg_excess_return > 0 else 0.0
    
    # Calculate downside deviation
    squared_downside = downside_returns.with_columns(
        (pl.col("excess_returns") ** 2).alias("squared_excess")
    )
    
    sum_squared = squared_downside.select(pl.sum("squared_excess")).item()
    downside_deviation = np.sqrt(sum_squared / len(downside_returns))
    
    # Calculate annualized values
    annualized_downside_deviation = downside_deviation * np.sqrt(annualization_factor)
    annualized_excess_return = (1 + avg_excess_return) ** annualization_factor - 1
    
    # Calculate Sortino ratio
    sortino_ratio = (annualized_excess_return - risk_free_rate) / annualized_downside_deviation if annualized_downside_deviation > 0 else 0.0
    
    return sortino_ratio

def calculate_drawdown_over_time(df):
    """
    Calculate drawdown over time in BTC terms
    
    Args:
        df (polars.DataFrame): DataFrame with 'cumulative_btc' column
        
    Returns:
        polars.Series: Drawdown over time
    """
    # Create a running maximum expression
    cummax_expr = pl.col("cumulative_btc").cum_max()
    
    # Calculate drawdown with a window expression
    with_drawdown = df.with_columns([
        ((cummax_expr - pl.col("cumulative_btc")) / cummax_expr).alias("drawdown")
    ])
    
    return with_drawdown["drawdown"]
