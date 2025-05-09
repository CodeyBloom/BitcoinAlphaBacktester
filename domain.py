"""
Domain module for Bitcoin investment strategy calculations.

This module follows functional programming principles from "Grokking Simplicity":
- Functions are categorized as calculations (pure functions) or actions (with side effects)
- Data is treated as immutable
- Complex operations are composed of smaller, reusable functions
"""

import xgboost as xgb

import polars as pl
import numpy as np
from datetime import datetime

# ===== CALCULATIONS (PURE FUNCTIONS) =====

def calculate_investment_for_dca(is_sunday, weekly_investment):
    """
    Pure function to calculate investment amount for DCA strategy.
    
    Args:
        is_sunday (bool): Whether the current day is Sunday
        weekly_investment (float): Amount to invest weekly
        
    Returns:
        float: Amount to invest on this day
    """
    return weekly_investment if is_sunday else 0.0

def calculate_btc_bought(investment, price, exchange_id=None, use_discount=False):
    """
    Pure function to calculate BTC amount bought with a given investment,
    accounting for exchange fees if specified.
    
    Args:
        investment (float): Amount invested
        price (float): Bitcoin price
        exchange_id (str, optional): Exchange identifier for fee calculation
        use_discount (bool, optional): Whether to apply exchange discounts
        
    Returns:
        float: Amount of BTC bought
    """
    if price <= 0:
        return 0
        
    if exchange_id:
        try:
            from fee_models import calculate_transaction_cost, TransactionType
            
            # Calculate the net investment after fees
            net_investment, _ = calculate_transaction_cost(
                investment, 
                exchange_id, 
                TransactionType.BUY, 
                use_discount=use_discount
            )
            return net_investment / price
        except (ImportError, ValueError):
            # If fee calculation fails, fall back to no fees
            return investment / price
    else:
        # No exchange specified, assume no fees
        return investment / price

def calculate_moving_average(prices, window):
    """
    Pure function to calculate a moving average over a window of prices.
    
    Args:
        prices (np.array): Array of price values
        window (int): Window size for the moving average
    
    Returns:
        np.array: Array with moving averages (same length as prices)
    """
    result = np.zeros_like(prices)
    for i in range(len(prices)):
        if i >= window - 1:  # Changed this to >= window - 1 to include the current price
            result[i] = np.mean(prices[i-(window-1):i+1])
    return result

def calculate_rsi(prices, period=14):
    """
    Pure function to calculate RSI values for a series of prices.
    
    Args:
        prices (np.array): Array of price values
        period (int): Period for RSI calculation
    
    Returns:
        np.array: Array with RSI values (same length as prices)
    """
    # Calculate price changes
    delta = np.zeros_like(prices)
    delta[1:] = prices[1:] - prices[:-1]
    
    # Create gains and losses arrays
    gains = np.copy(delta)
    losses = np.copy(delta)
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = np.abs(losses)
    
    # Calculate average gains and losses
    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)
    
    # Initialize RSI array
    rsi = np.zeros_like(prices)
    
    # Calculate initial averages
    for i in range(period, len(prices)):
        if i == period:
            avg_gains[i] = np.mean(gains[1:i+1])
            avg_losses[i] = np.mean(losses[1:i+1])
        else:
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
        
        # Calculate RS and RSI
        if avg_losses[i] != 0:
            rs = avg_gains[i] / avg_losses[i]
        else:
            rs = 100  # Avoid division by zero
            
        rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_volatility(returns, window):
    """
    Pure function to calculate volatility (standard deviation) over a window of returns.
    
    Args:
        returns (np.array): Array of return values
        window (int): Window size for the volatility calculation
    
    Returns:
        np.array: Array with volatility values (same length as returns)
    """
    result = np.zeros_like(returns)
    for i in range(window, len(returns)):
        result[i] = np.nanstd(returns[i-window+1:i+1])
    return result

def calculate_price_drop(prices, lookback_period):
    """
    Pure function to calculate price drop percentage from maximum in lookback period.
    
    Args:
        prices (np.array): Array of price values
        lookback_period (int): Period to look back for maximum price
    
    Returns:
        np.array: Array with price drop percentages (same length as prices)
    """
    # Calculate rolling max prices
    rolling_max = np.zeros_like(prices)
    for i in range(lookback_period, len(prices)):
        rolling_max[i] = np.max(prices[i-lookback_period:i+1])
    
    # Calculate price drop percentages
    price_drop = np.zeros_like(prices)
    for i in range(lookback_period, len(prices)):
        if rolling_max[i] > 0:
            price_drop[i] = (rolling_max[i] - prices[i]) / rolling_max[i]
    
    return price_drop

def is_dip(price_drop, threshold):
    """
    Pure function to determine if a price drop constitutes a dip.
    
    Args:
        price_drop (float): Price drop percentage
        threshold (float): Threshold to consider a dip
    
    Returns:
        bool: True if the price drop exceeds the threshold
    """
    return price_drop >= threshold

def determine_rsi_investment_factor(rsi, oversold_threshold, overbought_threshold):
    """
    Pure function to determine investment factor based on RSI.
    
    Args:
        rsi (float): Current RSI value
        oversold_threshold (float): RSI threshold for oversold condition
        overbought_threshold (float): RSI threshold for overbought condition
    
    Returns:
        float: Investment factor (multiplier for base investment)
    """
    if rsi <= oversold_threshold:
        # Oversold - invest double
        return 2.0
    elif rsi >= overbought_threshold:
        # Overbought - invest half
        return 0.5
    else:
        # Normal range - linear scale between 0.5 and 2.0
        range_size = overbought_threshold - oversold_threshold
        position_in_range = rsi - oversold_threshold
        normalized_position = position_in_range / range_size
        return 2.0 - (normalized_position * 1.5)  # Scale from 2.0 down to 0.5

def determine_volatility_investment_factor(volatility, avg_volatility, threshold):
    """
    Pure function to determine investment factor based on volatility.
    
    Args:
        volatility (float): Current volatility
        avg_volatility (float): Average volatility
        threshold (float): Volatility threshold multiplier
    
    Returns:
        float: Investment factor (multiplier for base investment)
    """
    if avg_volatility <= 0:
        return 1.0
    
    vol_ratio = volatility / avg_volatility
    
    # When ratio is exactly at threshold, return 1.5 (exact threshold point)
    if vol_ratio == threshold:
        return 1.5
    elif vol_ratio > threshold:
        # Higher volatility - invest more (scaled between 1.5 and 2.0)
        # Scale from 1.5 (at threshold) up to 2.0 (at threshold*2)
        scale_factor = min(1.0, (vol_ratio - threshold) / threshold)
        return 1.5 + (0.5 * scale_factor)
    else:
        # Lower volatility - invest less (scaled between 0.5 and 1.5)
        # Scale from 0.5 (at 0) up to 1.5 (at threshold)
        scale_factor = vol_ratio / threshold
        return 0.5 + (scale_factor * 1.0)

def forward_fill_cumulative_values(dates, values):
    """
    Pure function to forward fill cumulative values.
    
    Args:
        dates (np.array): Array of dates
        values (np.array): Array of values to forward fill
    
    Returns:
        np.array: Array with forward-filled values
    """
    result = np.copy(values)
    last_valid_value = 0.0
    
    for i in range(len(values)):
        if values[i] != 0:
            last_valid_value = values[i]
        else:
            result[i] = last_valid_value
    
    return result

def find_sundays(dates):
    """
    Pure function to find Sundays in a date array.
    
    Args:
        dates (np.array): Array of datetime values
    
    Returns:
        np.array: Boolean array indicating which dates are Sundays
    """
    return np.array([d.weekday() == 6 for d in dates])

# ===== XGBOOST FEATURE CREATION AND PROCESSING =====

def create_xgboost_features(prices, returns, dates, day_of_week, window_sizes=[7, 14, 30]):
    """
    Pure function to create features for XGBoost model.
    
    Args:
        prices (np.array): Array of price values
        returns (np.array): Array of return values
        dates (np.array): Array of date values
        day_of_week (np.array): Array of day of week values (0-6)
        window_sizes (list): List of window sizes for feature calculation
    
    Returns:
        tuple: (features, column_names)
            - features: numpy array of feature values
            - column_names: list of feature names
    """
    # Create output arrays with enough space for all features
    num_features = len(window_sizes) * 3 + 2 + 3  # 3 features per window + 2 day features + 3 MACD features
    features = np.zeros((len(prices), num_features))
    column_names = []
    
    feature_idx = 0
    
    # Create day of week cyclical features
    features[:, feature_idx] = np.sin(2 * np.pi * day_of_week / 7)
    column_names.append("day_sin")
    feature_idx += 1
    
    features[:, feature_idx] = np.cos(2 * np.pi * day_of_week / 7)
    column_names.append("day_cos")
    feature_idx += 1
    
    # For each window size, calculate features
    for window in window_sizes:
        # Price momentum features - Rolling returns
        roll_returns = np.zeros_like(prices)
        for i in range(window, len(prices)):
            roll_returns[i] = prices[i] / prices[i - window] - 1
        
        features[:, feature_idx] = roll_returns
        column_names.append(f"return_{window}d")
        feature_idx += 1
        
        # Moving averages
        ma = calculate_moving_average(prices, window)
        features[:, feature_idx] = ma
        column_names.append(f"ma_{window}d")
        feature_idx += 1
        
        # Volatility
        vol = calculate_volatility(returns, window)
        features[:, feature_idx] = vol
        column_names.append(f"volatility_{window}d")
        feature_idx += 1
    
    # Calculate MACD-like features
    if len(window_sizes) >= 2:
        # Ensure we have at least short and long periods
        short_idx = column_names.index(f"ma_{min(window_sizes)}d")
        long_idx = column_names.index(f"ma_{max(window_sizes)}d")
        
        # MACD line
        macd = features[:, short_idx] - features[:, long_idx]
        features[:, feature_idx] = macd
        column_names.append("macd")
        feature_idx += 1
        
        # Signal line (9-day EMA of MACD)
        signal = calculate_moving_average(macd, 9)
        features[:, feature_idx] = signal
        column_names.append("macd_signal")
        feature_idx += 1
        
        # Histogram
        features[:, feature_idx] = macd - signal
        column_names.append("macd_hist")
        feature_idx += 1
    
    return features, column_names

def predict_returns_xgboost(model, features):
    """
    Pure function to predict returns using an XGBoost model.
    
    Args:
        model (xgb.Booster): Trained XGBoost model
        features (np.array): Feature array
    
    Returns:
        np.array: Predicted returns
    """
    # Convert features to DMatrix
    dmatrix = xgb.DMatrix(features)
    
    # Make predictions
    return model.predict(dmatrix)

def calculate_investment_factor_from_prediction(predicted_return, max_factor=2.0, min_factor=0.5):
    """
    Pure function to calculate investment factor based on predicted return.
    
    Args:
        predicted_return (float): Predicted return
        max_factor (float): Maximum investment factor
        min_factor (float): Minimum investment factor
    
    Returns:
        float: Investment factor (multiplier for base investment)
    """
    # Use sigmoid-like function to map predicted return to investment factor
    # Scale predicted return by 5 to amplify the effect
    factor = 1.0 + np.tanh(predicted_return * 5.0)
    
    # Clip factor to min/max
    return min(max_factor, max(min_factor, factor))

# ===== STRATEGY CALCULATIONS =====

def apply_dca_strategy(df, weekly_investment, exchange_id=None, use_discount=False):
    """
    Apply Dollar Cost Averaging strategy to price data.
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        exchange_id (str, optional): Exchange identifier for fee calculation
        use_discount (bool, optional): Whether to apply exchange discounts
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    # Create a copy of the dataframe (treating data as immutable)
    df = df.clone()
    
    # Add row index if it doesn't exist
    if "row_index" not in df.columns:
        df = df.with_row_index("row_index")
    
    # Convert to numpy arrays for faster processing
    is_sunday = df["is_sunday"].to_numpy()
    prices = df["price"].to_numpy()
    
    # Calculate investments (invest on Sundays)
    investments = np.array([calculate_investment_for_dca(is_sun, weekly_investment) 
                           for is_sun in is_sunday])
    
    # Calculate BTC bought (with exchange fees if specified)
    btc_bought = np.array([calculate_btc_bought(inv, price, exchange_id, use_discount) 
                          for inv, price in zip(investments, prices)])
    
    # Calculate cumulative values
    cumulative_investment = np.cumsum(investments)
    cumulative_btc = np.cumsum(btc_bought)
    
    # Add calculated columns to the dataframe
    df = df.with_columns([
        pl.Series(investments).alias("investment"),
        pl.Series(btc_bought).alias("btc_bought"),
        pl.Series(cumulative_investment).alias("cumulative_investment"),
        pl.Series(cumulative_btc).alias("cumulative_btc")
    ])
    
    # Add exchange information if provided
    if exchange_id:
        try:
            from fee_models import get_exchange_fee, TransactionType
            # Get the fee percentage for reference
            fee = get_exchange_fee(exchange_id, TransactionType.BUY, use_discount=use_discount)
            df = df.with_columns([
                pl.lit(exchange_id).alias("exchange"),
                pl.lit(fee).alias("fee_percentage")
            ])
        except (ImportError, ValueError):
            pass
    
    return df

def apply_value_averaging_strategy(df, weekly_base_investment, target_growth_rate):
    """
    Apply Value Averaging strategy to price data.
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_base_investment (float): Base amount to invest weekly
        target_growth_rate (float): Target monthly growth rate (decimal)
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    # Create a copy of the dataframe (treating data as immutable)
    df = df.clone()
    
    # Add row index if it doesn't exist
    if "row_index" not in df.columns:
        df = df.with_row_index("row_index")
    
    # Convert weekly to monthly target
    monthly_investment = weekly_base_investment * 4.33  # Average weeks per month
    
    # Initialize columns
    df = df.with_columns([
        pl.lit(0.0).alias("investment"),
        pl.lit(0.0).alias("target_portfolio_value"),
        pl.lit(0.0).alias("portfolio_value"),
        pl.lit(0.0).alias("btc_bought"),
        pl.lit(0.0).alias("cumulative_investment"),
        pl.lit(0.0).alias("cumulative_btc")
    ])
    
    # Only invest on Sundays
    sundays = df.filter(pl.col("is_sunday"))
    
    # If there are no Sundays, return the DataFrame as is
    if len(sundays) == 0:
        return df
    
    # Create arrays to track values for each Sunday
    sunday_indices = sundays["row_index"].to_numpy()
    sunday_prices = sundays["price"].to_numpy()
    sunday_investments = np.zeros(len(sunday_indices))
    sunday_btc_bought = np.zeros(len(sunday_indices))
    sunday_cumulative_investment = np.zeros(len(sunday_indices))
    sunday_cumulative_btc = np.zeros(len(sunday_indices))
    sunday_portfolio_value = np.zeros(len(sunday_indices))
    sunday_target_value = np.zeros(len(sunday_indices))
    
    # Counter for months
    month_counter = 0
    
    # For the first Sunday, do regular DCA
    sunday_investments[0] = weekly_base_investment
    sunday_btc_bought[0] = calculate_btc_bought(weekly_base_investment, sunday_prices[0])
    sunday_cumulative_btc[0] = sunday_btc_bought[0]
    sunday_cumulative_investment[0] = weekly_base_investment
    sunday_portfolio_value[0] = sunday_cumulative_btc[0] * sunday_prices[0]
    sunday_target_value[0] = monthly_investment / 4.33  # First week's target
    
    # For each subsequent Sunday
    for i in range(1, len(sunday_indices)):
        # Update cumulative values from previous Sunday
        sunday_cumulative_btc[i] = sunday_cumulative_btc[i-1]
        sunday_cumulative_investment[i] = sunday_cumulative_investment[i-1]
        
        # Every 4 Sundays (approximately monthly), increase the target
        if i % 4 == 0:
            month_counter += 1
        
        # Calculate target portfolio value (grows by target rate each month)
        sunday_target_value[i] = (monthly_investment * (i+1)/4.33) * (1 + target_growth_rate)**(month_counter)
        
        # Calculate current portfolio value before new investment
        sunday_portfolio_value[i] = sunday_cumulative_btc[i] * sunday_prices[i]
        
        # Determine investment needed to reach target
        investment_needed = sunday_target_value[i] - sunday_portfolio_value[i]
        
        # Limit maximum investment to 3x weekly base for safety
        max_investment = weekly_base_investment * 3
        min_investment = -max_investment  # Allow selling (negative investment) but with limits
        
        # Ensure investment is within bounds
        if investment_needed > max_investment:
            investment_needed = max_investment
        elif investment_needed < min_investment:
            investment_needed = min_investment
        
        # If investment is non-zero, adjust holdings
        if investment_needed != 0:
            sunday_investments[i] = investment_needed
            sunday_btc_bought[i] = calculate_btc_bought(investment_needed, sunday_prices[i])
            
            # Update cumulative values
            sunday_cumulative_investment[i] += sunday_investments[i]
            sunday_cumulative_btc[i] += sunday_btc_bought[i]
    
    # Create a mapping DataFrame for lookups
    sunday_map_df = pl.DataFrame({
        "row_index": sunday_indices,
        "investment_value": sunday_investments,
        "btc_bought_value": sunday_btc_bought,
        "cumulative_investment_value": sunday_cumulative_investment,
        "cumulative_btc_value": sunday_cumulative_btc,
        "target_value": sunday_target_value,
        "portfolio_value_on_sunday": sunday_portfolio_value
    })
    
    # Join the mapping dataframe to update values for Sunday rows
    df = df.join(
        sunday_map_df, 
        on="row_index", 
        how="left"
    )
    
    # Apply values for Sundays and keep zeros for non-Sundays
    df = df.with_columns([
        pl.when(pl.col("is_sunday"))
        .then(pl.col("investment_value"))
        .otherwise(0.0)
        .alias("investment")
    ])
    
    df = df.with_columns([
        pl.when(pl.col("is_sunday"))
        .then(pl.col("btc_bought_value"))
        .otherwise(0.0)
        .alias("btc_bought")
    ])
    
    df = df.with_columns([
        pl.when(pl.col("is_sunday"))
        .then(pl.col("target_value"))
        .otherwise(0.0)
        .alias("target_portfolio_value")
    ])
    
    df = df.with_columns([
        pl.when(pl.col("is_sunday"))
        .then(pl.col("portfolio_value_on_sunday"))
        .otherwise(0.0)
        .alias("portfolio_value")
    ])
    
    # Create a forward-fill function for cumulative values
    # For each row, we'll find the last Sunday and use its cumulative values
    def find_last_sunday_values(row_idx):
        prev_sundays = sunday_map_df.filter(
            pl.col("row_index") <= row_idx
        )
        if len(prev_sundays) > 0:
            last_sunday = prev_sundays[-1]
            return last_sunday["cumulative_investment_value"].item(), last_sunday["cumulative_btc_value"].item()
        else:
            return 0.0, 0.0
    
    # This is a slower approach but avoids the need for multiple iterations
    cumulative_values = []
    for row_idx in df["row_index"]:
        cumulative_investment, cumulative_btc = find_last_sunday_values(row_idx)
        cumulative_values.append((cumulative_investment, cumulative_btc))
    
    # Update cumulative columns
    df = df.with_columns([
        pl.Series([v[0] for v in cumulative_values]).alias("cumulative_investment"),
        pl.Series([v[1] for v in cumulative_values]).alias("cumulative_btc")
    ])
    
    # Drop temporary columns
    df = df.drop([
        "investment_value", 
        "btc_bought_value", 
        "cumulative_investment_value", 
        "cumulative_btc_value", 
        "target_value", 
        "portfolio_value_on_sunday"
    ])
    
    return df

def apply_maco_strategy(df, weekly_investment, short_window=20, long_window=100):
    """
    Apply Moving Average Crossover strategy to price data.
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        short_window (int): Short-term moving average window in days
        long_window (int): Long-term moving average window in days
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    # Create a copy of the dataframe (treating data as immutable)
    df = df.clone()
    
    # Add row index if it doesn't exist
    if "row_index" not in df.columns:
        df = df.with_row_index("row_index")
    
    # Convert data to numpy arrays for faster processing
    prices = df["price"].to_numpy()
    is_sunday = df["is_sunday"].to_numpy()
    
    # Calculate moving averages
    short_ma = calculate_moving_average(prices, short_window)
    long_ma = calculate_moving_average(prices, long_window)
    
    # Calculate signal (1 when short MA > long MA, otherwise 0)
    signal = np.zeros_like(prices)
    signal[short_ma > long_ma] = 1
    
    # Calculate signal changes (1 when signal changes from 0 to 1)
    signal_change = np.zeros_like(prices)
    signal_change[1:] = np.diff(signal)
    
    # Prepare arrays for storing results
    investment = np.zeros_like(prices)
    btc_bought = np.zeros_like(prices)
    cumulative_investment = np.zeros_like(prices)
    cumulative_btc = np.zeros_like(prices)
    
    # Weekly investment strategy
    accumulated_funds = 0
    
    for i in range(len(prices)):
        # Accumulate weekly investment on Sundays
        if is_sunday[i]:
            accumulated_funds += weekly_investment
            cumulative_investment[i] = cumulative_investment[i-1] if i > 0 else 0
        
        # Buy when there's a crossover to bullish and we have funds
        # Or on Sundays when we're in a bullish trend
        if ((signal_change[i] == 1 or (is_sunday[i] and signal[i] == 1)) 
            and accumulated_funds > 0 and i >= long_window):
            
            investment[i] = accumulated_funds
            btc_bought[i] = calculate_btc_bought(accumulated_funds, prices[i])
            
            # Update cumulative values
            if i > 0:
                cumulative_investment[i] = cumulative_investment[i-1] + accumulated_funds
                cumulative_btc[i] = cumulative_btc[i-1] + btc_bought[i]
            else:
                cumulative_investment[i] = accumulated_funds
                cumulative_btc[i] = btc_bought[i]
            
            accumulated_funds = 0
        
        # If no investment on this day, carry forward cumulative values
        elif i > 0:
            cumulative_investment[i] = cumulative_investment[i-1]
            cumulative_btc[i] = cumulative_btc[i-1]
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_funds > 0:
        i = len(prices) - 1
        investment[i] += accumulated_funds
        btc_bought[i] += calculate_btc_bought(accumulated_funds, prices[i])
        cumulative_investment[i] += accumulated_funds
        cumulative_btc[i] += calculate_btc_bought(accumulated_funds, prices[i])
    
    # Add calculated columns to the dataframe
    df = df.with_columns([
        pl.Series(short_ma).alias("short_ma"),
        pl.Series(long_ma).alias("long_ma"),
        pl.Series(signal).alias("signal"),
        pl.Series(signal_change).alias("signal_change"),
        pl.Series(investment).alias("investment"),
        pl.Series(btc_bought).alias("btc_bought"),
        pl.Series(cumulative_investment).alias("cumulative_investment"),
        pl.Series(cumulative_btc).alias("cumulative_btc")
    ])
    
    return df

def apply_rsi_strategy(df, weekly_investment, rsi_period=14, oversold_threshold=30, overbought_threshold=70):
    """
    Apply RSI-based investment strategy to price data.
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        rsi_period (int): Period for calculating RSI
        oversold_threshold (int): RSI threshold for oversold condition
        overbought_threshold (int): RSI threshold for overbought condition
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    # Create a copy of the dataframe (treating data as immutable)
    df = df.clone()
    
    # Add row index if it doesn't exist
    if "row_index" not in df.columns:
        df = df.with_row_index("row_index")
    
    # Convert data to numpy arrays for faster processing
    prices = df["price"].to_numpy()
    is_sunday = df["is_sunday"].to_numpy()
    
    # Calculate RSI
    rsi = calculate_rsi(prices, rsi_period)
    
    # Prepare arrays for storing results
    investment = np.zeros_like(prices)
    btc_bought = np.zeros_like(prices)
    cumulative_investment = np.zeros_like(prices)
    cumulative_btc = np.zeros_like(prices)
    
    # Weekly investment strategy
    accumulated_funds = 0
    
    for i in range(len(prices)):
        # Get current RSI (or 50 if not available yet)
        current_rsi = rsi[i] if i >= rsi_period else 50
        
        # Accumulate weekly investment on Sundays
        if is_sunday[i]:
            accumulated_funds += weekly_investment
        
        # Only start investing after RSI is available
        if i >= rsi_period and is_sunday[i]:
            # Determine investment factor based on RSI
            investment_factor = determine_rsi_investment_factor(
                current_rsi, oversold_threshold, overbought_threshold
            )
            
            # Calculate investment and BTC bought
            to_invest = min(accumulated_funds, weekly_investment * investment_factor)
            investment[i] = to_invest
            btc_bought[i] = calculate_btc_bought(to_invest, prices[i])
            
            # Update accumulated funds
            accumulated_funds -= to_invest
            
            # Update cumulative values
            if i > 0:
                cumulative_investment[i] = cumulative_investment[i-1] + to_invest
                cumulative_btc[i] = cumulative_btc[i-1] + btc_bought[i]
            else:
                cumulative_investment[i] = to_invest
                cumulative_btc[i] = btc_bought[i]
        
        # If no investment on this day, carry forward cumulative values
        elif i > 0:
            cumulative_investment[i] = cumulative_investment[i-1]
            cumulative_btc[i] = cumulative_btc[i-1]
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_funds > 0:
        i = len(prices) - 1
        investment[i] += accumulated_funds
        btc_bought[i] += calculate_btc_bought(accumulated_funds, prices[i])
        cumulative_investment[i] += accumulated_funds
        cumulative_btc[i] += calculate_btc_bought(accumulated_funds, prices[i])
    
    # Add calculated columns to the dataframe
    df = df.with_columns([
        pl.Series(rsi).alias("rsi"),
        pl.Series(investment).alias("investment"),
        pl.Series(btc_bought).alias("btc_bought"),
        pl.Series(cumulative_investment).alias("cumulative_investment"),
        pl.Series(cumulative_btc).alias("cumulative_btc")
    ])
    
    return df

def apply_volatility_strategy(df, weekly_investment, vol_window=14, vol_threshold=1.5):
    """
    Apply volatility-based investment strategy to price data.
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        vol_window (int): Window for volatility calculation
        vol_threshold (float): Threshold multiplier for increased investment
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    # Create a copy of the dataframe (treating data as immutable)
    df = df.clone()
    
    # Add row index if it doesn't exist
    if "row_index" not in df.columns:
        df = df.with_row_index("row_index")
    
    # Convert data to numpy arrays for faster processing
    prices = df["price"].to_numpy()
    returns = df["returns"].to_numpy()
    is_sunday = df["is_sunday"].to_numpy()
    
    # Calculate volatility (standard deviation of returns)
    volatility = calculate_volatility(returns, vol_window)
    
    # Calculate average volatility for comparison
    avg_volatility = calculate_volatility(volatility, vol_window*5)
    
    # Prepare arrays for storing results
    investment = np.zeros_like(prices)
    btc_bought = np.zeros_like(prices)
    cumulative_investment = np.zeros_like(prices)
    cumulative_btc = np.zeros_like(prices)
    
    # Weekly investment strategy
    accumulated_funds = 0
    
    for i in range(len(prices)):
        # Accumulate weekly investment on Sundays
        if is_sunday[i]:
            accumulated_funds += weekly_investment
        
        # Only start investing after volatility metrics are available
        if i >= vol_window*5 and is_sunday[i]:
            # Determine if current volatility is high
            current_vol = volatility[i]
            avg_vol = avg_volatility[i]
            
            if not np.isnan(current_vol) and not np.isnan(avg_vol) and avg_vol > 0:
                # Determine investment factor based on volatility
                investment_factor = determine_volatility_investment_factor(
                    current_vol, avg_vol, vol_threshold
                )
                
                # Calculate investment and BTC bought
                to_invest = min(accumulated_funds, weekly_investment * investment_factor)
                investment[i] = to_invest
                btc_bought[i] = calculate_btc_bought(to_invest, prices[i])
                
                # Update accumulated funds
                accumulated_funds -= to_invest
            else:
                # If volatility data not available, use standard DCA
                to_invest = min(accumulated_funds, weekly_investment)
                investment[i] = to_invest
                btc_bought[i] = calculate_btc_bought(to_invest, prices[i])
                accumulated_funds -= to_invest
            
            # Update cumulative values
            if i > 0:
                cumulative_investment[i] = cumulative_investment[i-1] + investment[i]
                cumulative_btc[i] = cumulative_btc[i-1] + btc_bought[i]
            else:
                cumulative_investment[i] = investment[i]
                cumulative_btc[i] = btc_bought[i]
        
        # If no investment on this day, carry forward cumulative values
        elif i > 0:
            cumulative_investment[i] = cumulative_investment[i-1]
            cumulative_btc[i] = cumulative_btc[i-1]
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_funds > 0:
        i = len(prices) - 1
        investment[i] += accumulated_funds
        btc_bought[i] += calculate_btc_bought(accumulated_funds, prices[i])
        cumulative_investment[i] += accumulated_funds
        cumulative_btc[i] += calculate_btc_bought(accumulated_funds, prices[i])
    
    # Add calculated columns to the dataframe
    df = df.with_columns([
        pl.Series(volatility).alias("volatility"),
        pl.Series(avg_volatility).alias("avg_volatility"),
        pl.Series(investment).alias("investment"),
        pl.Series(btc_bought).alias("btc_bought"),
        pl.Series(cumulative_investment).alias("cumulative_investment"),
        pl.Series(cumulative_btc).alias("cumulative_btc")
    ])
    
    return df

def apply_xgboost_strategy(df, weekly_investment=100.0, prediction_horizon=7, 
                          training_days=365, max_investment_factor=2.0, 
                          min_investment_factor=0.5, exchange_id=None, 
                          use_discount=False, window_sizes=[7, 14, 30]):
    """
    Apply XGBoost-based investment strategy to price data.
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Base amount to invest weekly
        prediction_horizon (int): Days ahead to predict returns
        training_days (int): Number of days to use for training
        max_investment_factor (float): Maximum multiplier for weekly investment
        min_investment_factor (float): Minimum multiplier for weekly investment
        exchange_id (str, optional): Exchange identifier for fee calculation
        use_discount (bool, optional): Whether to apply exchange discounts
        window_sizes (list): Window sizes for feature calculation
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    # Create a copy of the dataframe (treating data as immutable)
    df = df.clone()
    
    # Add row index if it doesn't exist
    if "row_index" not in df.columns:
        df = df.with_row_index("row_index")
    
    # Convert data to numpy arrays for faster processing
    dates = df["date"].to_numpy()
    prices = df["price"].to_numpy()
    returns = df["returns"].to_numpy()
    is_sunday = df["is_sunday"].to_numpy()
    day_of_week = df["day_of_week"].to_numpy()
    
    # Prepare arrays for storing results
    investment = np.zeros_like(prices)
    investment_factor = np.zeros_like(prices)
    btc_bought = np.zeros_like(prices)
    cumulative_investment = np.zeros_like(prices)
    cumulative_btc = np.zeros_like(prices)
    predicted_returns = np.zeros_like(prices)
    
    # Create features for XGBoost model
    features, feature_names = create_xgboost_features(prices, returns, dates, day_of_week, window_sizes)
    
    # Initialize weekly accumulation
    accumulated_funds = 0
    
    # Only start investing after sufficient data for training
    min_idx = max(training_days, max(window_sizes) * 2)
    
    # For each day after minimum index
    for i in range(min_idx, len(prices)):
        # Accumulate weekly investment on Sundays
        if is_sunday[i]:
            accumulated_funds += weekly_investment
        
        # Only predict and invest on Sundays after we have enough training data
        if i >= min_idx and is_sunday[i]:
            # Create target for training - future returns over prediction horizon
            future_returns = np.zeros(i - prediction_horizon)
            for j in range(i - prediction_horizon):
                if j + prediction_horizon < len(prices):
                    price_now = prices[j]
                    price_future = prices[j + prediction_horizon]
                    if price_now > 0:
                        future_returns[j] = price_future / price_now - 1
            
            # Create training data
            X_train = features[:i - prediction_horizon]
            y_train = future_returns
            
            # Filter out NaN values
            valid_idx = ~np.isnan(y_train)
            X_train = X_train[valid_idx]
            y_train = y_train[valid_idx]
            
            # Only train if we have enough valid data
            if len(y_train) > max(window_sizes):
                # Train model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror'
                )
                model.fit(X_train, y_train)
                
                # Get current features
                X_current = features[i].reshape(1, -1)
                
                # Predict future return
                pred_return = model.predict(X_current)[0]
                predicted_returns[i] = pred_return
                
                # Calculate investment factor based on predicted return
                factor = calculate_investment_factor_from_prediction(
                    pred_return, max_investment_factor, min_investment_factor
                )
                investment_factor[i] = factor
                
                # Calculate investment amount
                to_invest = min(accumulated_funds, weekly_investment * factor)
                investment[i] = to_invest
                btc_bought[i] = calculate_btc_bought(to_invest, prices[i], exchange_id, use_discount)
                
                # Update accumulated funds
                accumulated_funds -= to_invest
            else:
                # If insufficient data, use standard DCA
                to_invest = min(accumulated_funds, weekly_investment)
                investment[i] = to_invest
                btc_bought[i] = calculate_btc_bought(to_invest, prices[i], exchange_id, use_discount)
                accumulated_funds -= to_invest
            
            # Update cumulative values
            if i > 0:
                cumulative_investment[i] = cumulative_investment[i-1] + investment[i]
                cumulative_btc[i] = cumulative_btc[i-1] + btc_bought[i]
            else:
                cumulative_investment[i] = investment[i]
                cumulative_btc[i] = btc_bought[i]
        
        # If no investment on this day, carry forward cumulative values
        elif i > 0:
            cumulative_investment[i] = cumulative_investment[i-1]
            cumulative_btc[i] = cumulative_btc[i-1]
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_funds > 0:
        i = len(prices) - 1
        investment[i] += accumulated_funds
        btc_bought[i] += calculate_btc_bought(accumulated_funds, prices[i], exchange_id, use_discount)
        cumulative_investment[i] += accumulated_funds
        cumulative_btc[i] += calculate_btc_bought(accumulated_funds, prices[i], exchange_id, use_discount)
    
    # Add calculated columns to the dataframe
    df = df.with_columns([
        pl.Series(investment).alias("investment"),
        pl.Series(investment_factor).alias("investment_factor"),
        pl.Series(predicted_returns).alias("predicted_returns"),
        pl.Series(btc_bought).alias("btc_bought"),
        pl.Series(cumulative_investment).alias("cumulative_investment"),
        pl.Series(cumulative_btc).alias("cumulative_btc")
    ])
    
    return df