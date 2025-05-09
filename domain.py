"""
Domain module for Bitcoin investment strategy calculations.

This module follows functional programming principles from "Grokking Simplicity":
- Functions are categorized as calculations (pure functions) or actions (with side effects)
- Data is treated as immutable
- Complex operations are composed of smaller, reusable functions
"""

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
def prepare_features_for_ml(df, features, target_col="returns", target_horizon=1):
    """
    Pure function to prepare features for ML training and prediction.
    
    Args:
        df (polars.DataFrame): Price data with various columns
        features (list): List of column names to use as features
        target_col (str): Column to use for target values
        target_horizon (int): Number of days ahead to predict
        
    Returns:
        tuple: (X, y) where X is feature array and y is target array
    """
    # Create a copy of the dataframe (immutability principle)
    df = df.clone()
    
    # Create target values (future returns)
    if target_horizon > 0:
        target_values = df[target_col].shift(-target_horizon).to_numpy()
        # Convert to binary classification: 1 for positive return, 0 for negative
        target = np.where(target_values > 0, 1, 0)
    else:
        target = df[target_col].to_numpy()
    
    # Extract features
    feature_data = df.select(features).to_numpy()
    
    # Remove rows with NaN values
    valid_indices = ~np.isnan(target) & ~np.any(np.isnan(feature_data), axis=1)
    X = feature_data[valid_indices]
    y = target[valid_indices]
    
    return X, y

def apply_xgboost_ml_strategy(df, weekly_investment, training_window=14, prediction_threshold=0.55, features=None):
    """
    Apply XGBoost ML strategy to price data.
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday', 'returns' columns
        weekly_investment (float): Amount to invest weekly
        training_window (int): Number of days to use for initial training
        prediction_threshold (float): Confidence threshold for making investments
        features (list): List of column names to use as features, defaults to ["returns", "price"]
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    
    # Create a copy of the dataframe (treating data as immutable)
    df = df.clone()
    
    # Add row index if it doesn't exist
    if "row_index" not in df.columns:
        df = df.with_row_index("row_index")
    
    # Default features if none provided
    if features is None:
        features = ["returns", "price"]
        
    # Add some technical indicators as features
    prices = df["price"].to_numpy()
    returns = df["returns"].to_numpy()
    
    # 5-day moving average
    ma5 = calculate_moving_average(prices, 5)
    df = df.with_columns(pl.Series(ma5).alias("ma5"))
    
    # 20-day moving average
    ma20 = calculate_moving_average(prices, 20)
    df = df.with_columns(pl.Series(ma20).alias("ma20"))
    
    # Price relative to moving averages
    if "ma5" not in features and "ma5" in df.columns:
        features.append("ma5")
    if "ma20" not in features and "ma20" in df.columns:
        features.append("ma20")
    
    # 14-day RSI
    rsi14 = calculate_rsi(prices, 14)
    df = df.with_columns(pl.Series(rsi14).alias("rsi14"))
    if "rsi14" not in features and "rsi14" in df.columns:
        features.append("rsi14")
    
    # 14-day volatility
    vol14 = calculate_volatility(returns, 14)
    df = df.with_columns(pl.Series(vol14).alias("vol14"))
    if "vol14" not in features and "vol14" in df.columns:
        features.append("vol14")
    
    # Arrays for storing predictions and confidence levels
    predictions = np.zeros(len(df))
    confidences = np.zeros(len(df))
    
    # Arrays for investment decision
    is_sunday = df["is_sunday"].to_numpy()
    investment_factors = np.zeros(len(df))
    
    # Initialize the model with some data
    if len(df) > training_window:
        # Initial training data
        X_train, y_train = prepare_features_for_ml(
            df.slice(0, training_window), features, "returns", 1
        )
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Initialize and train model
        model = xgb.XGBClassifier(
            n_estimators=50, 
            learning_rate=0.1,
            max_depth=3,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Track if we have a trained model
        has_trained_model = False
        
        if len(X_train_scaled) > 0 and len(np.unique(y_train)) > 1:
            model.fit(X_train_scaled, y_train)
            has_trained_model = True
            
            # Make predictions for future days
            for i in range(training_window, len(df)):
                if i % 7 == 0:  # Retrain periodically
                    # Update training data with a sliding window
                    X_train, y_train = prepare_features_for_ml(
                        df.slice(max(0, i-training_window), i), features, "returns", 1
                    )
                    
                    # Retrain model if we have enough data and multiple classes
                    if len(X_train) > 0 and len(np.unique(y_train)) > 1:
                        X_train_scaled = scaler.fit_transform(X_train)
                        model.fit(X_train_scaled, y_train)
                        has_trained_model = True
                
                # Get current features for prediction
                X_current, _ = prepare_features_for_ml(
                    df.slice(i, i+1), features, "returns", 0
                )
                
                if len(X_current) > 0 and has_trained_model:
                    X_current_scaled = scaler.transform(X_current)
                    
                    # Get prediction
                    pred_proba = model.predict_proba(X_current_scaled)[0]
                    predictions[i] = 1 if pred_proba[1] > prediction_threshold else 0
                    confidences[i] = pred_proba[1]  # Probability of positive class
                    
                    # Set investment factor based on prediction and confidence
                    if is_sunday[i]:
                        if predictions[i] == 1:  # Model predicts price will go up
                            # Scale investment by confidence level (avoiding division by zero)
                            if prediction_threshold < 1.0:
                                confidence_boost = (confidences[i] - prediction_threshold) / (1 - prediction_threshold)
                                investment_factors[i] = 1.0 + min(1.0, confidence_boost)
                            else:
                                # If prediction_threshold is 1.0, use a simple approach
                                investment_factors[i] = 1.0 + (confidences[i] - 0.5)
                        else:
                            # Invest less when prediction is negative
                            investment_factors[i] = 0.5
    
    # Default investment factor for days we couldn't predict
    for i in range(len(df)):
        if is_sunday[i] and investment_factors[i] == 0:
            investment_factors[i] = 1.0
    
    # Calculate investments
    investments = np.array([
        weekly_investment * factor if is_sun else 0
        for is_sun, factor in zip(is_sunday, investment_factors)
    ])
    
    # Calculate BTC bought
    prices = df["price"].to_numpy()
    btc_bought = np.array([inv / price if price > 0 else 0 
                          for inv, price in zip(investments, prices)])
    
    # Calculate cumulative values
    cumulative_investment = np.cumsum(investments)
    cumulative_btc = np.cumsum(btc_bought)
    
    # Add calculated columns to the dataframe
    df = df.with_columns([
        pl.Series(predictions).alias("prediction"),
        pl.Series(confidences).alias("confidence"),
        pl.Series(investment_factors).alias("investment_factor"),
        pl.Series(investments).alias("investment"),
        pl.Series(btc_bought).alias("btc_bought"),
        pl.Series(cumulative_investment).alias("cumulative_investment"),
        pl.Series(cumulative_btc).alias("cumulative_btc")
    ])
    
    return df
