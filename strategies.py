import polars as pl
import numpy as np

def dca_strategy(df, weekly_investment):
    """
    Implement Dollar Cost Averaging strategy (buying a fixed amount on Sundays)
    
    Args:
        df (polars.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        
    Returns:
        polars.DataFrame: DataFrame with strategy results
    """
    # Create a copy of the dataframe
    df = df.clone()
    
    # Add row index if it doesn't exist
    if "row_index" not in df.columns:
        df = df.with_row_index("row_index")
    
    # Initialize investment column (invest on Sundays)
    df = df.with_columns(
        pl.when(pl.col("is_sunday"))
        .then(weekly_investment)
        .otherwise(0.0)
        .alias("investment")
    )
    
    # Calculate BTC bought
    df = df.with_columns(
        (pl.col("investment") / pl.col("price")).alias("btc_bought")
    )
    
    # Calculate cumulative values
    df = df.with_columns([
        pl.col("investment").cum_sum().alias("cumulative_investment"),
        pl.col("btc_bought").cum_sum().alias("cumulative_btc")
    ])
    
    return df

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
    sunday_btc_bought[0] = weekly_base_investment / sunday_prices[0]
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
            sunday_btc_bought[i] = investment_needed / sunday_prices[i]
            
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

def maco_strategy(df, weekly_investment, short_window=20, long_window=100):
    """
    Implement Moving Average Crossover strategy
    
    Args:
        df (pandas.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        short_window (int): Short-term moving average window in days
        long_window (int): Long-term moving average window in days
        
    Returns:
        pandas.DataFrame: DataFrame with strategy results
    """
    df = df.copy()
    
    # Calculate moving averages
    df["short_ma"] = df["price"].rolling(window=short_window).mean()
    df["long_ma"] = df["price"].rolling(window=long_window).mean()
    
    # Calculate crossover signal
    df["signal"] = 0
    df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1  # Buy signal when short MA > long MA
    
    # Calculate signal changes
    df["signal_change"] = df["signal"].diff()
    
    # Initialize columns
    df["investment"] = 0.0
    df["btc_bought"] = 0.0
    df["cumulative_investment"] = 0.0
    df["cumulative_btc"] = 0.0
    
    # Weekly investment amount (only on Sundays)
    total_sundays = df["is_sunday"].sum()
    accumulated_funds = 0
    
    for i in range(len(df)):
        # Accumulate weekly investment on Sundays
        if df.iloc[i]["is_sunday"]:
            accumulated_funds += weekly_investment
            df.loc[df.index[i], "cumulative_investment"] = df.iloc[i-1]["cumulative_investment"] if i > 0 else 0
        
        # Buy when there's a crossover to bullish and we have funds
        # Or on Sundays when we're in a bullish trend
        if ((df.iloc[i]["signal_change"] == 1 or (df.iloc[i]["is_sunday"] and df.iloc[i]["signal"] == 1)) 
            and accumulated_funds > 0 and i >= long_window):
            
            df.loc[df.index[i], "investment"] = accumulated_funds
            df.loc[df.index[i], "btc_bought"] = accumulated_funds / df.iloc[i]["price"]
            
            # Update cumulative values
            if i > 0:
                df.loc[df.index[i], "cumulative_investment"] = df.iloc[i-1]["cumulative_investment"] + accumulated_funds
                df.loc[df.index[i], "cumulative_btc"] = df.iloc[i-1]["cumulative_btc"] + df.iloc[i]["btc_bought"]
            else:
                df.loc[df.index[i], "cumulative_investment"] = accumulated_funds
                df.loc[df.index[i], "cumulative_btc"] = df.iloc[i]["btc_bought"]
            
            accumulated_funds = 0
        
        # If no investment on this day, carry forward cumulative values
        elif i > 0:
            df.loc[df.index[i], "cumulative_investment"] = df.iloc[i-1]["cumulative_investment"]
            df.loc[df.index[i], "cumulative_btc"] = df.iloc[i-1]["cumulative_btc"]
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_funds > 0:
        last_idx = df.index[-1]
        df.loc[last_idx, "investment"] += accumulated_funds
        df.loc[last_idx, "btc_bought"] += accumulated_funds / df.loc[last_idx, "price"]
        df.loc[last_idx, "cumulative_investment"] += accumulated_funds
        df.loc[last_idx, "cumulative_btc"] += accumulated_funds / df.loc[last_idx, "price"]
    
    return df

def rsi_strategy(df, weekly_investment, rsi_period=14, oversold_threshold=30, overbought_threshold=70):
    """
    Implement RSI-based investment strategy
    
    Args:
        df (pandas.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        rsi_period (int): Period for calculating RSI
        oversold_threshold (int): RSI threshold for oversold condition
        overbought_threshold (int): RSI threshold for overbought condition
        
    Returns:
        pandas.DataFrame: DataFrame with strategy results
    """
    df = df.copy()
    
    # Calculate RSI
    # First calculate price changes
    delta = df["price"].diff()
    
    # Create gains (up) and losses (down) series
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss over the specified period
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    # Calculate relative strength (RS) and RSI
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Initialize columns
    df["investment"] = 0.0
    df["btc_bought"] = 0.0
    df["cumulative_investment"] = 0.0
    df["cumulative_btc"] = 0.0
    
    # Strategy: Invest more when RSI is low (oversold), less when RSI is high (overbought)
    accumulated_funds = 0
    
    for i in range(len(df)):
        # Get current RSI (or 50 if not available yet)
        current_rsi = df.iloc[i]["rsi"] if not pl.Series([df.iloc[i]["rsi"]]).is_null().item() else 50
        
        # Accumulate weekly investment on Sundays
        if df.iloc[i]["is_sunday"]:
            accumulated_funds += weekly_investment
        
        # Only start investing after RSI is available
        if i >= rsi_period and df.iloc[i]["is_sunday"]:
            # Determine investment factor based on RSI
            if current_rsi <= oversold_threshold:
                # Oversold - invest double
                investment_factor = 2.0
            elif current_rsi >= overbought_threshold:
                # Overbought - invest half
                investment_factor = 0.5
            else:
                # Normal range - linear scale between 0.5 and 2.0
                range_size = overbought_threshold - oversold_threshold
                position_in_range = current_rsi - oversold_threshold
                normalized_position = position_in_range / range_size
                investment_factor = 2.0 - (normalized_position * 1.5)  # Scale from 2.0 down to 0.5
            
            # Calculate investment and BTC bought
            to_invest = min(accumulated_funds, weekly_investment * investment_factor)
            df.loc[df.index[i], "investment"] = to_invest
            df.loc[df.index[i], "btc_bought"] = to_invest / df.iloc[i]["price"]
            
            # Update accumulated funds
            accumulated_funds -= to_invest
            
            # Update cumulative values
            if i > 0:
                df.loc[df.index[i], "cumulative_investment"] = df.iloc[i-1]["cumulative_investment"] + to_invest
                df.loc[df.index[i], "cumulative_btc"] = df.iloc[i-1]["cumulative_btc"] + df.iloc[i]["btc_bought"]
            else:
                df.loc[df.index[i], "cumulative_investment"] = to_invest
                df.loc[df.index[i], "cumulative_btc"] = df.iloc[i]["btc_bought"]
        
        # If no investment on this day, carry forward cumulative values
        elif i > 0:
            df.loc[df.index[i], "cumulative_investment"] = df.iloc[i-1]["cumulative_investment"]
            df.loc[df.index[i], "cumulative_btc"] = df.iloc[i-1]["cumulative_btc"]
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_funds > 0:
        last_idx = df.index[-1]
        df.loc[last_idx, "investment"] += accumulated_funds
        df.loc[last_idx, "btc_bought"] += accumulated_funds / df.loc[last_idx, "price"]
        df.loc[last_idx, "cumulative_investment"] += accumulated_funds
        df.loc[last_idx, "cumulative_btc"] += accumulated_funds / df.loc[last_idx, "price"]
    
    return df

def volatility_strategy(df, weekly_investment, vol_window=14, vol_threshold=1.5):
    """
    Implement volatility-based investment strategy
    
    Args:
        df (pandas.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Amount to invest weekly
        vol_window (int): Window for volatility calculation
        vol_threshold (float): Threshold multiplier for increased investment
        
    Returns:
        pandas.DataFrame: DataFrame with strategy results
    """
    df = df.copy()
    
    # Calculate volatility (standard deviation of returns)
    df["volatility"] = df["returns"].rolling(window=vol_window).std()
    
    # Calculate average volatility for comparison
    df["avg_volatility"] = df["volatility"].rolling(window=vol_window*5).mean()
    
    # Initialize columns
    df["investment"] = 0.0
    df["btc_bought"] = 0.0
    df["cumulative_investment"] = 0.0
    df["cumulative_btc"] = 0.0
    
    # Strategy: Invest more during high volatility periods
    accumulated_funds = 0
    
    for i in range(len(df)):
        # Accumulate weekly investment on Sundays
        if df.iloc[i]["is_sunday"]:
            accumulated_funds += weekly_investment
        
        # Only start investing after volatility metrics are available
        if i >= vol_window*5 and df.iloc[i]["is_sunday"]:
            # Determine if current volatility is high
            current_vol = df.iloc[i]["volatility"]
            avg_vol = df.iloc[i]["avg_volatility"]
            
            if not pl.Series([current_vol]).is_null().item() and not pl.Series([avg_vol]).is_null().item() and avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                
                # Determine investment factor based on volatility
                if vol_ratio >= vol_threshold:
                    # Higher volatility - invest more
                    investment_factor = min(2.0, vol_ratio / vol_threshold)
                else:
                    # Lower volatility - invest less
                    investment_factor = max(0.5, vol_ratio / vol_threshold)
                
                # Calculate investment and BTC bought
                to_invest = min(accumulated_funds, weekly_investment * investment_factor)
                df.loc[df.index[i], "investment"] = to_invest
                df.loc[df.index[i], "btc_bought"] = to_invest / df.iloc[i]["price"]
                
                # Update accumulated funds
                accumulated_funds -= to_invest
            else:
                # If volatility data not available, use standard DCA
                to_invest = min(accumulated_funds, weekly_investment)
                df.loc[df.index[i], "investment"] = to_invest
                df.loc[df.index[i], "btc_bought"] = to_invest / df.iloc[i]["price"]
                accumulated_funds -= to_invest
            
            # Update cumulative values
            if i > 0:
                df.loc[df.index[i], "cumulative_investment"] = df.iloc[i-1]["cumulative_investment"] + df.iloc[i]["investment"]
                df.loc[df.index[i], "cumulative_btc"] = df.iloc[i-1]["cumulative_btc"] + df.iloc[i]["btc_bought"]
            else:
                df.loc[df.index[i], "cumulative_investment"] = df.iloc[i]["investment"]
                df.loc[df.index[i], "cumulative_btc"] = df.iloc[i]["btc_bought"]
        
        # If no investment on this day, carry forward cumulative values
        elif i > 0:
            df.loc[df.index[i], "cumulative_investment"] = df.iloc[i-1]["cumulative_investment"]
            df.loc[df.index[i], "cumulative_btc"] = df.iloc[i-1]["cumulative_btc"]
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_funds > 0:
        last_idx = df.index[-1]
        df.loc[last_idx, "investment"] += accumulated_funds
        df.loc[last_idx, "btc_bought"] += accumulated_funds / df.loc[last_idx, "price"]
        df.loc[last_idx, "cumulative_investment"] += accumulated_funds
        df.loc[last_idx, "cumulative_btc"] += accumulated_funds / df.loc[last_idx, "price"]
    
    return df

def lump_sum_strategy(df, weekly_investment, period_months=3, multiplier=12):
    """
    Implement periodic lump sum investment strategy
    
    Args:
        df (pandas.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Base weekly investment amount
        period_months (int): Number of months between investments
        multiplier (float): Multiplier for lump sum amount
        
    Returns:
        pandas.DataFrame: DataFrame with strategy results
    """
    df = df.copy()
    
    # Initialize columns
    df["investment"] = 0.0
    df["btc_bought"] = 0.0
    df["cumulative_investment"] = 0.0
    df["cumulative_btc"] = 0.0
    
    # Calculate period in weeks
    period_weeks = int(period_months * 4.33)  # Approximate weeks per month
    
    # Find all Sundays
    sundays = df[df["is_sunday"]]
    
    # Only invest on certain Sundays (every period_weeks)
    investment_sundays = []
    week_counter = 0
    
    for idx in sundays.index:
        week_counter += 1
        if week_counter % period_weeks == 0:
            investment_sundays.append(idx)
    
    # Calculate lump sum amount (equal to weekly_investment * weeks * multiplier)
    # This is to ensure the total investment is comparable to DCA
    lump_sum_amount = weekly_investment * period_weeks
    
    # Apply investment on selected Sundays
    accumulated_investment = 0
    
    for i, idx in enumerate(sundays.index):
        # Accumulate weekly equivalent
        accumulated_investment += weekly_investment
        
        # If this is an investment Sunday, invest the lump sum
        if idx in investment_sundays:
            to_invest = min(accumulated_investment, lump_sum_amount)
            df.loc[idx, "investment"] = to_invest
            df.loc[idx, "btc_bought"] = to_invest / df.loc[idx, "price"]
            accumulated_investment -= to_invest
        
        # Update cumulative values
        if i > 0:
            prev_idx = sundays.index[i-1]
            df.loc[idx, "cumulative_investment"] = df.loc[prev_idx, "cumulative_investment"] + df.loc[idx, "investment"]
            df.loc[idx, "cumulative_btc"] = df.loc[prev_idx, "cumulative_btc"] + df.loc[idx, "btc_bought"]
        else:
            df.loc[idx, "cumulative_investment"] = df.loc[idx, "investment"]
            df.loc[idx, "cumulative_btc"] = df.loc[idx, "btc_bought"]
    
    # Forward fill cumulative values for all days
    df["cumulative_investment"] = df["cumulative_investment"].replace(0, np.nan).fillna(method="ffill").fillna(0)
    df["cumulative_btc"] = df["cumulative_btc"].replace(0, np.nan).fillna(method="ffill").fillna(0)
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_investment > 0:
        last_idx = df.index[-1]
        df.loc[last_idx, "investment"] += accumulated_investment
        df.loc[last_idx, "btc_bought"] += accumulated_investment / df.loc[last_idx, "price"]
        df.loc[last_idx, "cumulative_investment"] += accumulated_investment
        df.loc[last_idx, "cumulative_btc"] += accumulated_investment / df.loc[last_idx, "price"]
    
    return df

def btd_strategy(df, weekly_investment, dip_threshold=0.1, lookback_period=7, multiplier=2.0):
    """
    Implement Buy The Dip investment strategy
    
    Args:
        df (pandas.DataFrame): Price data with 'date', 'price', 'is_sunday' columns
        weekly_investment (float): Base weekly investment amount
        dip_threshold (float): Price drop threshold to trigger buying (decimal)
        lookback_period (int): Period to look back for price drop
        multiplier (float): Multiplier for investment amount during dips
        
    Returns:
        pandas.DataFrame: DataFrame with strategy results
    """
    df = df.copy()
    
    # Initialize columns
    df["investment"] = 0.0
    df["btc_bought"] = 0.0
    df["cumulative_investment"] = 0.0
    df["cumulative_btc"] = 0.0
    
    # Calculate rolling max price for lookback period
    df["rolling_max_price"] = df["price"].rolling(window=lookback_period).max()
    
    # Calculate price drop percentage
    df["price_drop"] = (df["rolling_max_price"] - df["price"]) / df["rolling_max_price"]
    
    # Strategy: Accumulate funds and invest more when there's a dip
    accumulated_funds = 0
    
    for i in range(len(df)):
        # Accumulate weekly investment on Sundays
        if df.iloc[i]["is_sunday"]:
            accumulated_funds += weekly_investment
            
        # Check if we have enough data and funds
        if i >= lookback_period and accumulated_funds > 0:
            price_drop = df.iloc[i]["price_drop"]
            is_dip = not pl.Series([price_drop]).is_null().item() and price_drop >= dip_threshold
            
            # Decide investment amount
            if is_dip:
                # Dip detected - invest more
                to_invest = min(accumulated_funds, weekly_investment * multiplier)
                df.loc[df.index[i], "investment"] = to_invest
                df.loc[df.index[i], "btc_bought"] = to_invest / df.iloc[i]["price"]
                accumulated_funds -= to_invest
            elif df.iloc[i]["is_sunday"]:
                # Regular Sunday and no dip - invest regular amount
                to_invest = weekly_investment
                df.loc[df.index[i], "investment"] = to_invest
                df.loc[df.index[i], "btc_bought"] = to_invest / df.iloc[i]["price"]
                accumulated_funds -= to_invest
        
        # Update cumulative values
        if i > 0:
            df.loc[df.index[i], "cumulative_investment"] = df.iloc[i-1]["cumulative_investment"] + df.iloc[i]["investment"]
            df.loc[df.index[i], "cumulative_btc"] = df.iloc[i-1]["cumulative_btc"] + df.iloc[i]["btc_bought"]
        else:
            df.loc[df.index[i], "cumulative_investment"] = df.iloc[i]["investment"]
            df.loc[df.index[i], "cumulative_btc"] = df.iloc[i]["btc_bought"]
    
    # Make sure we invest any remaining funds on the last day to make fair comparison
    if accumulated_funds > 0:
        last_idx = df.index[-1]
        df.loc[last_idx, "investment"] += accumulated_funds
        df.loc[last_idx, "btc_bought"] += accumulated_funds / df.loc[last_idx, "price"]
        df.loc[last_idx, "cumulative_investment"] += accumulated_funds
        df.loc[last_idx, "cumulative_btc"] += accumulated_funds / df.loc[last_idx, "price"]
    
    return df
