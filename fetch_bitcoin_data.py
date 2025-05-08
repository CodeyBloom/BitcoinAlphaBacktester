import httpx
import polars as pl
from datetime import datetime, timedelta, date
import time
import os
import sys

def fetch_last_year_bitcoin_data(currency="AUD"):
    """
    Fetch Bitcoin historical price data from CoinGecko API for the last year.
    
    Args:
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame: Dataframe with historical price data
    """
    today = date.today()
    start_date = today - timedelta(days=364)  # API allows max 1 year
    
    # CoinGecko API URL for Bitcoin market chart
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    
    print(f"Fetching data from {start_date} to {today}...")
    
    # Convert dates to Unix timestamps (in seconds)
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_timestamp = int(datetime.combine(today, datetime.max.time()).timestamp())
    
    params = {
        "vs_currency": currency.lower(),
        "from": start_timestamp,
        "to": end_timestamp
    }
    
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            with httpx.Client() as client:
                response = client.get(url, params=params, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract price data
                prices = data.get("prices", [])
                
                if not prices:
                    print("No data returned from API")
                    retries += 1
                    time.sleep(5)
                    continue
                
                # Create DataFrame from price data
                df_prices = pl.DataFrame(
                    {"timestamp": [p[0] for p in prices], 
                     "price": [p[1] for p in prices]}
                )
                
                # Convert timestamp (milliseconds) to datetime
                df_prices = df_prices.with_columns(
                    pl.from_epoch("timestamp", time_unit="ms").alias("date")
                )
                
                # Drop timestamp column
                df_prices = df_prices.drop("timestamp")
                
                # Sort by date
                df_prices = df_prices.sort("date")
                
                # Resample to daily data (last price of each day)
                daily_df = df_prices.group_by_dynamic("date", every="1d").agg(
                    pl.last("price").alias("price")
                )
                
                # Add day of week
                daily_df = daily_df.with_columns(
                    pl.col("date").dt.weekday().alias("day_of_week")
                )
                
                # Add is_sunday flag
                daily_df = daily_df.with_columns(
                    (pl.col("day_of_week") == 6).alias("is_sunday")
                )
                
                # Add returns column
                daily_df = daily_df.with_columns(
                    pl.col("price").pct_change().alias("returns")
                )
                
                # Add row_index (for strategy calculations)
                daily_df = daily_df.with_row_index("row_index")
                
                return daily_df
                
            elif response.status_code == 429:
                # Rate limit hit, wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limit hit, waiting for {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            else:
                print(f"Error fetching data: {response.status_code} - {response.text}")
                retries += 1
                time.sleep(5)
                
        except Exception as e:
            print(f"Exception while fetching data: {str(e)}")
            retries += 1
            time.sleep(5)
    
    print(f"Failed to fetch data after {max_retries} retries")
    return None

def simulate_historical_data(df, years_to_simulate=9):
    """
    Since CoinGecko's free API limits to 1 year of data, this function
    simulates older historical data using the returns pattern of the available year.
    This is for demonstration purposes only and should be replaced with real data in production.
    
    Args:
        df (polars.DataFrame): DataFrame with 1 year of daily price data
        years_to_simulate (int): Number of additional years to simulate
        
    Returns:
        polars.DataFrame: DataFrame with simulated historical data
    """
    # First ensure we have a full year of data to work with
    if len(df) < 360:  # Allow some missing days
        print("Not enough data to simulate historical prices")
        return df
    
    # Get returns data for the year we have
    actual_returns = df["returns"].drop_nulls().to_numpy()
    
    # Use the row with the oldest date as starting point
    oldest_row = df.sort("date").row(0)
    oldest_date = oldest_row["date"]
    oldest_price = oldest_row["price"]
    
    # Generate dates going back years_to_simulate years
    start_date = oldest_date - timedelta(days=365 * years_to_simulate)
    date_range = []
    current_date = start_date
    
    while current_date < oldest_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Create simulated prices using the return pattern
    simulated_prices = []
    current_price = oldest_price
    
    # Work backwards using returns to simulate older prices
    for i in range(len(date_range)):
        # Use modulo to cycle through the returns for repetition
        daily_return = actual_returns[i % len(actual_returns)]
        # To go backwards in time, we divide by (1 + return) instead of multiplying
        if not np.isnan(daily_return):
            current_price = current_price / (1 + daily_return)
        simulated_prices.append(current_price)
    
    # Reverse to get chronological order
    date_range.reverse()
    simulated_prices.reverse()
    
    # Create DataFrame with simulated data
    simulated_df = pl.DataFrame({
        "date": date_range,
        "price": simulated_prices
    })
    
    # Add day of week and is_sunday columns
    simulated_df = simulated_df.with_columns(
        pl.col("date").dt.weekday().alias("day_of_week")
    )
    
    simulated_df = simulated_df.with_columns(
        (pl.col("day_of_week") == 6).alias("is_sunday")
    )
    
    # Add returns column
    simulated_df = simulated_df.with_columns(
        pl.col("price").pct_change().alias("returns")
    )
    
    # Combine simulated data with actual data
    combined_df = pl.concat([simulated_df, df])
    
    # Add row_index column
    combined_df = combined_df.with_row_index("row_index")
    
    return combined_df

def main():
    # Parse arguments
    currency = "AUD"
    output_file = "bitcoin_prices.arrow"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--currency" and i+2 <= len(sys.argv):
                currency = sys.argv[i+2]
            elif arg == "--output" and i+2 <= len(sys.argv):
                output_file = sys.argv[i+2]
    
    print(f"Fetching Bitcoin price data in {currency} for the last year...")
    
    # Fetch data for the last year (API limit)
    df = fetch_last_year_bitcoin_data(currency=currency)
    
    if df is not None and len(df) > 0:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save to Arrow file
        file_path = os.path.join("data", output_file)
        df.write_ipc(file_path)
        
        print(f"Successfully saved {len(df)} days of Bitcoin price data to {file_path}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    # Import numpy only when needed
    import numpy as np
    main()