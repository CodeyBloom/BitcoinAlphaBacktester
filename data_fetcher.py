"""
Data fetcher module for Bitcoin price data.

This module follows functional programming principles from "Grokking Simplicity":
- Functions are categorized as calculations (pure functions) or actions (with side effects)
- Complex operations are composed of smaller, reusable functions
"""

import polars as pl
import httpx
import os
from datetime import datetime, date, timedelta
import time

# ===== UTILITY FUNCTIONS (PURE) =====

def parse_date_string(date_str):
    """
    Parse a date string in DD-MM-YYYY format.
    
    Args:
        date_str (str): Date string in DD-MM-YYYY format
        
    Returns:
        datetime.date: Parsed date object
    """
    return datetime.strptime(date_str, "%d-%m-%Y").date()

def get_arrow_path(currency="AUD"):
    """
    Get the path to the Arrow file for the specified currency.
    
    Args:
        currency (str): Currency code
        
    Returns:
        str: Path to the Arrow file
    """
    return os.path.join("data", f"bitcoin_prices.arrow")

def format_date_for_api(date_obj):
    """
    Format a date object as a Unix timestamp for the CoinGecko API.
    
    Args:
        date_obj (datetime.date): Date object
        
    Returns:
        int: Unix timestamp in seconds
    """
    return int(datetime.combine(date_obj, datetime.min.time()).timestamp())

def filter_dataframe_by_date_range(df, start_date, end_date):
    """
    Filter a DataFrame to only include rows within a date range.
    
    Args:
        df (polars.DataFrame): DataFrame with a 'date' column
        start_date (datetime.date): Start date (inclusive)
        end_date (datetime.date): End date (inclusive)
        
    Returns:
        polars.DataFrame: Filtered DataFrame
    """
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    return df.filter(
        (pl.col("date") >= start_datetime) & 
        (pl.col("date") <= end_datetime)
    )

def calculate_day_of_week(df):
    """
    Add day_of_week column to DataFrame.
    
    Args:
        df (polars.DataFrame): DataFrame with a 'date' column
        
    Returns:
        polars.DataFrame: DataFrame with day_of_week column
    """
    return df.with_columns(
        pl.col("date").dt.weekday().alias("day_of_week")
    )

def flag_sundays(df):
    """
    Add is_sunday column to DataFrame.
    
    Args:
        df (polars.DataFrame): DataFrame with a 'day_of_week' column
        
    Returns:
        polars.DataFrame: DataFrame with is_sunday column
    """
    return df.with_columns(
        (pl.col("day_of_week") == 7).alias("is_sunday")
    )

def calculate_returns(df):
    """
    Add returns column to DataFrame.
    
    Args:
        df (polars.DataFrame): DataFrame with a 'price' column
        
    Returns:
        polars.DataFrame: DataFrame with returns column
    """
    return df.with_columns(
        pl.col("price").pct_change().alias("returns")
    )

def ensure_row_index(df):
    """
    Add row_index column to DataFrame if it doesn't exist.
    
    Args:
        df (polars.DataFrame): DataFrame
        
    Returns:
        polars.DataFrame: DataFrame with row_index column
    """
    if "row_index" not in df.columns:
        return df.with_row_index("row_index")
    return df

# ===== API INTERACTION FUNCTIONS (ACTIONS WITH SIDE EFFECTS) =====

def read_from_arrow_file(file_path):
    """
    Read data from an Arrow file.
    
    Args:
        file_path (str): Path to the Arrow file
        
    Returns:
        polars.DataFrame or None: DataFrame if file exists, None otherwise
    """
    if os.path.exists(file_path):
        try:
            return pl.read_ipc(file_path)
        except Exception as e:
            print(f"Error reading Arrow file: {str(e)}")
            return None
    return None

def fetch_from_api(start_date, end_date, currency="AUD"):
    """
    Fetch Bitcoin historical price data from CoinGecko API.
    
    Args:
        start_date (date): Start date
        end_date (date): End date
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame: Dataframe with historical price data
    """
    print(f"Fetching data from API for {start_date} to {end_date}...")
    
    # CoinGecko API URL for Bitcoin market chart
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    
    # Convert dates to Unix timestamps (in seconds)
    start_timestamp = format_date_for_api(start_date)
    end_timestamp = format_date_for_api(end_date)
    
    params = {
        "vs_currency": currency.lower(),
        "from": start_timestamp,
        "to": end_timestamp
    }
    
    max_retries = 3
    for retry in range(max_retries):
        try:
            with httpx.Client() as client:
                response = client.get(url, params=params, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract price data
                prices = data.get("prices", [])
                
                if not prices:
                    print(f"No price data returned from API")
                    return None
                
                # Create DataFrame from price data
                df_prices = pl.DataFrame(
                    {"timestamp": [p[0] for p in prices], 
                     "price": [p[1] for p in prices]}
                )
                
                # Process the DataFrame
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
                
                # Add day of week and is_sunday columns
                daily_df = calculate_day_of_week(daily_df)
                daily_df = flag_sundays(daily_df)
                
                # Add returns column
                daily_df = calculate_returns(daily_df)
                
                # Add row_index column
                daily_df = ensure_row_index(daily_df)
                
                return daily_df
            
            elif response.status_code == 429:
                # Rate limit hit, wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limit hit, waiting for {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            else:
                print(f"API error: {response.status_code} - {response.text}")
                time.sleep(5)  # Wait before retry
        
        except Exception as e:
            print(f"Exception while fetching data: {str(e)}")
            time.sleep(5)  # Wait before retry
    
    print(f"Failed to fetch data after {max_retries} retries")
    return None

# ===== MAIN FUNCTION =====

def fetch_bitcoin_price_data(start_date_str, end_date_str, currency="AUD"):
    """
    Fetch Bitcoin historical price data from local Arrow file or CoinGecko API.
    
    Args:
        start_date_str (str): Start date in DD-MM-YYYY format
        end_date_str (str): End date in DD-MM-YYYY format
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame: Dataframe with historical price data
    """
    # Parse date strings
    start_date = parse_date_string(start_date_str)
    end_date = parse_date_string(end_date_str)
    
    # Try to read from local Arrow file first
    arrow_path = get_arrow_path(currency)
    df = read_from_arrow_file(arrow_path)
    
    if df is not None:
        # Filter to requested date range
        filtered_df = filter_dataframe_by_date_range(df, start_date, end_date)
        
        if len(filtered_df) > 0:
            print(f"Using local data with {len(filtered_df)} days in the requested date range")
            return filtered_df
    
    # If local data not available or insufficient, fetch from API
    return fetch_from_api(start_date, end_date, currency)