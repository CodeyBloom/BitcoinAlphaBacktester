#!/usr/bin/env python3
"""
Script to fetch 10 years of Bitcoin price data using the CryptoCompare API.
This API provides free access to historical cryptocurrency data without the 365-day
limitation that exists in CoinGecko's free tier.

Usage:
    python scripts/fetch_cryptocompare_data.py [--currency AUD] [--years 10]

The script will save the data to data/bitcoin_prices.arrow
"""

import os
import sys
import argparse
import httpx
import polars as pl
from datetime import datetime, timedelta, date
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def fetch_historical_daily_data(start_date, end_date, currency="AUD"):
    """
    Fetch Bitcoin historical daily price data from CryptoCompare API.
    
    Args:
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame or None: DataFrame with historical price data or None on failure
    """
    # CryptoCompare API base URL for historical daily data
    base_url = "https://min-api.cryptocompare.com/data/v2/histoday"
    
    # Calculate number of days in the range
    days_range = (end_date - start_date).days + 1
    
    print(f"Fetching {days_range} days of data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Set up parameters for the API request
    params = {
        "fsym": "BTC",  # From Symbol (Bitcoin)
        "tsym": currency.upper(),  # To Symbol (currency)
        "limit": 2000,  # Max 2000 days per request
        "aggregate": 1,  # 1 day aggregation
        "toTs": int(datetime.combine(end_date, datetime.max.time()).timestamp())  # End timestamp
    }
    
    all_data = []
    current_end = end_date
    
    # Keep fetching data until we've covered the entire date range
    while current_end >= start_date:
        print(f"Fetching data chunk ending on {current_end.strftime('%Y-%m-%d')}...")
        
        # Update the end timestamp for this request
        params["toTs"] = int(datetime.combine(current_end, datetime.max.time()).timestamp())
        
        max_retries = 3
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                response = httpx.get(base_url, params=params, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if we got valid data
                    if data.get("Response") == "Success":
                        daily_data = data.get("Data", {}).get("Data", [])
                        
                        if not daily_data:
                            print("No data returned from API")
                            break
                        
                        # Filter out timestamps with zero values (sometimes included in the API response)
                        daily_data = [d for d in daily_data if d.get("close", 0) != 0]
                        
                        # Add to our collection
                        all_data.extend(daily_data)
                        
                        # Get the timestamp of the earliest data point
                        earliest_timestamp = min(d["time"] for d in daily_data)
                        earliest_date = datetime.fromtimestamp(earliest_timestamp).date()
                        
                        # Move the end date back for the next request
                        current_end = earliest_date - timedelta(days=1)
                        
                        print(f"  Retrieved {len(daily_data)} days of data")
                        
                        # If we've reached or gone past our start date, we're done
                        if earliest_date <= start_date:
                            break
                        
                        # Be nice to the API - add a delay
                        time.sleep(1)
                        success = True
                    else:
                        error_msg = data.get("Message", "Unknown error")
                        print(f"API error: {error_msg}")
                        retries += 1
                        time.sleep(2)
                
                elif response.status_code == 429:
                    # Rate limit hit, wait and retry
                    print(f"Rate limit hit, waiting for 60 seconds...")
                    time.sleep(60)
                    
                else:
                    print(f"Error fetching data: {response.status_code} - {response.text}")
                    retries += 1
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Exception while fetching data: {str(e)}")
                retries += 1
                time.sleep(5)
                
        if retries >= max_retries and not success:
            print(f"Failed to fetch data chunk after {max_retries} retries")
            # Continue with what we've got so far rather than failing completely
            break
    
    # Check if we got any data
    if not all_data:
        print("No data retrieved from API")
        return None
    
    # Process the data into a DataFrame
    try:
        # Create DataFrame
        df = pl.DataFrame({
            "timestamp": [d["time"] for d in all_data],
            "price": [d["close"] for d in all_data]
        })
        
        # Convert timestamp to datetime
        df = df.with_columns(
            pl.from_epoch("timestamp").alias("date")
        )
        
        # Drop timestamp column
        df = df.drop("timestamp")
        
        # Sort by date
        df = df.sort("date")
        
        # Filter to the requested date range
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        df = df.filter(
            (pl.col("date") >= start_datetime) & 
            (pl.col("date") <= end_datetime)
        )
        
        # Add day of week column
        df = df.with_columns(
            pl.col("date").dt.weekday().alias("day_of_week")
        )
        
        # Add is_sunday column (where 7 is Sunday in our system)
        df = df.with_columns(
            (pl.col("day_of_week") == 6).alias("is_sunday")
        )
        
        # Add returns column
        df = df.with_columns(
            pl.col("price").pct_change().fill_null(0).alias("returns")
        )
        
        # Add row_index column
        df = df.with_row_index("row_index")
        
        print(f"Successfully processed {len(df)} days of data "
              f"from {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

def fetch_bitcoin_data_for_years(years=10, currency="AUD"):
    """
    Fetch Bitcoin historical price data for a specified number of years.
    
    Args:
        years (int): Number of years of data to fetch
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame or None: DataFrame with historical price data or None on failure
    """
    # Calculate date range
    end_date = date.today()
    start_date = end_date.replace(year=end_date.year - years)
    
    print(f"Fetching {years} years of Bitcoin price data from {start_date} to {end_date}...")
    
    # Fetch the data
    return fetch_historical_daily_data(start_date, end_date, currency)

def main():
    parser = argparse.ArgumentParser(description="Fetch historical Bitcoin price data from CryptoCompare API")
    parser.add_argument("--currency", type=str, default="AUD", help="Currency (default: AUD)")
    parser.add_argument("--years", type=int, default=10, help="Number of years of data to fetch (default: 10)")
    parser.add_argument("--output", type=str, default="data/bitcoin_prices.arrow", help="Output file path")
    
    args = parser.parse_args()
    
    print(f"Attempting to fetch {args.years} years of Bitcoin price data in {args.currency}...")
    
    # Fetch the requested years of data
    df = fetch_bitcoin_data_for_years(years=args.years, currency=args.currency)
    
    if df is not None and len(df) > 0:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save to Arrow file
        df.write_ipc(args.output)
        
        print(f"Successfully saved {len(df)} days of Bitcoin price data to {args.output}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Report data coverage
        actual_days = (df['date'].max() - df['date'].min()).days
        actual_years = actual_days / 365.25
        print(f"Data spans {actual_days} days (approximately {actual_years:.2f} years)")
        
        # Check if we got the full requested range
        requested_days = args.years * 365
        if actual_days < requested_days * 0.9:  # If we got less than 90% of requested days
            print(f"\nWARNING: Only fetched {actual_days} days of data out of {requested_days} requested")
        
        return 0
    else:
        print("Failed to fetch Bitcoin price data")
        return 1

if __name__ == "__main__":
    sys.exit(main())