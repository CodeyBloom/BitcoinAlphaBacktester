#!/usr/bin/env python3
"""
Script to fetch Bitcoin price data using the CoinGecko API.
This script handles the API's limitation of 365 days of historical data for free API users.

For free API users:
- Fetches the maximum allowed 365 days of historical data
- Adds a note about the API limitations

For paid API users (requires API key):
- Attempts to fetch multiple years of data by making sequential requests
- Combines the results into a single dataset

Usage:
    python scripts/fetch_10yr_bitcoin_data.py [--currency AUD] [--api-key YOUR_API_KEY]

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

# Import project modules
from data_fetcher import parse_date_string
from fetch_bitcoin_data import parse_api_response, process_price_data

def fetch_bitcoin_chunk(start_date, end_date, currency="AUD", api_key=None):
    """
    Fetch Bitcoin historical price data from CoinGecko API for a specific date range.
    Handles rate limiting and retries.
    
    Args:
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        currency (str): Currency to fetch prices in (default: AUD)
        api_key (str, optional): CoinGecko API key for paid plans
        
    Returns:
        polars.DataFrame or None: DataFrame with historical price data or None on failure
    """
    # CoinGecko API URL for Bitcoin market chart
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Check for free API limitations
    if api_key is None:
        # Free API can only access 365 days of historical data
        one_year_ago = date.today() - timedelta(days=365)
        if start_date < one_year_ago:
            print("WARNING: Free API can only access 365 days of historical data")
            print(f"Adjusting start date from {start_date.strftime('%Y-%m-%d')} to {one_year_ago.strftime('%Y-%m-%d')}")
            start_date = one_year_ago
    
    # Convert dates to Unix timestamps (in seconds)
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
    
    params = {
        "vs_currency": currency.lower(),
        "from": start_timestamp,
        "to": end_timestamp
    }
    
    # Set up headers if API key is provided
    headers = {}
    if api_key:
        headers["x-cg-pro-api-key"] = api_key
    
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            response = httpx.get(url, params=params, headers=headers, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse API response
                price_data = parse_api_response(data)
                
                if not price_data:
                    print("No data returned from API")
                    retries += 1
                    time.sleep(5)
                    continue
                
                # Process data into DataFrame
                daily_df = process_price_data(price_data)
                
                # Resample to daily data if needed (last price of each day)
                # This ensures we have exactly one price per day
                if daily_df.select(pl.col("date").dt.date().unique()).height != daily_df.height:
                    daily_df = daily_df.group_by_dynamic("date", every="1d").agg(
                        pl.last("price").alias("price"),
                        pl.last("day_of_week").alias("day_of_week"),
                        pl.last("is_sunday").alias("is_sunday")
                    )
                    
                    # Recalculate returns after resampling
                    daily_df = daily_df.with_columns(
                        pl.col("price").pct_change().fill_null(0).alias("returns")
                    )
                
                print(f"Successfully fetched {len(daily_df)} days of data")
                return daily_df
                
            elif response.status_code == 429:
                # Rate limit hit, wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limit hit, waiting for {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            elif response.status_code == 401 and "Your request exceeds the allowed time range" in response.text:
                # Free API limitation
                print("Free API limitation: Cannot fetch data older than 365 days")
                print("To access historical data, a CoinGecko paid plan is required")
                if api_key:
                    print("The provided API key may be invalid or insufficient for historical data")
                return None
                
            else:
                print(f"Error fetching data: {response.status_code} - {response.text}")
                retries += 1
                time.sleep(5)
                
        except Exception as e:
            print(f"Exception while fetching data: {str(e)}")
            retries += 1
            time.sleep(5)
    
    print(f"Failed to fetch data chunk after {max_retries} retries")
    return None

def fetch_bitcoin_data_for_years(years=10, currency="AUD", api_key=None):
    """
    Fetch Bitcoin historical price data from CoinGecko API for a specified number of years.
    Handles the 1-year limit by making multiple API calls.
    
    Args:
        years (int): Number of years of data to fetch
        currency (str): Currency to fetch prices in (default: AUD)
        api_key (str, optional): CoinGecko API key for paid plans
        
    Returns:
        polars.DataFrame or None: DataFrame with historical price data or None on failure
    """
    today = date.today()
    all_dataframes = []
    
    # Calculate the start date based on the requested number of years
    initial_start_date = today.replace(year=today.year - years)
    
    # If using free API and requesting more than 1 year, print a warning
    if api_key is None and years > 1:
        print("WARNING: Free API can only access 365 days of historical data")
        print(f"You requested {years} years, but only the last year will be available")

    # Split into chunks to respect API's 1-year limitation
    current_end_date = today
    
    # Start from today and go backwards in time, chunk by chunk
    while current_end_date >= initial_start_date:
        # Calculate start date for this chunk (max 364 days earlier)
        days_to_subtract = min(364, (current_end_date - initial_start_date).days)
        current_start_date = current_end_date - timedelta(days=days_to_subtract)
        
        # Ensure we don't go back further than the initial start date
        if current_start_date < initial_start_date:
            current_start_date = initial_start_date
        
        # Fetch this chunk of data
        chunk_df = fetch_bitcoin_chunk(current_start_date, current_end_date, currency, api_key)
        
        if chunk_df is not None and len(chunk_df) > 0:
            all_dataframes.append(chunk_df)
        else:
            # If we can't fetch historical data (likely due to free API limits), stop trying
            if api_key is None and current_end_date.year < today.year:
                print("Unable to fetch historical data beyond 1 year with free API")
                break
        
        # Move end date to one day before start date for next chunk
        current_end_date = current_start_date - timedelta(days=1)
        
        # If we've reached or passed our desired start date, we're done
        if current_end_date < initial_start_date:
            break
            
        # Be nice to the API - add a delay between chunks
        print("Waiting 2 seconds before next API call...")
        time.sleep(2)
    
    # If no data was fetched, return None
    if not all_dataframes:
        print("Failed to fetch any data")
        return None
    
    # Combine all dataframes
    print(f"Combining {len(all_dataframes)} data chunks...")
    
    # Remove row_index columns before concatenating
    for i in range(len(all_dataframes)):
        if "row_index" in all_dataframes[i].columns:
            all_dataframes[i] = all_dataframes[i].drop("row_index")
    
    try:
        # Combine the dataframes, handling potential schema differences
        combined_df = pl.concat(all_dataframes, how="vertical_relaxed")
        
        # Sort by date and deduplicate
        combined_df = (
            combined_df
            .sort("date")
            .unique(subset=["date"])  # Remove any duplicate dates
        )
        
        # Recalculate returns for the full dataset
        combined_df = combined_df.with_columns(
            pl.col("price").pct_change().fill_null(0).alias("returns")
        )
        
        # Add row_index for strategy calculations
        combined_df = combined_df.with_row_index("row_index")
        
        print(f"Successfully compiled {len(combined_df)} days of data "
              f"from {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        return combined_df
    
    except Exception as e:
        print(f"Error combining data: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fetch historical Bitcoin price data from CoinGecko API")
    parser.add_argument("--currency", type=str, default="AUD", help="Currency (default: AUD)")
    parser.add_argument("--years", type=int, default=10, help="Number of years of data to fetch (default: 10)")
    parser.add_argument("--output", type=str, default="data/bitcoin_prices.arrow", help="Output file path")
    parser.add_argument("--api-key", type=str, help="CoinGecko API key for paid plans (allows access to data older than 1 year)")
    
    args = parser.parse_args()
    
    # Check for API key in environment variable if not provided via command line
    api_key = args.api_key or os.environ.get("COINGECKO_API_KEY")
    
    if not api_key and args.years > 1:
        print("NOTE: No API key provided. Free API limited to 365 days of historical data.")
        print("To fetch more historical data, upgrade to a paid CoinGecko plan and provide an API key.")
        print("You can still run the script, but only the most recent 365 days will be retrieved.")
    
    print(f"Fetching {args.years} years of Bitcoin price data in {args.currency}...")
    
    # Fetch the requested years of data
    df = fetch_bitcoin_data_for_years(years=args.years, currency=args.currency, api_key=api_key)
    
    if df is not None and len(df) > 0:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save to Arrow file
        df.write_ipc(args.output)
        
        print(f"Successfully saved {len(df)} days of Bitcoin price data to {args.output}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Add a note about the data range compared to what was requested
        days_fetched = len(df)
        days_requested = 365 * args.years
        if days_fetched < days_requested * 0.9:  # If we got less than 90% of requested days
            print(f"WARNING: Only fetched {days_fetched} days of data out of {days_requested} requested.")
            print("This is likely due to free API limitations. Consider upgrading to a paid plan.")
        
        return 0
    else:
        print("Failed to fetch Bitcoin price data")
        return 1

if __name__ == "__main__":
    sys.exit(main())