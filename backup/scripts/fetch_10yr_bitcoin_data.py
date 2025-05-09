#!/usr/bin/env python3
"""
Script to fetch 10 years of Bitcoin price data using the CoinGecko API.
This script works around the API's limitation of 365 days per request by making
multiple sequential 365-day requests, each for a different time period,
and combining them into a complete dataset.

Usage:
    python scripts/fetch_10yr_bitcoin_data.py [--currency AUD] [--years 10]

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

def fetch_bitcoin_chunk(start_date, end_date, currency="AUD"):
    """
    Fetch Bitcoin historical price data from CoinGecko API for a specific date range.
    Handles rate limiting and retries. For free API users, handles the 365-day limit.
    
    Args:
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame or None: DataFrame with historical price data or None on failure
    """
    # CoinGecko API URL for Bitcoin market chart
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    
    # Check if we're requesting data beyond the 365-day limit
    one_year_ago = date.today() - timedelta(days=365)
    if start_date < one_year_ago:
        print(f"WARNING: Free API limited to last 365 days of data (since {one_year_ago.strftime('%Y-%m-%d')})")
        print(f"Requested start date {start_date.strftime('%Y-%m-%d')} is outside this range")
        print("Skipping this chunk")
        return None
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Convert dates to Unix timestamps (in seconds)
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
    
    params = {
        "vs_currency": currency.lower(),
        "from": start_timestamp,
        "to": end_timestamp
    }
    
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            response = httpx.get(url, params=params, timeout=30.0)
            
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
                
            elif response.status_code == 401 and "exceeds the allowed time range" in response.text:
                # This is the free API limitation
                print("Data requested exceeds the free API's 365-day historical limit")
                print("Skipping this chunk")
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

def generate_date_chunks(years=10):
    """
    Generate date chunks to fetch 10 years of Bitcoin data, working backwards from today.
    Each chunk is 365 days or less in length to comply with API limitations.
    
    Args:
        years (int): Number of years of data to fetch
        
    Returns:
        list: List of (start_date, end_date) tuples
    """
    today = date.today()
    chunks = []
    
    # Calculate the start date based on the requested number of years
    target_start_date = today.replace(year=today.year - years)
    
    # Work backwards from today in chunks of at most 365 days
    chunk_end_date = today
    
    while chunk_end_date > target_start_date:
        # Calculate start date for this chunk (max 365 days earlier)
        chunk_start_date = chunk_end_date - timedelta(days=364)
        
        # Don't go earlier than our target start date
        if chunk_start_date < target_start_date:
            chunk_start_date = target_start_date
            
        # Add this chunk to our list
        chunks.append((chunk_start_date, chunk_end_date))
        
        # Move to the next chunk
        chunk_end_date = chunk_start_date - timedelta(days=1)
    
    return chunks

def fetch_bitcoin_data_for_years(years=10, currency="AUD"):
    """
    Fetch Bitcoin historical price data from CoinGecko API for a specified number of years.
    Handles the 365-day limit by making multiple sequential API calls for different time periods.
    
    Args:
        years (int): Number of years of data to fetch
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame or None: DataFrame with historical price data or None on failure
    """
    all_dataframes = []
    
    # Generate date chunks to fetch
    date_chunks = generate_date_chunks(years)
    print(f"Generated {len(date_chunks)} date chunks to fetch {years} years of data")
    
    # Fetch each chunk
    for i, (start_date, end_date) in enumerate(date_chunks):
        print(f"Fetching chunk {i+1}/{len(date_chunks)}")
        chunk_df = fetch_bitcoin_chunk(start_date, end_date, currency)
        
        if chunk_df is not None and len(chunk_df) > 0:
            all_dataframes.append(chunk_df)
        else:
            print(f"Failed to fetch chunk {i+1}/{len(date_chunks)}")
        
        # Be nice to the API - add a delay between chunks
        if i < len(date_chunks) - 1:
            delay = 2  # seconds
            print(f"Waiting {delay} seconds before next API call...")
            time.sleep(delay)
    
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
    
    args = parser.parse_args()
    
    print(f"Attempting to fetch {args.years} years of Bitcoin price data in {args.currency}...")
    print("NOTE: Free CoinGecko API only allows access to the last 365 days of data")
    print("We'll fetch as much data as possible within this limitation")
    
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
        
        # Warn about limited data if less than requested
        requested_days = args.years * 365
        if actual_days < requested_days * 0.9:  # If we got less than 90% of requested days
            print(f"\nWARNING: Only fetched {actual_days} days of data out of {requested_days} requested")
            print("This is due to the CoinGecko free API's 365-day historical limit")
            print("The application will work with the available data")
        
        return 0
    else:
        print("Failed to fetch Bitcoin price data")
        return 1

if __name__ == "__main__":
    sys.exit(main())