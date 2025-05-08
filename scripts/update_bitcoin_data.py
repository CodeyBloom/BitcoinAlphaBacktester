#!/usr/bin/env python3
"""
Script to update the Bitcoin price data using the CoinGecko API.
This script is designed to be run by GitHub Actions on a weekly basis.
"""

import os
import sys
import polars as pl
from datetime import datetime, timedelta
import httpx
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project modules
from data_fetcher import fetch_bitcoin_price_data, parse_date_string
from fetch_bitcoin_data import fetch_last_year_bitcoin_data

# Directory for storing price data
DATA_DIR = "data/bitcoin_prices"
os.makedirs(DATA_DIR, exist_ok=True)

# Currencies to fetch
CURRENCIES = ["AUD", "USD"]

def format_date(date_obj):
    """Format date object as DD-MM-YYYY string"""
    return date_obj.strftime("%d-%m-%Y")

def update_bitcoin_data_for_currency(currency):
    """
    Update Bitcoin price data for a specific currency.
    
    Args:
        currency (str): Currency code (e.g., "AUD", "USD")
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Updating Bitcoin price data for {currency}...")
    
    try:
        # Fetch the last year's worth of data
        df = fetch_last_year_bitcoin_data(currency)
        
        if df is None or df.height == 0:
            print(f"  Failed to fetch data for {currency}")
            return False
        
        # Save the data to an Arrow file
        file_path = os.path.join(DATA_DIR, f"bitcoin_prices_{currency}.arrow")
        df.write_ipc(file_path)
        print(f"  Saved data to {file_path}")
        
        return True
    
    except Exception as e:
        print(f"  Error updating data for {currency}: {str(e)}")
        return False

def update_historical_data_for_timeperiods(currency):
    """
    Ensure we have historical data for the predefined time periods.
    
    Args:
        currency (str): Currency code (e.g., "AUD", "USD")
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Updating historical data for predefined time periods ({currency})...")
    
    # Predefined time periods (in years)
    time_periods = [1, 5, 10]
    
    today = datetime.now()
    
    for years in time_periods:
        end_date = today
        start_date = end_date.replace(year=end_date.year - years)
        
        start_date_str = format_date(start_date)
        end_date_str = format_date(end_date)
        
        print(f"  Fetching data for {years} year(s): {start_date_str} to {end_date_str}")
        
        try:
            # Use existing function to fetch data
            df = fetch_bitcoin_price_data(start_date_str, end_date_str, currency)
            
            if df is None or df.height == 0:
                print(f"    Failed to fetch data for {years} year(s) in {currency}")
                continue
            
            # Save the data to a specific file for this time period
            file_path = os.path.join(DATA_DIR, f"bitcoin_{years}year_{currency}.arrow")
            df.write_ipc(file_path)
            print(f"    Saved {df.height} days of data to {file_path}")
        
        except Exception as e:
            print(f"    Error fetching data for {years} year(s) in {currency}: {str(e)}")
    
    return True

def main():
    """Update Bitcoin price data for all currencies and time periods"""
    success = True
    
    for currency in CURRENCIES:
        # Update the latest data first
        if not update_bitcoin_data_for_currency(currency):
            success = False
        
        # Then update the historical data for predefined time periods
        if not update_historical_data_for_timeperiods(currency):
            success = False
        
        # Be nice to the API - add a delay between currencies
        time.sleep(5)
    
    if success:
        print("All Bitcoin price data updated successfully")
        return 0
    else:
        print("Some errors occurred while updating Bitcoin price data")
        return 1

if __name__ == "__main__":
    sys.exit(main())