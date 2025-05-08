import httpx
import polars as pl
from datetime import datetime, timedelta, date
import time
import os
import sys

def fetch_bitcoin_historical_data(start_year=2013, end_date=None, currency="AUD"):
    """
    Fetch Bitcoin historical price data from CoinGecko API for multiple years
    by making successive calls within the API's time constraints.
    
    Args:
        start_year (int): The first year to fetch data from
        end_date (date): The end date (defaults to today)
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame: Dataframe with historical price data
    """
    if end_date is None:
        end_date = date.today()
    
    # CoinGecko API URL for Bitcoin market chart
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    
    # Start with an empty dataframe
    all_data = pl.DataFrame()
    
    # Fetch data in 1-year chunks to stay within API limits
    current_start_date = date(start_year, 1, 1)
    
    while current_start_date < end_date:
        # Calculate end date for this chunk (either 1 year from start or the final end date)
        chunk_end_date = min(
            date(current_start_date.year + 1, current_start_date.month, current_start_date.day) - timedelta(days=1),
            end_date
        )
        
        print(f"Fetching data from {current_start_date} to {chunk_end_date}...")
        
        # Convert dates to Unix timestamps (in seconds)
        start_timestamp = int(datetime.combine(current_start_date, datetime.min.time()).timestamp())
        end_timestamp = int(datetime.combine(chunk_end_date, datetime.max.time()).timestamp())
        
        params = {
            "vs_currency": currency.lower(),
            "from": start_timestamp,
            "to": end_timestamp
        }
        
        max_retries = 3
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                with httpx.Client() as client:
                    response = client.get(url, params=params, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract price data
                    prices = data.get("prices", [])
                    
                    if not prices:
                        print(f"No data returned for period {current_start_date} to {chunk_end_date}")
                    else:
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
                        
                        # Append to all data
                        if len(all_data) == 0:
                            all_data = df_prices
                        else:
                            all_data = pl.concat([all_data, df_prices])
                    
                    success = True
                    
                elif response.status_code == 429:
                    # Rate limit hit, wait and retry
                    retry_after = int(response.headers.get("Retry-After", 60))
                    print(f"Rate limit hit, waiting for {retry_after} seconds...")
                    time.sleep(retry_after)
                    
                    # Don't count this as a retry, just a rate limit wait
                    continue
                    
                else:
                    print(f"Error fetching data: {response.status_code} - {response.text}")
                    retries += 1
                    time.sleep(5)  # Wait before retry
                    
            except Exception as e:
                print(f"Exception while fetching data: {str(e)}")
                retries += 1
                time.sleep(5)  # Wait before retry
        
        if not success:
            print(f"Failed to fetch data for period {current_start_date} to {chunk_end_date} after {max_retries} retries")
        
        # Move to next chunk
        current_start_date = chunk_end_date + timedelta(days=1)
        
        # Be nice to the API - wait between chunks
        time.sleep(2)
    
    # Process the combined data
    if len(all_data) > 0:
        # Remove any duplicates
        all_data = all_data.unique(subset=["date"])
        
        # Sort by date
        all_data = all_data.sort("date")
        
        # Resample to daily data (last price of each day)
        daily_df = all_data.group_by_dynamic("date", every="1d").agg(
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
        
        return daily_df
    else:
        print("No data was fetched")
        return None

def main():
    # Parse arguments
    currency = "AUD"
    start_year = 2013  # Default to a reasonable start year for Bitcoin
    output_file = "bitcoin_prices.arrow"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--currency" and i+2 <= len(sys.argv):
                currency = sys.argv[i+2]
            elif arg == "--start-year" and i+2 <= len(sys.argv):
                start_year = int(sys.argv[i+2])
            elif arg == "--output" and i+2 <= len(sys.argv):
                output_file = sys.argv[i+2]
    
    print(f"Fetching Bitcoin price data in {currency} from {start_year} to present...")
    
    # Fetch data
    df = fetch_bitcoin_historical_data(start_year=start_year, currency=currency)
    
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
    main()