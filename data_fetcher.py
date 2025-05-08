import httpx
import polars as pl
from datetime import datetime, timedelta, date
import time

def fetch_bitcoin_price_data(start_date_str, end_date_str, currency="AUD"):
    """
    Fetch Bitcoin historical price data from CoinGecko API
    
    Args:
        start_date_str (str): Start date in DD-MM-YYYY format
        end_date_str (str): End date in DD-MM-YYYY format
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        polars.DataFrame: Dataframe with historical price data
    """
    # Convert dates to required format
    start_date = datetime.strptime(start_date_str, "%d-%m-%Y")
    end_date = datetime.strptime(end_date_str, "%d-%m-%Y")
    
    # Convert to Unix timestamps (in seconds)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    # CoinGecko API URL for Bitcoin market chart
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    
    params = {
        "vs_currency": currency.lower(),
        "from": start_timestamp,
        "to": end_timestamp
    }
    
    try:
        with httpx.Client() as client:
            response = client.get(url, params=params, timeout=30.0)
            
        if response.status_code == 200:
            data = response.json()
            
            # Extract price data
            prices = data.get("prices", [])
            
            if not prices:
                return None
            
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
            
            # Sort by date to ensure chronological order
            df_prices = df_prices.sort("date")
            
            # Convert to daily data (last price of each day)
            daily_df = df_prices.group_by_dynamic("date", every="1d").agg(
                pl.last("price").alias("price")
            )
            
            # Make sure we have a row for each day in the range
            if len(df_prices) > 0:
                # Get the min and max date objects from the dataframe
                # Convert to native Python datetime objects for easier handling
                first_date_series = df_prices.select(pl.min("date"))
                last_date_series = df_prices.select(pl.max("date"))
                
                first_date = first_date_series[0, 0]
                last_date = last_date_series[0, 0]
                
                # Only proceed if we have valid dates
                if first_date is not None and last_date is not None:
                    # Ensure they're Python datetime objects
                    if not isinstance(first_date, datetime):
                        first_date = datetime.fromisoformat(str(first_date))
                    
                    if not isinstance(last_date, datetime):
                        last_date = datetime.fromisoformat(str(last_date))
                    
                    # Create date range manually
                    days_diff = (last_date - first_date).days + 1
                    date_list = [first_date + timedelta(days=i) for i in range(days_diff)]
                    full_date_range = pl.Series(date_list)
                else:
                    # If we couldn't get valid dates, use the existing date column
                    full_date_range = df_prices["date"]
            else:
                # Fallback if no valid dates
                full_date_range = df_prices["date"]
            
            daily_df = pl.DataFrame({"date": full_date_range}).join(
                daily_df, on="date", how="left"
            )
            
            # Forward fill missing values
            daily_df = daily_df.with_columns(
                pl.col("price").fill_null(strategy="forward")
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
            
            # Add columns that will be used by strategies
            daily_df = daily_df.with_columns([
                pl.lit(0.0).alias("cumulative_investment"),
                pl.lit(0.0).alias("btc_bought"),
                pl.lit(0.0).alias("cumulative_btc")
            ])
            
            return daily_df
        
        elif response.status_code == 429:
            # Rate limit hit, wait and retry
            retry_after = int(response.headers.get("Retry-After", 60))
            time.sleep(retry_after)
            return fetch_bitcoin_price_data(start_date_str, end_date_str, currency)
        
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception while fetching data: {str(e)}")
        return None
