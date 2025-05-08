import httpx
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_bitcoin_price_data(start_date_str, end_date_str, currency="AUD"):
    """
    Fetch Bitcoin historical price data from CoinGecko API
    
    Args:
        start_date_str (str): Start date in DD-MM-YYYY format
        end_date_str (str): End date in DD-MM-YYYY format
        currency (str): Currency to fetch prices in (default: AUD)
        
    Returns:
        pandas.DataFrame: Dataframe with historical price data
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
            df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
            
            # Convert timestamp (milliseconds) to datetime
            df_prices["date"] = pd.to_datetime(df_prices["timestamp"], unit="ms")
            
            # Set date as index and drop timestamp column
            df_prices = df_prices.drop(columns=["timestamp"])
            df_prices = df_prices.set_index("date")
            
            # Resample to daily data (close price)
            daily_df = df_prices.resample("D").last()
            
            # Reset index for easier manipulation
            daily_df = daily_df.reset_index()
            
            # Add day of week
            daily_df["day_of_week"] = daily_df["date"].dt.dayofweek
            
            # Add is_sunday flag
            daily_df["is_sunday"] = daily_df["day_of_week"] == 6
            
            # Add volatility columns (will be filled by strategy functions)
            daily_df["returns"] = daily_df["price"].pct_change()
            
            # Add columns that will be used by strategies
            daily_df["cumulative_investment"] = 0.0
            daily_df["btc_bought"] = 0.0
            daily_df["cumulative_btc"] = 0.0
            
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
