#!/usr/bin/env python3
"""
Script to generate optimization files for each time period (1, 5, and 10 years).
This script creates the necessary files for testing the time period selection functionality.
"""

import os
import sys
import polars as pl
from datetime import datetime, timedelta

# Create a directory for storing optimization results if it doesn't exist
OPTIMIZATION_DIR = "data/optimizations"
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# Only use AUD as requested
CURRENCY = "AUD"

# Strategies
STRATEGIES = ["dca", "maco", "rsi", "volatility", "xgboost_ml"]

# Time periods
TIME_PERIODS = {
    "1 Year": 1,
    "5 Years": 5,
    "10 Years": 10
}

def format_date(date_obj):
    """Format date object as DDMMYYYY string"""
    return date_obj.strftime("%d%m%Y")

def create_dca_optimization(years, currency="AUD"):
    """Create sample DCA optimization results for a specific time period"""
    today = datetime.now()
    end_date = today
    start_date = end_date.replace(year=end_date.year - years)
    
    start_date_str = format_date(start_date)
    end_date_str = format_date(end_date)
    
    # Scale performance metrics based on time period
    btc_accumulated = 0.15678912 * years  # More BTC for longer periods
    efficiency = 0.000087  # Efficiency remains fairly constant
    
    # For DCA, optimize weekly investment amount, exchange, and day of week
    data = {
        "strategy": "dca",
        "param_exchange_id": "binance",
        "param_weekly_investment": 100.0,
        "param_use_discount": True,
        "param_day_of_week": "Sunday",
        "param_frequency": "Weekly",
        "performance_final_btc": btc_accumulated,
        "performance_max_drawdown": max(0.10, min(0.45, 0.21 - (0.02 * years))),  # Drawdown tends to increase with time
        "performance_sortino_ratio": 1.35,
        "performance_efficiency": efficiency,  # BTC per currency unit
        "performance_total_invested": 100.0 * 52 * years,  # Weekly investment * weeks
    }
    df = pl.DataFrame([data])
    filename = f"dca_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def create_maco_optimization(years, currency="AUD"):
    """Create sample MACO optimization results for a specific time period"""
    today = datetime.now()
    end_date = today
    start_date = end_date.replace(year=end_date.year - years)
    
    start_date_str = format_date(start_date)
    end_date_str = format_date(end_date)
    
    # Scale performance metrics based on time period
    btc_accumulated = 0.19678912 * years  # More BTC for longer periods
    efficiency = 0.000079  # Efficiency remains fairly constant
    
    # MACO parameters
    data = {
        "strategy": "maco",
        "param_exchange_id": "coinbase",
        "param_weekly_investment": 150.0,
        "param_use_discount": False,
        "param_short_window": 15,  # Short-term moving average window in days
        "param_long_window": 75,   # Long-term moving average window in days
        "param_signal_threshold": 0.01,  # Percentage difference required for crossover signal
        "param_max_allocation": 0.8,     # Maximum allocation on strong signals
        "performance_final_btc": btc_accumulated,
        "performance_max_drawdown": max(0.15, min(0.50, 0.28 - (0.01 * years))),
        "performance_sortino_ratio": 1.12 + (0.05 * years),  # Sortino ratio tends to improve with time
        "performance_efficiency": efficiency,  # BTC per currency unit
        "performance_total_invested": 150.0 * 52 * years,  # Weekly investment * weeks
    }
    df = pl.DataFrame([data])
    filename = f"maco_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def create_rsi_optimization(years, currency="AUD"):
    """Create sample RSI optimization results for a specific time period"""
    today = datetime.now()
    end_date = today
    start_date = end_date.replace(year=end_date.year - years)
    
    start_date_str = format_date(start_date)
    end_date_str = format_date(end_date)
    
    # Scale performance metrics based on time period
    btc_accumulated = 0.22123456 * years  # More BTC for longer periods
    efficiency = 0.000092  # Efficiency remains fairly constant
    
    # RSI parameters
    data = {
        "strategy": "rsi",
        "param_exchange_id": "kraken",
        "param_weekly_investment": 120.0,
        "param_use_discount": True,
        "param_rsi_period": 12,              # Period for RSI calculation
        "param_oversold_threshold": 28,       # RSI threshold for oversold condition
        "param_overbought_threshold": 72,     # RSI threshold for overbought condition
        "param_max_increase_factor": 2.5,     # Maximum investment increase during oversold
        "param_min_decrease_factor": 0.5,     # Minimum investment during overbought
        "performance_final_btc": btc_accumulated,
        "performance_max_drawdown": max(0.12, min(0.48, 0.25 - (0.015 * years))),
        "performance_sortino_ratio": 1.48 + (0.07 * years),  # Sortino ratio tends to improve with time
        "performance_efficiency": efficiency,   # BTC per currency unit
        "performance_total_invested": 120.0 * 52 * years,  # Weekly investment * weeks
    }
    df = pl.DataFrame([data])
    filename = f"rsi_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def create_volatility_optimization(years, currency="AUD"):
    """Create sample volatility optimization results for a specific time period"""
    today = datetime.now()
    end_date = today
    start_date = end_date.replace(year=end_date.year - years)
    
    start_date_str = format_date(start_date)
    end_date_str = format_date(end_date)
    
    # Scale performance metrics based on time period
    btc_accumulated = 0.20123456 * years  # More BTC for longer periods
    efficiency = 0.000084  # Efficiency remains fairly constant
    
    # Volatility parameters
    data = {
        "strategy": "volatility",
        "param_exchange_id": "binance",
        "param_weekly_investment": 200.0,
        "param_use_discount": True,
        "param_vol_window": 18,               # Window for volatility calculation
        "param_vol_threshold": 1.75,          # Threshold multiplier for increased investment
        "param_max_increase_factor": 3.0,     # Maximum investment increase during high volatility
        "param_lookback_period": 90,          # Period to calculate average volatility
        "performance_final_btc": btc_accumulated,
        "performance_max_drawdown": max(0.18, min(0.55, 0.30 - (0.01 * years))),
        "performance_sortino_ratio": 1.22 + (0.04 * years),  # Sortino ratio tends to improve with time
        "performance_efficiency": efficiency,   # BTC per currency unit
        "performance_total_invested": 200.0 * 52 * years,  # Weekly investment * weeks
    }
    df = pl.DataFrame([data])
    filename = f"volatility_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def create_xgboost_ml_optimization(years, currency="AUD"):
    """Create sample XGBoost ML optimization results for a specific time period"""
    today = datetime.now()
    end_date = today
    start_date = end_date.replace(year=end_date.year - years)
    
    start_date_str = format_date(start_date)
    end_date_str = format_date(end_date)
    
    # Scale performance metrics based on time period
    btc_accumulated = 0.24567890 * years  # More BTC for longer periods
    efficiency = 0.000097  # Efficiency remains fairly constant
    
    # XGBoost ML parameters
    data = {
        "strategy": "xgboost_ml",
        "param_exchange_id": "kraken",
        "param_weekly_investment": 180.0,
        "param_use_discount": True,
        "param_training_window": 14,              # Training window in days
        "param_prediction_threshold": 0.58,       # Confidence threshold for investment
        "param_max_allocation": 2.0,              # Maximum allocation on strong predictions
        "param_feature_set": "price,returns,volume,volatility",  # Features used for prediction
        "performance_final_btc": btc_accumulated,
        "performance_max_drawdown": max(0.14, min(0.47, 0.24 - (0.012 * years))),
        "performance_sortino_ratio": 1.55 + (0.08 * years),  # Sortino ratio tends to improve with time
        "performance_efficiency": efficiency,   # BTC per currency unit
        "performance_total_invested": 180.0 * 52 * years,  # Weekly investment * weeks
    }
    df = pl.DataFrame([data])
    filename = f"xgboost_ml_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def main():
    """Generate optimization files for each time period and strategy"""
    # Clean up existing USD files (as we're focusing only on AUD)
    for filename in os.listdir(OPTIMIZATION_DIR):
        if filename.endswith("_USD.arrow"):
            file_path = os.path.join(OPTIMIZATION_DIR, filename)
            os.remove(file_path)
            print(f"Removed USD file: {file_path}")
    
    # Generate new files for each time period
    for period_name, years in TIME_PERIODS.items():
        print(f"\nGenerating optimization files for {period_name} ({years} years)...")
        
        for strategy in STRATEGIES:
            if strategy == "dca":
                create_dca_optimization(years, CURRENCY)
            elif strategy == "maco":
                create_maco_optimization(years, CURRENCY)
            elif strategy == "rsi":
                create_rsi_optimization(years, CURRENCY)
            elif strategy == "volatility":
                create_volatility_optimization(years, CURRENCY)
            elif strategy == "xgboost_ml":
                create_xgboost_ml_optimization(years, CURRENCY)
    
    print("\nAll optimization files generated successfully!")

if __name__ == "__main__":
    main()