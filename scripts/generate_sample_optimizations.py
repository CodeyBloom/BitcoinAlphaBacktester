#!/usr/bin/env python3
"""
Script to generate sample optimization results for the Bitcoin Strategy Backtester.
This script creates sample files for testing the UI.
"""

import os
import sys
import polars as pl
from datetime import datetime, timedelta

# Create a directory for storing optimization results if it doesn't exist
OPTIMIZATION_DIR = "data/optimizations"
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# Sample strategies
STRATEGIES = ["dca", "maco", "rsi", "volatility"]

# Predefined time periods (in years)
TIME_PERIODS = [1, 5, 10]

# Sample currencies
CURRENCIES = ["AUD", "USD"]

def format_date(date_obj):
    """Format date object as DDMMYYYY string"""
    return date_obj.strftime("%d%m%Y")

def create_dca_optimization(start_date_str, end_date_str, currency):
    """Create sample DCA optimization results"""
    data = {
        "strategy": "dca",
        "param_exchange_id": "binance",
        "param_weekly_investment": 100.0,
        "param_use_discount": True,
        "performance_final_btc": 0.45678912,
        "performance_max_drawdown": 0.21,
        "performance_sortino_ratio": 1.35,
    }
    df = pl.DataFrame([data])
    filename = f"dca_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")

def create_maco_optimization(start_date_str, end_date_str, currency):
    """Create sample MACO optimization results"""
    data = {
        "strategy": "maco",
        "param_exchange_id": "coinbase",
        "param_weekly_investment": 150.0,
        "param_use_discount": False,
        "param_short_window": 15,
        "param_long_window": 75,
        "performance_final_btc": 0.55678912,
        "performance_max_drawdown": 0.28,
        "performance_sortino_ratio": 1.12,
    }
    df = pl.DataFrame([data])
    filename = f"maco_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")

def create_rsi_optimization(start_date_str, end_date_str, currency):
    """Create sample RSI optimization results"""
    data = {
        "strategy": "rsi",
        "param_exchange_id": "kraken",
        "param_weekly_investment": 120.0,
        "param_use_discount": True,
        "param_rsi_period": 12,
        "param_oversold_threshold": 28,
        "param_overbought_threshold": 72,
        "performance_final_btc": 0.60123456,
        "performance_max_drawdown": 0.25,
        "performance_sortino_ratio": 1.48,
    }
    df = pl.DataFrame([data])
    filename = f"rsi_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")

def create_volatility_optimization(start_date_str, end_date_str, currency):
    """Create sample volatility optimization results"""
    data = {
        "strategy": "volatility",
        "param_exchange_id": "binance",
        "param_weekly_investment": 200.0,
        "param_use_discount": True,
        "param_vol_window": 18,
        "param_vol_threshold": 1.75,
        "performance_final_btc": 0.58123456,
        "performance_max_drawdown": 0.30,
        "performance_sortino_ratio": 1.22,
    }
    df = pl.DataFrame([data])
    filename = f"volatility_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")

def generate_sample_data_for_timeperiod(years, currency):
    """Generate sample data for a specific time period"""
    today = datetime.now()
    end_date = today
    start_date = end_date.replace(year=end_date.year - years)
    
    start_date_str = format_date(start_date)
    end_date_str = format_date(end_date)
    
    print(f"Generating data for {years} year(s) in {currency}...")
    
    create_dca_optimization(start_date_str, end_date_str, currency)
    create_maco_optimization(start_date_str, end_date_str, currency)
    create_rsi_optimization(start_date_str, end_date_str, currency)
    create_volatility_optimization(start_date_str, end_date_str, currency)

def main():
    """Generate sample optimization results for all time periods and currencies"""
    for currency in CURRENCIES:
        for years in TIME_PERIODS:
            generate_sample_data_for_timeperiod(years, currency)
    
    print(f"Generated {len(TIME_PERIODS) * len(CURRENCIES) * len(STRATEGIES)} sample optimization files")

if __name__ == "__main__":
    main()