"""
Script to generate sample optimization results for the Bitcoin Strategy Backtester.

This script creates sample Arrow files with optimization results
to demonstrate the loading functionality of the optimizer page.
"""

import os
import polars as pl
from datetime import datetime, timedelta

# Create a directory for storing optimization results if it doesn't exist
OPTIMIZATION_DIR = "data/optimizations"
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# Sample strategies
strategies = ["dca", "maco", "rsi", "volatility"]

# Sample dates
today = datetime.now()
one_year_ago = today - timedelta(days=365)
start_date_str = one_year_ago.strftime("%d%m%Y")
end_date_str = today.strftime("%d%m%Y")

# Sample currencies
currencies = ["AUD", "USD"]

def create_dca_optimization(currency):
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

def create_maco_optimization(currency):
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

def create_rsi_optimization(currency):
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

def create_volatility_optimization(currency):
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

def main():
    """Generate sample optimization results for all strategies and currencies"""
    for currency in currencies:
        create_dca_optimization(currency)
        create_maco_optimization(currency)
        create_rsi_optimization(currency)
        create_volatility_optimization(currency)
    
    print(f"Generated {len(strategies) * len(currencies)} sample optimization files")

if __name__ == "__main__":
    main()