#!/usr/bin/env python3
"""
Script to run all optimizations for the Bitcoin Strategy Backtester.
This script is designed to be run by GitHub Actions on a weekly basis.
"""

import os
import sys
import polars as pl
from datetime import datetime, timedelta
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Directory for storing optimization results
OPTIMIZATION_DIR = "data/optimizations"
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# Currencies to optimize for
CURRENCIES = ["AUD", "USD"]

# Strategies to optimize
STRATEGIES = ["dca", "maco", "rsi", "volatility"]

# Predefined time periods (in years)
TIME_PERIODS = [1, 5, 10]

# Number of optimization iterations
N_CALLS = 50  # Increase for better results, but longer runtime

def format_date(date_obj):
    """Format date object as DD-MM-YYYY string"""
    return date_obj.strftime("%d%m%Y")  # Format used in optimization filenames

def optimize_strategy(strategy, start_date_str, end_date_str, currency, n_calls=N_CALLS):
    """
    Run optimization for a specific strategy and save results.
    
    Args:
        strategy (str): Strategy name (e.g., "dca", "maco")
        start_date_str (str): Start date in DDMMYYYY format
        end_date_str (str): End date in DDMMYYYY format
        currency (str): Currency code
        n_calls (int): Number of optimization iterations
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Import here to avoid circular imports
    try:
        if strategy == "dca":
            from optimize_strategies import optimize_dca_strategy as optimize_func
        elif strategy == "maco":
            from optimize_strategies import optimize_maco_strategy as optimize_func
        elif strategy == "rsi":
            from optimize_strategies import optimize_rsi_strategy as optimize_func
        elif strategy == "volatility":
            from optimize_strategies import optimize_volatility_strategy as optimize_func
        else:
            print(f"Unknown strategy: {strategy}")
            return False
        
        print(f"Optimizing {strategy.upper()} for {currency} from {start_date_str} to {end_date_str}...")
        
        # Convert date format for the optimization function
        formatted_start = f"{start_date_str[:2]}-{start_date_str[2:4]}-{start_date_str[4:]}"
        formatted_end = f"{end_date_str[:2]}-{end_date_str[2:4]}-{end_date_str[4:]}"
        
        # Run the optimization
        result = optimize_func(formatted_start, formatted_end, currency, n_calls)
        
        if result is None:
            print(f"  Failed to optimize {strategy}")
            return False
        
        # Create a DataFrame from the optimization result
        data = {
            "strategy": strategy,
        }
        
        # Add parameters with param_ prefix
        for param_name, param_value in result["best_params"].items():
            data[f"param_{param_name}"] = param_value
        
        # Add performance metrics with performance_ prefix
        for metric_name, metric_value in result["performance"].items():
            data[f"performance_{metric_name}"] = metric_value
        
        # Create DataFrame
        df = pl.DataFrame([data])
        
        # Save to Arrow file
        file_path = os.path.join(OPTIMIZATION_DIR, f"{strategy}_{start_date_str}_{end_date_str}_{currency}.arrow")
        df.write_ipc(file_path)
        print(f"  Saved optimization results to {file_path}")
        
        return True
    
    except Exception as e:
        print(f"  Error optimizing {strategy}: {str(e)}")
        return False

def run_all_optimizations():
    """Run all optimizations for all time periods, strategies, and currencies"""
    success = True
    
    # Calculate date ranges for each time period
    today = datetime.now()
    date_ranges = []
    
    for years in TIME_PERIODS:
        end_date = today
        start_date = end_date.replace(year=end_date.year - years)
        
        date_ranges.append((
            format_date(start_date),
            format_date(end_date),
            years
        ))
    
    # Run optimizations for each combination
    for start_date_str, end_date_str, years in date_ranges:
        for currency in CURRENCIES:
            for strategy in STRATEGIES:
                if not optimize_strategy(strategy, start_date_str, end_date_str, currency):
                    success = False
                
                # Be nice to the system - add a small delay between optimizations
                time.sleep(1)
    
    return success

def main():
    """Run all optimizations"""
    success = run_all_optimizations()
    
    if success:
        print("All optimizations completed successfully")
        return 0
    else:
        print("Some errors occurred during optimization")
        return 1

if __name__ == "__main__":
    sys.exit(main())