"""
Script to run optimizations and save results to file.

This script is meant to be run separately from the Streamlit app.
It will run optimizations for all strategies and save the results to Arrow files.
"""

import os
import argparse
from datetime import datetime
import polars as pl
import pandas as pd

from optimize_strategies import (
    optimize_dca_strategy,
    optimize_maco_strategy,
    optimize_rsi_strategy,
    optimize_volatility_strategy,
    optimize_all_strategies
)

# Create a directory for storing optimization results
OPTIMIZATION_DIR = "data/optimizations"
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

def run_and_save_optimization(strategy, start_date, end_date, currency="AUD", n_calls=50):
    """
    Run optimization for a single strategy and save results to an Arrow file.
    
    Args:
        strategy (str): Strategy name ('dca', 'maco', 'rsi', 'volatility', 'all')
        start_date (str): Start date in DD-MM-YYYY format
        end_date (str): End date in DD-MM-YYYY format
        currency (str): Currency to use
        n_calls (int): Number of optimization iterations
        
    Returns:
        str: Path to the saved results file
    """
    print(f"Running optimization for {strategy.upper()} strategy...")
    
    # Run the appropriate optimization function
    if strategy == "dca":
        result = optimize_dca_strategy(start_date, end_date, currency, n_calls)
    elif strategy == "maco":
        result = optimize_maco_strategy(start_date, end_date, currency, n_calls)
    elif strategy == "rsi":
        result = optimize_rsi_strategy(start_date, end_date, currency, n_calls)
    elif strategy == "volatility":
        result = optimize_volatility_strategy(start_date, end_date, currency, n_calls)
    elif strategy == "all":
        result = optimize_all_strategies(start_date, end_date, currency, n_calls)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # If result is None, don't save
    if result is None:
        print(f"Optimization for {strategy} failed. No results to save.")
        return None
    
    # Create filename based on strategy and dates
    filename = f"{strategy}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{currency}.ipc"
    filepath = os.path.join(OPTIMIZATION_DIR, filename)
    
    # Convert result to DataFrame for saving to Arrow format
    # We need to flatten the nested structure
    result_data = {
        "strategy": strategy,
        "start_date": start_date,
        "end_date": end_date,
        "currency": currency,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add best parameters
    for param, value in result["best_params"].items():
        result_data[f"param_{param}"] = value
    
    # Add performance metrics
    for metric, value in result["performance"].items():
        result_data[f"performance_{metric}"] = value
    
    # Convert to DataFrame
    # First create a pandas DataFrame with a single row
    pdf = pd.DataFrame([result_data])
    # Convert to Polars DataFrame
    df = pl.from_pandas(pdf)
    
    # Save to Arrow file
    df.write_ipc(filepath)
    
    print(f"Optimization results saved to {filepath}")
    return filepath

def run_all_optimizations(start_date, end_date, currency="AUD", n_calls=50):
    """
    Run optimizations for all strategies and save results to files.
    
    Args:
        start_date (str): Start date in DD-MM-YYYY format
        end_date (str): End date in DD-MM-YYYY format
        currency (str): Currency to use
        n_calls (int): Number of optimization iterations
        
    Returns:
        dict: Paths to the saved results files
    """
    strategies = ["dca", "maco", "rsi", "volatility"]
    result_files = {}
    
    for strategy in strategies:
        filepath = run_and_save_optimization(strategy, start_date, end_date, currency, n_calls)
        if filepath:
            result_files[strategy] = filepath
    
    return result_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimizations and save results to Arrow files")
    parser.add_argument("--strategy", choices=["dca", "maco", "rsi", "volatility", "all"], 
                        default="all", help="Strategy to optimize")
    parser.add_argument("--start-date", required=True, help="Start date (DD-MM-YYYY)")
    parser.add_argument("--end-date", required=True, help="End date (DD-MM-YYYY)")
    parser.add_argument("--currency", default="AUD", help="Currency to use")
    parser.add_argument("--n-calls", type=int, default=50, help="Number of optimization iterations")
    
    args = parser.parse_args()
    
    if args.strategy == "all":
        run_all_optimizations(args.start_date, args.end_date, args.currency, args.n_calls)
    else:
        run_and_save_optimization(args.strategy, args.start_date, args.end_date, args.currency, args.n_calls)