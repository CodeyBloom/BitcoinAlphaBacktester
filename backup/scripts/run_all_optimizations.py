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

# Currencies to optimize for - focus on AUD only as requested
CURRENCIES = ["AUD"]

# Strategies to optimize
STRATEGIES = ["dca", "maco", "rsi", "volatility"]

# Predefined time periods (in years)
TIME_PERIODS = [1, 5, 10]

# Number of optimization iterations
N_CALLS = 50  # Increase for better results, but longer runtime

def format_date(date_obj):
    """Format date object as DD-MM-YYYY string"""
    return date_obj.strftime("%d%m%Y")  # Format used in optimization filenames

def optimize_strategy(strategy, start_date_str, end_date_str, currency, run_number=1, n_calls=N_CALLS):
    """
    Run optimization for a specific strategy and save results.
    
    Args:
        strategy (str): Strategy name (e.g., "dca", "maco")
        start_date_str (str): Start date in DDMMYYYY format
        end_date_str (str): End date in DDMMYYYY format
        currency (str): Currency code
        run_number (int): Optimization run number (1, 2, or 3)
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
        
        print(f"Optimizing {strategy.upper()} (Run {run_number}/3) for {currency} from {start_date_str} to {end_date_str}...")
        
        # Convert date format for the optimization function
        formatted_start = f"{start_date_str[:2]}-{start_date_str[2:4]}-{start_date_str[4:]}"
        formatted_end = f"{end_date_str[:2]}-{end_date_str[2:4]}-{end_date_str[4:]}"
        
        # Run the optimization
        result = optimize_func(formatted_start, formatted_end, currency, n_calls)
        
        if result is None:
            print(f"  Failed to optimize {strategy} (Run {run_number})")
            return False
        
        # Create a DataFrame from the optimization result
        data = {
            "strategy": strategy,
            "run_number": run_number
        }
        
        # Add parameters with param_ prefix
        for param_name, param_value in result["best_params"].items():
            data[f"param_{param_name}"] = param_value
        
        # Add performance metrics with performance_ prefix
        for metric_name, metric_value in result["performance"].items():
            data[f"performance_{metric_name}"] = metric_value
        
        # Create DataFrame
        df = pl.DataFrame([data])
        
        # Save to Arrow file - include run number in the filename
        # Make sure run number is part of the filename to distinguish different runs
        file_path = os.path.join(OPTIMIZATION_DIR, f"{strategy}_{start_date_str}_{end_date_str}_{currency}_run{run_number}.arrow")
        df.write_ipc(file_path)
        print(f"  Saved optimization results to {file_path}")
        
        return True
    
    except Exception as e:
        print(f"  Error optimizing {strategy} (Run {run_number}): {str(e)}")
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
                # Run each optimization 3 times as requested
                for run_number in range(1, 4):  # 1, 2, 3
                    if not optimize_strategy(strategy, start_date_str, end_date_str, currency, run_number):
                        success = False
                    
                    # Be nice to the system - add a small delay between optimizations
                    time.sleep(1)
    
    # Also create a consolidated file for each strategy, time period, and currency
    # by averaging the results of the 3 runs
    print("Creating consolidated optimization files...")
    consolidate_optimization_results()
    
    return success

def consolidate_optimization_results():
    """
    Consolidate the 3 optimization runs for each strategy, time period, and currency
    by averaging the results and saving them to a file without '_run{number}' in the name.
    """
    try:
        # Get all optimization files
        optimization_files = os.listdir(OPTIMIZATION_DIR)
        
        # Group files by strategy, time period, and currency
        file_groups = {}
        for filename in optimization_files:
            if not filename.endswith(".arrow"):
                continue
                
            # Parse the filename to get strategy, start_date, end_date, currency, and run_number
            if "_run" in filename:
                # Example: dca_09052024_09052025_AUD_run1.arrow
                base_name = filename.split("_run")[0]  # dca_09052024_09052025_AUD
                run_number = int(filename.split("_run")[1].split(".")[0])  # 1
                
                if base_name not in file_groups:
                    file_groups[base_name] = []
                
                file_groups[base_name].append((filename, run_number))
        
        # Process each group
        for base_name, files in file_groups.items():
            # Skip if we don't have 3 runs
            if len(files) != 3:
                print(f"Warning: {base_name} has {len(files)} runs, expected 3. Skipping consolidation.")
                continue
            
            # Sort by run number
            files.sort(key=lambda x: x[1])
            
            # Load the data from each file
            dfs = []
            for filename, _ in files:
                file_path = os.path.join(OPTIMIZATION_DIR, filename)
                df = pl.read_ipc(file_path)
                dfs.append(df)
            
            # Find the best run based on efficiency
            best_df_index = 0
            best_efficiency = 0
            for i, df in enumerate(dfs):
                # Extract efficiency from performance metrics
                for col in df.columns:
                    if col == "performance_efficiency":
                        efficiency = df[col][0]
                        if efficiency > best_efficiency:
                            best_efficiency = efficiency
                            best_df_index = i
            
            # Use the best run as the consolidated result
            best_df = dfs[best_df_index]
            
            # Remove run_number column if it exists
            if "run_number" in best_df.columns:
                best_df = best_df.drop("run_number")
            
            # Save the consolidated result
            consolidated_file_path = os.path.join(OPTIMIZATION_DIR, f"{base_name}.arrow")
            best_df.write_ipc(consolidated_file_path)
            print(f"Created consolidated file: {consolidated_file_path}")
    
    except Exception as e:
        print(f"Error consolidating optimization results: {str(e)}")
        return False
    
    return True

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