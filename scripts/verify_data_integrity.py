#!/usr/bin/env python3
"""
Script to verify the integrity of Bitcoin data and optimization results.
This script checks:
1. Bitcoin price data completeness
2. Optimization files existence and validity
3. Data freshness (how recent the data is)

This can be used as a pre-run check for the application or as a standalone verification.
"""

import os
import sys
import polars as pl
from datetime import datetime, timedelta
import json
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Paths
BITCOIN_DATA_PATH = "data/bitcoin_prices.arrow"
OPTIMIZATION_DIR = "data/optimizations"

# Define expected strategies and time periods
STRATEGIES = ["dca", "maco", "rsi", "volatility", "xgboost_ml"]
TIME_PERIODS = {
    "1 Year": 1,
    "5 Years": 5,
    "10 Years": 10
}

def format_date_for_filename(date_obj):
    """Format date object as DDMMYYYY string for filenames"""
    return date_obj.strftime("%d%m%Y")

def check_bitcoin_data_integrity(currency="AUD", min_required_days=30, verbose=True):
    """
    Check the integrity of the Bitcoin price data.
    
    Args:
        currency (str): Currency code
        min_required_days (int): Minimum number of days required for valid data
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (is_valid, issues, stats)
            - is_valid: Boolean indicating whether the data is valid
            - issues: List of issues found
            - stats: Dictionary of data statistics
    """
    issues = []
    stats = {}
    
    # Check if Bitcoin data file exists
    if not os.path.exists(BITCOIN_DATA_PATH):
        issues.append(f"Bitcoin data file not found at {BITCOIN_DATA_PATH}")
        return False, issues, stats
    
    try:
        # Read Bitcoin data
        df = pl.read_ipc(BITCOIN_DATA_PATH)
        
        # Check if data is empty
        if len(df) == 0:
            issues.append("Bitcoin data file is empty")
            return False, issues, stats
        
        # Check if required columns exist
        required_columns = ["date", "price", "day_of_week", "is_sunday", "returns"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns in Bitcoin data: {missing_columns}")
            return False, issues, stats
        
        # Get date range - convert from polars date/timestamp to Python datetime if needed
        min_date = df["date"].min()
        max_date = df["date"].max()
        
        # Ensure we have valid datetime objects for calculations
        if isinstance(min_date, pl.Expr) or isinstance(min_date, pl.Series):
            min_date = min_date.to_python()
        if isinstance(max_date, pl.Expr) or isinstance(max_date, pl.Series):
            max_date = max_date.to_python()
        
        # Calculate days range
        days_range = (max_date - min_date).days if hasattr(max_date, 'days') else (max_date - min_date).total_seconds() / (60 * 60 * 24)
        days_range = int(days_range)
        
        # Check if there's enough data
        if days_range < min_required_days:
            issues.append(f"Insufficient data: only {days_range} days (minimum {min_required_days} required)")
        
        # Check data freshness
        today = datetime.now()
        days_since_last_data = (today - max_date).days if hasattr((today - max_date), 'days') else (today - max_date).total_seconds() / (60 * 60 * 24)
        days_since_last_data = int(days_since_last_data)
        
        if days_since_last_data > 7:
            issues.append(f"Data is stale: last entry is {days_since_last_data} days old")
        
        # Format dates for display with datetime-agnostic approach
        min_date_str = min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date)[:10]
        max_date_str = max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)[:10]
        
        # Collect statistics
        stats = {
            "num_days": len(df),
            "date_range": f"{min_date_str} to {max_date_str}",
            "days_since_last_update": days_since_last_data,
            "years_covered": days_range / 365.25,
            "min_price": float(df["price"].min()),
            "max_price": float(df["price"].max()),
            "avg_price": float(df["price"].mean())
        }
        
        if verbose:
            print(f"Bitcoin data stats:")
            print(f"  - Entries: {stats['num_days']} days")
            print(f"  - Date range: {stats['date_range']}")
            print(f"  - Years covered: {stats['years_covered']:.2f}")
            print(f"  - Last update: {days_since_last_data} days ago")
        
        return len(issues) == 0, issues, stats
        
    except Exception as e:
        issues.append(f"Error reading Bitcoin data: {str(e)}")
        return False, issues, stats

def check_optimization_files_integrity(currency="AUD", verbose=True):
    """
    Check the integrity of optimization files.
    
    Args:
        currency (str): Currency code
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (is_valid, issues, stats)
            - is_valid: Boolean indicating whether all optimization files are valid
            - issues: List of issues found
            - stats: Dictionary of optimization file statistics
    """
    issues = []
    stats = {"missing_files": [], "valid_files": [], "invalid_files": []}
    
    # Check if optimization directory exists
    if not os.path.exists(OPTIMIZATION_DIR):
        issues.append(f"Optimization directory not found at {OPTIMIZATION_DIR}")
        return False, issues, stats
    
    # Generate expected filenames
    expected_files = []
    today = datetime.now()
    
    for period_name, years in TIME_PERIODS.items():
        end_date = today
        start_date = end_date.replace(year=end_date.year - years)
        
        start_date_str = format_date_for_filename(start_date)
        end_date_str = format_date_for_filename(end_date)
        
        for strategy in STRATEGIES:
            expected_file = f"{strategy}_{start_date_str}_{end_date_str}_{currency}.arrow"
            expected_path = os.path.join(OPTIMIZATION_DIR, expected_file)
            expected_files.append((expected_file, expected_path))
    
    # Check each expected file
    for file_name, file_path in expected_files:
        if not os.path.exists(file_path):
            issues.append(f"Missing optimization file: {file_name}")
            stats["missing_files"].append(file_name)
            continue
        
        try:
            # Try reading the file to verify it's valid
            df = pl.read_ipc(file_path)
            
            # Check if the file has expected structure
            if len(df) == 0:
                issues.append(f"Empty optimization file: {file_name}")
                stats["invalid_files"].append(file_name)
                continue
                
            # Check for param_ and performance_ columns
            has_param_columns = any(col.startswith("param_") for col in df.columns)
            has_performance_columns = any(col.startswith("performance_") for col in df.columns)
            
            if not (has_param_columns and has_performance_columns):
                issues.append(f"Invalid structure in optimization file: {file_name}")
                stats["invalid_files"].append(file_name)
                continue
                
            # File is valid
            stats["valid_files"].append(file_name)
            
        except Exception as e:
            issues.append(f"Error reading optimization file {file_name}: {str(e)}")
            stats["invalid_files"].append(file_name)
    
    if verbose:
        print(f"Optimization files summary:")
        print(f"  - Valid: {len(stats['valid_files'])}")
        print(f"  - Missing: {len(stats['missing_files'])}")
        print(f"  - Invalid: {len(stats['invalid_files'])}")
    
    return len(issues) == 0, issues, stats

def run_data_verification(currency="AUD", output_json=None, verbose=True):
    """
    Run a complete data verification.
    
    Args:
        currency (str): Currency code
        output_json (str): Path to save verification results as JSON
        verbose (bool): Whether to print detailed information
        
    Returns:
        bool: Whether all data is valid
    """
    if verbose:
        print(f"Verifying Bitcoin data and optimization files...")
    
    # Check Bitcoin data
    bitcoin_valid, bitcoin_issues, bitcoin_stats = check_bitcoin_data_integrity(
        currency=currency, verbose=verbose
    )
    
    # Check optimization files
    opt_valid, opt_issues, opt_stats = check_optimization_files_integrity(
        currency=currency, verbose=verbose
    )
    
    # Combine results
    all_valid = bitcoin_valid and opt_valid
    all_issues = bitcoin_issues + opt_issues
    
    verification_results = {
        "timestamp": datetime.now().isoformat(),
        "overall_valid": all_valid,
        "bitcoin_data": {
            "valid": bitcoin_valid,
            "issues": bitcoin_issues,
            "stats": bitcoin_stats
        },
        "optimization_files": {
            "valid": opt_valid,
            "issues": opt_issues,
            "stats": {
                "valid_count": len(opt_stats["valid_files"]),
                "missing_count": len(opt_stats["missing_files"]),
                "invalid_count": len(opt_stats["invalid_files"]),
                "missing_files": opt_stats["missing_files"],
                "invalid_files": opt_stats["invalid_files"]
            }
        }
    }
    
    if verbose:
        print(f"\nOverall verification result: {'PASS' if all_valid else 'FAIL'}")
        if all_issues:
            print(f"Issues found ({len(all_issues)}):")
            for i, issue in enumerate(all_issues, 1):
                print(f"  {i}. {issue}")
    
    # Save to JSON if requested
    if output_json:
        try:
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, "w") as f:
                json.dump(verification_results, f, indent=2)
            if verbose:
                print(f"Verification results saved to {output_json}")
        except Exception as e:
            print(f"Error saving verification results to {output_json}: {str(e)}")
    
    return all_valid

def main():
    parser = argparse.ArgumentParser(description="Verify Bitcoin data and optimization files integrity")
    parser.add_argument("--currency", type=str, default="AUD", help="Currency code (default: AUD)")
    parser.add_argument("--output", type=str, help="Path to save verification results as JSON")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    
    args = parser.parse_args()
    
    result = run_data_verification(
        currency=args.currency,
        output_json=args.output,
        verbose=not args.quiet
    )
    
    # Return exit code (0 for success, 1 for failure)
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())