"""
Strategy optimizer using Bayesian Optimization.

This module uses scikit-optimize to find optimal parameters for investment strategies,
including the best exchange for each strategy given a certain investment amount.

The optimization process:
1. Defines a search space for parameters and exchanges
2. Uses Gaussian Process optimization to efficiently search for best parameters
3. Returns optimized parameters for each strategy
"""

import numpy as np
import pandas as pd
import polars as pl
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args

from data_fetcher import fetch_bitcoin_price_data
from strategies import (
    dca_strategy,
    value_averaging_strategy,
    maco_strategy,
    rsi_strategy,
    volatility_strategy
)
from fee_models import load_exchange_profiles
from metrics import calculate_max_drawdown, calculate_sortino_ratio


def optimize_dca_strategy(start_date, end_date, currency="AUD", n_calls=50):
    """
    Optimize the exchange selection for the DCA strategy.
    
    Args:
        start_date (str): Start date in DD-MM-YYYY format
        end_date (str): End date in DD-MM-YYYY format
        currency (str): Currency to use for optimization
        n_calls (int): Number of optimization iterations
        
    Returns:
        dict: Optimized parameters and performance metrics
    """
    # Load available exchanges
    exchange_profiles = load_exchange_profiles()
    exchanges = list(exchange_profiles.keys())
    
    # Define search space - for DCA we only need to optimize exchange and investment amount
    space = [
        Categorical(exchanges, name='exchange_id'),
        Categorical([True, False], name='use_discount'),
        Real(10, 500, name='weekly_investment')
    ]
    
    # Define objective function
    @use_named_args(space)
    def objective(exchange_id, use_discount, weekly_investment):
        # Fetch data
        df = fetch_bitcoin_price_data(start_date, end_date, currency)
        
        # Run DCA strategy with these parameters
        result = dca_strategy(df.clone(), weekly_investment, exchange_id, use_discount)
        
        # Calculate performance metrics
        final_btc = result["cumulative_btc"].tail(1).item()
        max_drawdown = calculate_max_drawdown(result)
        sortino = calculate_sortino_ratio(result)
        
        # Our objective is to maximize BTC holdings
        # Since skopt minimizes, return negative of BTC amount
        return -final_btc
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )
    
    # Get best parameters
    best_exchange = result.x[0]
    best_use_discount = result.x[1]
    best_weekly_investment = result.x[2]
    best_btc = -result.fun
    
    # Return optimized parameters and performance
    return {
        "strategy": "dca",
        "best_params": {
            "exchange_id": best_exchange,
            "use_discount": best_use_discount,
            "weekly_investment": best_weekly_investment
        },
        "performance": {
            "final_btc": best_btc
        },
        "result": result
    }


def optimize_maco_strategy(start_date, end_date, currency="AUD", n_calls=50):
    """
    Optimize the MACO strategy parameters and exchange selection.
    
    Args:
        start_date (str): Start date in DD-MM-YYYY format
        end_date (str): End date in DD-MM-YYYY format
        currency (str): Currency to use for optimization
        n_calls (int): Number of optimization iterations
        
    Returns:
        dict: Optimized parameters and performance metrics
    """
    # Load available exchanges
    exchange_profiles = load_exchange_profiles()
    exchanges = list(exchange_profiles.keys())
    
    # Define search space - for MACO we need to optimize window sizes
    space = [
        Categorical(exchanges, name='exchange_id'),
        Categorical([True, False], name='use_discount'),
        Real(10, 500, name='weekly_investment'),
        Integer(5, 50, name='short_window'),
        Integer(50, 200, name='long_window')
    ]
    
    # Define objective function
    @use_named_args(space)
    def objective(exchange_id, use_discount, weekly_investment, short_window, long_window):
        # Fetch data
        df = fetch_bitcoin_price_data(start_date, end_date, currency)
        
        # Skip invalid combinations
        if short_window >= long_window:
            return 0  # Penalty for invalid parameters
        
        # Run MACO strategy with these parameters
        result = maco_strategy(df.clone(), weekly_investment, short_window, long_window)
        
        # Calculate performance metrics
        final_btc = result["cumulative_btc"].tail(1).item()
        
        # Our objective is to maximize BTC holdings
        # Since skopt minimizes, return negative of BTC amount
        return -final_btc
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )
    
    # Get best parameters
    best_exchange = result.x[0]
    best_use_discount = result.x[1]
    best_weekly_investment = result.x[2]
    best_short_window = result.x[3]
    best_long_window = result.x[4]
    best_btc = -result.fun
    
    # Return optimized parameters and performance
    return {
        "strategy": "maco",
        "best_params": {
            "exchange_id": best_exchange,
            "use_discount": best_use_discount,
            "weekly_investment": best_weekly_investment,
            "short_window": best_short_window,
            "long_window": best_long_window
        },
        "performance": {
            "final_btc": best_btc
        },
        "result": result
    }


def optimize_rsi_strategy(start_date, end_date, currency="AUD", n_calls=50):
    """
    Optimize the RSI strategy parameters and exchange selection.
    
    Args:
        start_date (str): Start date in DD-MM-YYYY format
        end_date (str): End date in DD-MM-YYYY format
        currency (str): Currency to use for optimization
        n_calls (int): Number of optimization iterations
        
    Returns:
        dict: Optimized parameters and performance metrics
    """
    # Load available exchanges
    exchange_profiles = load_exchange_profiles()
    exchanges = list(exchange_profiles.keys())
    
    # Define search space - for RSI we optimize the RSI thresholds
    space = [
        Categorical(exchanges, name='exchange_id'),
        Categorical([True, False], name='use_discount'),
        Real(10, 500, name='weekly_investment'),
        Integer(7, 21, name='rsi_period'),
        Integer(20, 40, name='oversold_threshold'),
        Integer(60, 80, name='overbought_threshold')
    ]
    
    # Define objective function
    @use_named_args(space)
    def objective(exchange_id, use_discount, weekly_investment, 
                 rsi_period, oversold_threshold, overbought_threshold):
        # Fetch data
        df = fetch_bitcoin_price_data(start_date, end_date, currency)
        
        # Skip invalid combinations
        if oversold_threshold >= overbought_threshold:
            return 0  # Penalty for invalid parameters
        
        # Run RSI strategy with these parameters
        result = rsi_strategy(
            df.clone(), weekly_investment, rsi_period, 
            oversold_threshold, overbought_threshold
        )
        
        # Calculate performance metrics
        final_btc = result["cumulative_btc"].tail(1).item()
        
        # Our objective is to maximize BTC holdings
        # Since skopt minimizes, return negative of BTC amount
        return -final_btc
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )
    
    # Get best parameters
    best_exchange = result.x[0]
    best_use_discount = result.x[1]
    best_weekly_investment = result.x[2]
    best_rsi_period = result.x[3]
    best_oversold_threshold = result.x[4]
    best_overbought_threshold = result.x[5]
    best_btc = -result.fun
    
    # Return optimized parameters and performance
    return {
        "strategy": "rsi",
        "best_params": {
            "exchange_id": best_exchange,
            "use_discount": best_use_discount,
            "weekly_investment": best_weekly_investment,
            "rsi_period": best_rsi_period,
            "oversold_threshold": best_oversold_threshold,
            "overbought_threshold": best_overbought_threshold
        },
        "performance": {
            "final_btc": best_btc
        },
        "result": result
    }


def optimize_volatility_strategy(start_date, end_date, currency="AUD", n_calls=50):
    """
    Optimize the Volatility strategy parameters and exchange selection.
    
    Args:
        start_date (str): Start date in DD-MM-YYYY format
        end_date (str): End date in DD-MM-YYYY format
        currency (str): Currency to use for optimization
        n_calls (int): Number of optimization iterations
        
    Returns:
        dict: Optimized parameters and performance metrics
    """
    # Load available exchanges
    exchange_profiles = load_exchange_profiles()
    exchanges = list(exchange_profiles.keys())
    
    # Define search space - for Volatility we optimize window and threshold
    space = [
        Categorical(exchanges, name='exchange_id'),
        Categorical([True, False], name='use_discount'),
        Real(10, 500, name='weekly_investment'),
        Integer(5, 30, name='vol_window'),
        Real(0.5, 3.0, name='vol_threshold')
    ]
    
    # Define objective function
    @use_named_args(space)
    def objective(exchange_id, use_discount, weekly_investment, vol_window, vol_threshold):
        # Fetch data
        df = fetch_bitcoin_price_data(start_date, end_date, currency)
        
        # Run Volatility strategy with these parameters
        result = volatility_strategy(
            df.clone(), weekly_investment, vol_window, vol_threshold
        )
        
        # Calculate performance metrics
        final_btc = result["cumulative_btc"].tail(1).item()
        
        # Our objective is to maximize BTC holdings
        # Since skopt minimizes, return negative of BTC amount
        return -final_btc
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=True
    )
    
    # Get best parameters
    best_exchange = result.x[0]
    best_use_discount = result.x[1]
    best_weekly_investment = result.x[2]
    best_vol_window = result.x[3]
    best_vol_threshold = result.x[4]
    best_btc = -result.fun
    
    # Return optimized parameters and performance
    return {
        "strategy": "volatility",
        "best_params": {
            "exchange_id": best_exchange,
            "use_discount": best_use_discount,
            "weekly_investment": best_weekly_investment,
            "vol_window": best_vol_window,
            "vol_threshold": best_vol_threshold
        },
        "performance": {
            "final_btc": best_btc
        },
        "result": result
    }


def optimize_all_strategies(start_date, end_date, currency="AUD", n_calls=30):
    """
    Optimize all strategies and find the overall best strategy.
    
    Args:
        start_date (str): Start date in DD-MM-YYYY format
        end_date (str): End date in DD-MM-YYYY format
        currency (str): Currency to use for optimization
        n_calls (int): Number of optimization iterations per strategy
        
    Returns:
        dict: Results for all strategies and the best overall strategy
    """
    # Optimize each strategy
    dca_results = optimize_dca_strategy(start_date, end_date, currency, n_calls)
    maco_results = optimize_maco_strategy(start_date, end_date, currency, n_calls)
    rsi_results = optimize_rsi_strategy(start_date, end_date, currency, n_calls)
    vol_results = optimize_volatility_strategy(start_date, end_date, currency, n_calls)
    
    # Collect all results
    all_results = {
        "dca": dca_results,
        "maco": maco_results,
        "rsi": rsi_results,
        "volatility": vol_results
    }
    
    # Find the best overall strategy
    best_strategy = max(
        all_results.items(),
        key=lambda x: x[1]["performance"]["final_btc"]
    )
    
    return {
        "all_results": all_results,
        "best_strategy": best_strategy[0],
        "best_params": best_strategy[1]["best_params"],
        "best_performance": best_strategy[1]["performance"]
    }


# Command-line interface for running optimizations
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize cryptocurrency investment strategies")
    parser.add_argument("--strategy", choices=["dca", "maco", "rsi", "volatility", "all"], 
                        default="all", help="Strategy to optimize")
    parser.add_argument("--start_date", default="01-01-2022", 
                        help="Start date (DD-MM-YYYY)")
    parser.add_argument("--end_date", default="01-01-2023", 
                        help="End date (DD-MM-YYYY)")
    parser.add_argument("--currency", default="AUD", 
                        help="Currency to use")
    parser.add_argument("--n_calls", type=int, default=30, 
                        help="Number of optimization iterations")
    
    args = parser.parse_args()
    
    if args.strategy == "dca":
        result = optimize_dca_strategy(args.start_date, args.end_date, args.currency, args.n_calls)
    elif args.strategy == "maco":
        result = optimize_maco_strategy(args.start_date, args.end_date, args.currency, args.n_calls)
    elif args.strategy == "rsi":
        result = optimize_rsi_strategy(args.start_date, args.end_date, args.currency, args.n_calls)
    elif args.strategy == "volatility":
        result = optimize_volatility_strategy(args.start_date, args.end_date, args.currency, args.n_calls)
    else:  # "all"
        result = optimize_all_strategies(args.start_date, args.end_date, args.currency, args.n_calls)
    
    print("\nOptimization Results:")
    print("=====================")
    
    if args.strategy == "all":
        print(f"Best Strategy: {result['best_strategy']}")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Best BTC Performance: {result['best_performance']['final_btc']:.8f}")
    else:
        print(f"Strategy: {result['strategy']}")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Best BTC Performance: {result['performance']['final_btc']:.8f}")