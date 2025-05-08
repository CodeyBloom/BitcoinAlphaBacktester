"""
Strategy optimization page for the Bitcoin Strategy Backtester.

This module provides a Streamlit interface for viewing optimization results
that have been pre-computed and saved to Arrow files.
"""

import streamlit as st
import datetime
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import os
import subprocess
from datetime import timedelta
from pathlib import Path

# Create a directory for storing optimization results if it doesn't exist
OPTIMIZATION_DIR = "data/optimizations"
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

def run_optimizer_page():
    """Run the strategy optimizer page"""
    
    st.title("Bitcoin Strategy Optimizer")
    
    # Just run the optimized view directly
    run_optimizer_view()

def run_optimizer_view():
    """Display optimized strategy results"""
    
    st.markdown("""
    This tool shows the optimal parameters for each investment strategy,
    including the best exchange to use based on fee structures.
    
    **Note:** The optimization results are pre-calculated using Bayesian Optimization.
    """)
    
    # Predefined time periods
    time_periods = {
        "1 Year": 1,
        "5 Years": 5,
        "10 Years": 10
    }
    
    # Currency selection
    currency = st.sidebar.radio("Currency", ["AUD", "USD"], index=0)
    
    # Time period selection
    st.sidebar.header("Time Period")
    selected_period = st.sidebar.radio(
        "Select Time Period",
        list(time_periods.keys()),
        index=0
    )
    
    # Calculate dates based on selected period
    years = time_periods[selected_period]
    end_date = datetime.date.today()
    start_date = end_date.replace(year=end_date.year - years)
    
    # Format dates for file paths and display
    start_date_str = start_date.strftime("%d%m%Y")
    end_date_str = end_date.strftime("%d%m%Y")
    
    # Strategy selection
    st.sidebar.header("Strategies")
    strategy_selections = {}
    for strategy in ["dca", "maco", "rsi", "volatility"]:
        display_name = strategy.upper() if strategy != "maco" else "MACO"
        strategy_selections[strategy] = st.sidebar.checkbox(display_name, value=True)
    
    # Get selected strategies
    selected_strategies = [s for s, selected in strategy_selections.items() if selected]
    
    # Display explanatory text
    with st.expander("How Strategies Were Optimized"):
        st.markdown("""
        ### Bayesian Optimization
        
        The parameters for each strategy were optimized using Bayesian optimization, which:
        
        1. Builds a probabilistic model of the objective function
        2. Efficiently searches the parameter space for optimal results
        3. Maximizes efficiency (BTC per currency unit spent)
        
        The optimizer found parameters that maximize Bitcoin accumulation efficiency, considering:
        - Exchange fees and loyalty discounts
        - Strategy-specific parameters
        - Weekly investment amounts
        
        These optimizations are updated weekly via GitHub Actions to ensure they use the latest market data.
        """)
    
    # Function to get file path for a strategy
    def get_optimization_file_path(strategy):
        filename = f"{strategy}_{start_date_str}_{end_date_str}_{currency}.arrow"
        return os.path.join(OPTIMIZATION_DIR, filename)
    
    # Function to check if optimization results exist for a strategy
    def optimization_exists(strategy):
        file_path = get_optimization_file_path(strategy)
        return os.path.exists(file_path)
    
    # Function to load optimization results for a strategy
    def load_optimization_results(strategy):
        file_path = get_optimization_file_path(strategy)
        if os.path.exists(file_path):
            try:
                # Load Arrow file using Polars
                df = pl.read_ipc(file_path)
                
                # Convert to dictionary format expected by display function
                row = df.row(0, named=True)  # Get first row as dict
                
                # Extract strategy name
                strategy_name = row.get("strategy", strategy)
                
                # Extract parameters
                best_params = {}
                for key, value in row.items():
                    if key.startswith("param_"):
                        param_name = key[6:]  # Remove "param_" prefix
                        best_params[param_name] = value
                
                # Extract performance metrics
                performance = {}
                for key, value in row.items():
                    if key.startswith("performance_"):
                        metric_name = key[12:]  # Remove "performance_" prefix
                        performance[metric_name] = value
                
                # Create result dictionary in expected format
                result = {
                    "strategy": strategy_name,
                    "best_params": best_params,
                    "performance": performance
                }
                
                return result
            except Exception as e:
                st.error(f"Error loading results for {strategy}: {str(e)}")
                return None
        
        # If file doesn't exist, generate sample data for demonstration
        # This will be replaced by real data from GitHub Actions in production
        st.warning(f"No optimization file found for {strategy} with the selected time period. Using sample data for demonstration.")
        
        # Create sample data based on strategy
        if strategy == "dca":
            return {
                "strategy": strategy,
                "best_params": {
                    "exchange_id": "binance",
                    "weekly_investment": 100.0,
                    "use_discount": True
                },
                "performance": {
                    "final_btc": 0.45678912,
                    "max_drawdown": 0.21,
                    "sortino_ratio": 1.35
                }
            }
        elif strategy == "maco":
            return {
                "strategy": strategy,
                "best_params": {
                    "exchange_id": "coinbase",
                    "weekly_investment": 150.0,
                    "use_discount": False,
                    "short_window": 15,
                    "long_window": 75
                },
                "performance": {
                    "final_btc": 0.55678912,
                    "max_drawdown": 0.28,
                    "sortino_ratio": 1.12
                }
            }
        elif strategy == "rsi":
            return {
                "strategy": strategy,
                "best_params": {
                    "exchange_id": "kraken",
                    "weekly_investment": 120.0,
                    "use_discount": True,
                    "rsi_period": 12,
                    "oversold_threshold": 28,
                    "overbought_threshold": 72
                },
                "performance": {
                    "final_btc": 0.60123456,
                    "max_drawdown": 0.25,
                    "sortino_ratio": 1.48
                }
            }
        else:  # volatility
            return {
                "strategy": strategy,
                "best_params": {
                    "exchange_id": "binance",
                    "weekly_investment": 200.0,
                    "use_discount": True,
                    "vol_window": 18,
                    "vol_threshold": 1.75
                },
                "performance": {
                    "final_btc": 0.58123456,
                    "max_drawdown": 0.30,
                    "sortino_ratio": 1.22
                }
            }
    
    # Automatically load results for selected strategies
    if selected_strategies:
        all_results = {}
        
        # Load results for each selected strategy
        for strategy in selected_strategies:
            result = load_optimization_results(strategy)
            if result is not None:
                all_results[strategy] = result
        
        # Display results
        if len(all_results) == 1:
            # Single strategy optimization
            strategy = list(all_results.keys())[0]
            display_optimization_results(all_results[strategy], single_strategy=True, currency=currency)
        elif len(all_results) > 1:
            # Multi-strategy optimization - find most efficient strategy
            most_efficient_strategy = max(
                all_results.items(),
                key=lambda x: x[1]["performance"]["final_btc"] / (x[1]["best_params"]["weekly_investment"] * 52)
            )[0]
            
            display_optimization_results(all_results, most_efficient_strategy, single_strategy=False, currency=currency)
        else:
            st.error("No optimization results could be loaded.")
    else:
        st.warning("Please select at least one strategy to view results.")


def display_optimization_results(results, best_strategy_name=None, single_strategy=True, currency="AUD"):
    """Display the optimization results in the Streamlit interface"""
    
    st.header("Optimization Results")
    
    if single_strategy:
        # Display results for a single strategy
        strategy_name = results["strategy"].upper()
        best_params = results["best_params"]
        best_performance = results["performance"]
        
        # Calculate total investment
        weekly_investment = best_params.get("weekly_investment", 0)
        # Assume 52 weeks for 1 year of investment
        years = 1  # Default to 1 year
        weeks = 52 * years
        total_investment = weekly_investment * weeks
        
        # Calculate efficiency (BTC per currency unit spent)
        efficiency = best_performance["final_btc"] / total_investment if total_investment > 0 else 0
        
        st.subheader(f"{strategy_name} Strategy Optimization")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Best Parameters")
            for param, value in best_params.items():
                # Format nicely
                if param == "exchange_id":
                    st.info(f"**Exchange:** {value}")
                elif param == "use_discount":
                    st.info(f"**Use Loyalty Discount:** {'Yes' if value else 'No'}")
                elif param == "weekly_investment":
                    st.info(f"**Weekly Investment:** {value:.2f} {currency}")
                else:
                    # Strategy-specific parameters
                    formatted_param = param.replace("_", " ").title()
                    st.info(f"**{formatted_param}:** {value}")
        
        with col2:
            st.markdown("### Performance")
            st.metric(
                f"Efficiency (BTC per {currency})",
                f"{efficiency:.8f} BTC/{currency}"
            )
            
            st.metric(
                "Final BTC Holdings",
                f"{best_performance['final_btc']:.8f} BTC"
            )
            
            st.metric(
                "Total Investment",
                f"{total_investment:.2f} {currency}"
            )
            
            # If we have additional metrics, display them
            if "max_drawdown" in best_performance:
                st.metric(
                    "Max Drawdown",
                    f"{best_performance['max_drawdown']*100:.2f}%"
                )
            
            if "sortino_ratio" in best_performance:
                st.metric(
                    "Sortino Ratio",
                    f"{best_performance['sortino_ratio']:.2f}"
                )
    else:
        # Display comparative results for multiple strategies
        st.subheader("Strategy Comparison")
        
        # Create a table of all results
        comparison_data = []
        
        for strategy_name, result in results.items():
            weekly_investment = result["best_params"]["weekly_investment"]
            weeks = 52  # Assuming one year of investment
            total_investment = weekly_investment * weeks
            efficiency = result["performance"]["final_btc"] / total_investment if total_investment > 0 else 0
            
            comparison_data.append({
                "Strategy": strategy_name.upper(),
                "Exchange": result["best_params"]["exchange_id"],
                f"Weekly Investment ({currency})": f"{weekly_investment:.2f}",
                f"Total Investment ({currency})": f"{total_investment:.2f}",
                "BTC Accumulated": f"{result['performance']['final_btc']:.8f}",
                f"Efficiency (BTC/{currency})": f"{efficiency:.8f}",
                "Use Discount": "Yes" if result["best_params"]["use_discount"] else "No"
            })
        
        # Convert to DataFrame for display
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by efficiency descending
        efficiency_col = f"Efficiency (BTC/{currency})"
        comparison_df["Efficiency_Sort"] = [float(row[efficiency_col]) for _, row in comparison_df.iterrows()]
        comparison_df = comparison_df.sort_values("Efficiency_Sort", ascending=False)
        comparison_df = comparison_df.drop("Efficiency_Sort", axis=1)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Find the most efficient strategy
        most_efficient_strategy = max(
            results.items(),
            key=lambda x: x[1]["performance"]["final_btc"] / (x[1]["best_params"]["weekly_investment"] * 52)
        )[0]
        
        # Highlight the most efficient strategy
        most_efficient_display = most_efficient_strategy.upper()
        st.success(f"The most efficient strategy is **{most_efficient_display}** with these parameters:")
        
        efficient_result = results[most_efficient_strategy]
        efficient_params = efficient_result["best_params"]
        
        # Calculate efficiency metrics
        weekly_investment = efficient_params.get("weekly_investment", 0)
        weeks = 52  # Assuming one year of investment
        total_investment = weekly_investment * weeks
        efficiency = efficient_result["performance"]["final_btc"] / total_investment if total_investment > 0 else 0
        
        # Create two columns for the efficient strategy details
        col1, col2 = st.columns(2)
        
        with col1:
            # Show most efficient strategy parameters
            for param, value in efficient_params.items():
                # Format nicely
                if param == "exchange_id":
                    st.info(f"**Exchange:** {value}")
                elif param == "use_discount":
                    st.info(f"**Use Loyalty Discount:** {'Yes' if value else 'No'}")
                elif param == "weekly_investment":
                    st.info(f"**Weekly Investment:** {value:.2f} {currency}")
                else:
                    # Strategy-specific parameters
                    formatted_param = param.replace("_", " ").title()
                    st.info(f"**{formatted_param}:** {value}")
                    
        with col2:
            st.info(f"**Efficiency:** {efficiency:.8f} BTC/{currency}")
            st.info(f"**Final BTC:** {efficient_result['performance']['final_btc']:.8f} BTC")
            st.info(f"**Total Investment:** {total_investment:.2f} {currency}")


if __name__ == "__main__":
    run_optimizer_page()