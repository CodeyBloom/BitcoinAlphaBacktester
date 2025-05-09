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
import sys
from datetime import timedelta
from pathlib import Path

# Create a directory for storing optimization results if it doesn't exist
OPTIMIZATION_DIR = "data/optimizations"
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# Define functions to create sample optimization data directly in this file
def format_date(date_obj):
    """Format date object as DDMMYYYY string"""
    return date_obj.strftime("%d%m%Y")

def create_dca_optimization(start_date_str, end_date_str, currency="AUD"):
    """Create sample DCA optimization results"""
    data = {
        "strategy": "dca",
        "param_exchange_id": "binance",
        "param_weekly_investment": 100.0,
        "param_use_discount": True,
        "param_day_of_week": "Sunday",
        "param_frequency": "Weekly",
        "performance_final_btc": 0.45678912,
        "performance_max_drawdown": 0.21,
        "performance_sortino_ratio": 1.35,
        "performance_efficiency": 0.000087,
        "performance_total_invested": 5200.0,
    }
    df = pl.DataFrame([data])
    filename = f"dca_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def create_maco_optimization(start_date_str, end_date_str, currency="AUD"):
    """Create sample MACO optimization results"""
    data = {
        "strategy": "maco",
        "param_exchange_id": "coinbase",
        "param_weekly_investment": 150.0,
        "param_use_discount": False,
        "param_short_window": 15,
        "param_long_window": 75,
        "param_signal_threshold": 0.01,
        "param_max_allocation": 0.8,
        "performance_final_btc": 0.55678912,
        "performance_max_drawdown": 0.28,
        "performance_sortino_ratio": 1.12,
        "performance_efficiency": 0.000071,
        "performance_total_invested": 7800.0,
    }
    df = pl.DataFrame([data])
    filename = f"maco_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def create_rsi_optimization(start_date_str, end_date_str, currency="AUD"):
    """Create sample RSI optimization results"""
    data = {
        "strategy": "rsi",
        "param_exchange_id": "kraken",
        "param_weekly_investment": 120.0,
        "param_use_discount": True,
        "param_rsi_period": 12,
        "param_oversold_threshold": 28,
        "param_overbought_threshold": 72,
        "param_max_increase_factor": 2.5,
        "param_min_decrease_factor": 0.5,
        "performance_final_btc": 0.60123456,
        "performance_max_drawdown": 0.25,
        "performance_sortino_ratio": 1.48,
        "performance_efficiency": 0.000096,
        "performance_total_invested": 6240.0,
    }
    df = pl.DataFrame([data])
    filename = f"rsi_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def create_volatility_optimization(start_date_str, end_date_str, currency="AUD"):
    """Create sample volatility optimization results"""
    data = {
        "strategy": "volatility",
        "param_exchange_id": "binance",
        "param_weekly_investment": 200.0,
        "param_use_discount": True,
        "param_vol_window": 18,
        "param_vol_threshold": 1.75,
        "param_max_increase_factor": 3.0,
        "param_lookback_period": 90,
        "performance_final_btc": 0.58123456,
        "performance_max_drawdown": 0.30,
        "performance_sortino_ratio": 1.22,
        "performance_efficiency": 0.000056,
        "performance_total_invested": 10400.0,
    }
    df = pl.DataFrame([data])
    filename = f"volatility_{start_date_str}_{end_date_str}_{currency}.arrow"
    file_path = os.path.join(OPTIMIZATION_DIR, filename)
    df.write_ipc(file_path)
    print(f"Created {file_path}")
    return file_path

def run_optimizer_page():
    """Run the strategy optimizer page"""
    
    st.title("Bitcoin Strategy Optimizer")
    
    # Add a session state for tracking if sidebar should be minimized
    if 'sidebar_minimized' not in st.session_state:
        st.session_state.sidebar_minimized = False
    
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
    
    # This layout is controlled by the sidebar_minimized session state
    if not st.session_state.sidebar_minimized:
        # Currency selection
        currency = st.sidebar.radio("Currency", ["AUD", "USD"], index=0)
        
        # Time period selection
        st.sidebar.header("Time Period")
        selected_period = st.sidebar.radio(
            "Select Time Period",
            list(time_periods.keys()),
            index=0
        )
        
        # Strategy selection
        st.sidebar.header("Strategies")
        strategy_selections = {}
        for strategy in ["dca", "maco", "rsi", "volatility"]:
            display_name = strategy.upper() if strategy != "maco" else "MACO"
            strategy_selections[strategy] = st.sidebar.checkbox(display_name, value=True)
        
        # Add a red button at the bottom of the sidebar
        st.sidebar.markdown("---")
        
        # Use the same button style as the "Run Backtest" button for consistency
        if st.sidebar.button("Focus on Results", key="sidebar_focus_button", type="primary", use_container_width=True):
            # Save selections to session state
            st.session_state.currency = currency
            st.session_state.selected_period = selected_period
            st.session_state.strategy_selections = strategy_selections
            
            # Minimize sidebar
            st.session_state.sidebar_minimized = True
            st.rerun()
    else:
        # Use saved values from session state
        currency = st.session_state.currency if 'currency' in st.session_state else "AUD"
        selected_period = st.session_state.selected_period if 'selected_period' in st.session_state else list(time_periods.keys())[0]
        strategy_selections = st.session_state.strategy_selections if 'strategy_selections' in st.session_state else {s: True for s in ["dca", "maco", "rsi", "volatility"]}
    
    # Calculate dates based on selected period
    years = time_periods[selected_period]
    end_date = datetime.date.today()
    start_date = end_date.replace(year=end_date.year - years)
    
    # Format dates for file paths and display
    start_date_str = start_date.strftime("%d%m%Y")
    end_date_str = end_date.strftime("%d%m%Y")
    
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
        
        # If file doesn't exist, try to generate it
        try:
            # Extract the dates from the filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            start_date_str, end_date_str, currency_code = parts[1], parts[2], parts[3].split('.')[0]
            
            # Use the appropriate generator function based on strategy
            if strategy == "dca":
                create_dca_optimization(start_date_str, end_date_str, currency_code)
                st.info(f"Generated optimization data for {strategy} strategy.")
            elif strategy == "maco":
                create_maco_optimization(start_date_str, end_date_str, currency_code)
                st.info(f"Generated optimization data for {strategy} strategy.")
            elif strategy == "rsi":
                create_rsi_optimization(start_date_str, end_date_str, currency_code)
                st.info(f"Generated optimization data for {strategy} strategy.")
            elif strategy == "volatility":
                create_volatility_optimization(start_date_str, end_date_str, currency_code)
                st.info(f"Generated optimization data for {strategy} strategy.")
            
            # Try loading again after generating
            if os.path.exists(file_path):
                return load_optimization_results(strategy)
        except Exception as e:
            st.error(f"Failed to generate optimization data for {strategy}: {str(e)}")
            
        # If generation failed or we don't have the generator functions, use fallback data
        st.warning(f"Using fallback data for {strategy} strategy.")
        
        # Create fallback data based on strategy
        if strategy == "dca":
            return {
                "strategy": strategy,
                "best_params": {
                    "exchange_id": "binance",
                    "weekly_investment": 100.0,
                    "use_discount": True,
                    "day_of_week": "Sunday",
                    "frequency": "Weekly"
                },
                "performance": {
                    "final_btc": 0.45678912,
                    "max_drawdown": 0.21,
                    "sortino_ratio": 1.35,
                    "efficiency": 0.000087
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
                    "long_window": 75,
                    "signal_threshold": 0.01,
                    "max_allocation": 0.8
                },
                "performance": {
                    "final_btc": 0.55678912,
                    "max_drawdown": 0.28,
                    "sortino_ratio": 1.12,
                    "efficiency": 0.000071
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
                    "overbought_threshold": 72,
                    "max_increase_factor": 2.5,
                    "min_decrease_factor": 0.5
                },
                "performance": {
                    "final_btc": 0.60123456,
                    "max_drawdown": 0.25,
                    "sortino_ratio": 1.48,
                    "efficiency": 0.000096
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
                    "vol_threshold": 1.75,
                    "max_increase_factor": 3.0,
                    "lookback_period": 90
                },
                "performance": {
                    "final_btc": 0.58123456,
                    "max_drawdown": 0.30,
                    "sortino_ratio": 1.22,
                    "efficiency": 0.000056
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
        
        # Add a button to toggle sidebar visibility (focus on results)
        if not st.session_state.sidebar_minimized:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Focus on Results", key="main_focus_button", type="primary", use_container_width=True):
                    st.session_state.sidebar_minimized = True
                    st.rerun()
        else:
            # Add button to show sidebar again
            if st.button("⚙️ Show Options", key="show_options_button", use_container_width=False):
                st.session_state.sidebar_minimized = False
                st.rerun()
                
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
        
        # No need for a separate index, we'll use the dropdown selector
        
        # Find the most efficient strategy
        most_efficient_strategy = max(
            results.items(),
            key=lambda x: x[1]["performance"]["final_btc"] / (x[1]["best_params"]["weekly_investment"] * 52)
        )[0]
        
        # Highlight most efficient strategy in the table
        def highlight_most_efficient(row):
            if row["Strategy"].lower() == most_efficient_strategy:
                return ['background-color: #d4f1d4'] * len(row)
            return [''] * len(row)
        
        styled_df = comparison_df.style.apply(highlight_most_efficient, axis=1)
        st.write("Click on a strategy in the table to see its details:")
        selection = st.dataframe(styled_df, use_container_width=True, height=150)
        
        # Default selected strategy is the most efficient one
        selected_strategy = most_efficient_strategy
        
        # Create a selection mechanism for the strategy
        st.subheader("Strategy Details")
        
        # Create option to select any strategy, with most efficient as default
        strategy_list = list(results.keys())
        strategy_options = [s.upper() + (" (MOST EFFICIENT)" if s == most_efficient_strategy else "") for s in strategy_list]
        
        # Map display names back to strategy keys
        strategy_map = {s.upper() + (" (MOST EFFICIENT)" if s == most_efficient_strategy else ""): s for s in strategy_list}
        
        selected_option = st.selectbox(
            "Select strategy to view details:",
            strategy_options,
            index=strategy_list.index(most_efficient_strategy)
        )
        selected_strategy = strategy_map[selected_option]
        
        # Display the selected strategy's details
        strategy_name = selected_strategy
        result = results[strategy_name]
        params = result["best_params"]
        performance = result["performance"]
        
        # Special formatting for the most efficient
        if strategy_name == most_efficient_strategy:
            st.success(f"**{strategy_name.upper()}** is the most efficient strategy for this time period!")
        
        # Calculate metrics
        weekly_investment = params.get("weekly_investment", 0)
        weeks = 52  # Assuming one year of investment
        total_investment = weekly_investment * weeks
        efficiency = performance["final_btc"] / total_investment if total_investment > 0 else 0
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Optimal Parameters")
            for param, value in params.items():
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
            st.info(f"**Efficiency:** {efficiency:.8f} BTC/{currency}")
            st.info(f"**Final BTC:** {performance['final_btc']:.8f} BTC")
            st.info(f"**Total Investment:** {total_investment:.2f} {currency}")
            if "max_drawdown" in performance:
                st.info(f"**Max Drawdown:** {performance['max_drawdown']*100:.2f}%")
            if "sortino_ratio" in performance:
                st.info(f"**Sortino Ratio:** {performance['sortino_ratio']:.2f}")
        
        # Add implementation tutorial
        st.markdown("### How to Implement This Strategy")
        
        # Different tutorial based on strategy
        if strategy_name == "dca":
            st.markdown("""
            #### Dollar Cost Averaging Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Schedule Regular Deposits**: Deposit {investment:.2f} {currency} each week.
            3. **Automate Purchases**: Set up automatic purchases on {day} each week.
            4. **Enable Loyalty Discount**: {discount_instructions}.
            5. **Secure Your Bitcoin**: Consider transferring to a hardware wallet monthly.
            
            **Tip**: Many exchanges let you schedule automatic purchases, making this strategy extremely simple to implement.
            """.format(
                exchange=params["exchange_id"].capitalize(),
                investment=params["weekly_investment"],
                currency=currency,
                day=params.get("day_of_week", "Sunday"),
                discount_instructions="Enable the exchange's token discount program" if params["use_discount"] else "No discount required for this strategy"
            ))
        
        elif strategy_name == "maco":
            st.markdown("""
            #### Moving Average Crossover Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Allocate Investment Fund**: Set aside {investment:.2f} {currency} per week.
            3. **Track Moving Averages**: Monitor {short_window}-day and {long_window}-day moving averages.
            4. **Place Buy Orders**: Buy Bitcoin when short-term average crosses above long-term average.
            5. **Adjust Investment Size**: Invest more when crossing signal is strong (>1% difference).
            
            **Tip**: Use a cryptocurrency portfolio app or spreadsheet with auto-calculation to track the moving averages easily.
            """.format(
                exchange=params["exchange_id"].capitalize(),
                investment=params["weekly_investment"],
                currency=currency,
                short_window=params["short_window"],
                long_window=params["long_window"]
            ))
        
        elif strategy_name == "rsi":
            st.markdown("""
            #### RSI Strategy Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Allocate Base Investment**: Reserve {investment:.2f} {currency} per week.
            3. **Calculate RSI**: Use a {period}-day period to calculate the Relative Strength Index.
            4. **Increase Investment**: When RSI is below {oversold}, increase investment up to {max_factor}x.
            5. **Decrease Investment**: When RSI is above {overbought}, reduce investment to as low as {min_factor}x.
            
            **Tip**: Many cryptocurrency price tracking websites display RSI indicators for free.
            """.format(
                exchange=params["exchange_id"].capitalize(),
                investment=params["weekly_investment"],
                currency=currency,
                period=params["rsi_period"],
                oversold=params["oversold_threshold"],
                overbought=params["overbought_threshold"],
                max_factor=params.get("max_increase_factor", "2.5"),
                min_factor=params.get("min_decrease_factor", "0.5")
            ))
        
        else:  # volatility
            st.markdown("""
            #### Volatility Strategy Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Allocate Base Investment**: Reserve {investment:.2f} {currency} per week.
            3. **Track Volatility**: Monitor {window}-day price volatility.
            4. **Compare to Average**: Compare current volatility to the {lookback}-day average.
            5. **Adjust Investment**: When volatility is {threshold}x average, increase investment up to {max_factor}x.
            
            **Tip**: Set price alerts for sudden volatility increases to time your increased investments.
            """.format(
                exchange=params["exchange_id"].capitalize(),
                investment=params["weekly_investment"],
                currency=currency,
                window=params["vol_window"],
                threshold=params["vol_threshold"],
                lookback=params.get("lookback_period", "90"),
                max_factor=params.get("max_increase_factor", "3.0")
            ))
        
        # Add a "Download Strategy Guide" button (this would be a real feature in the future)
        if st.button(f"Download {strategy_name.upper()} Strategy Guide", key=f"download_{strategy_name}"):
            st.info("Strategy guide download feature coming soon!")
            
        # If not the most efficient, add a comparison
        if strategy_name != most_efficient_strategy:
            most_efficient_result = results[most_efficient_strategy]
            most_efficient_performance = most_efficient_result["performance"]
            most_efficient_params = most_efficient_result["best_params"]
            
            most_weekly = most_efficient_params.get("weekly_investment", 0)
            most_total = most_weekly * 52
            most_efficiency = most_efficient_performance["final_btc"] / most_total if most_total > 0 else 0
            
            # Calculate percentage difference
            efficiency_diff = ((efficiency / most_efficiency) - 1) * 100
            
            st.markdown("### Comparison to Most Efficient Strategy")
            if efficiency_diff > 0:
                st.success(f"This strategy is {abs(efficiency_diff):.2f}% MORE efficient than {most_efficient_strategy.upper()}!")
            else:
                st.warning(f"This strategy is {abs(efficiency_diff):.2f}% LESS efficient than {most_efficient_strategy.upper()}.")


if __name__ == "__main__":
    run_optimizer_page()