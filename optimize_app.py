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
    st.markdown("""
    This tool shows the optimal parameters for each investment strategy,
    including the best exchange to use based on fee structures.
    
    **Note:** The optimization results are pre-calculated using Bayesian Optimization.
    """)
    
    # Find all available optimization files
    optimization_files = list(Path(OPTIMIZATION_DIR).glob("*.arrow")) + list(Path(OPTIMIZATION_DIR).glob("*.ipc"))
    
    if not optimization_files:
        st.warning("No optimization results are available.")
        
        st.markdown("""
        ## Optimization Results Show:
        
        1. **The best exchange** for each strategy based on fee structures
        2. **Optimal parameters** for each trading strategy
        3. **Ideal weekly investment** amount for maximum BTC accumulation
        4. **Comparative performance** showing which strategy works best
        
        The optimization results are based on historical Bitcoin price data simulation
        of strategy performance under different parameter combinations.
        """)
        return
    
    # Extract available dates, currencies, and strategies from file names
    available_data = {}
    for file_path in optimization_files:
        file_name = file_path.name
        parts = file_name.split('_')
        
        if len(parts) >= 4:
            strategy = parts[0]
            start_date_str = parts[1]
            end_date_str = parts[2].split('.')[0] if '.' in parts[2] else parts[2]
            currency = parts[3].split('.')[0] if '.' in parts[3] else parts[3]
            
            # Format dates for display
            if len(start_date_str) == 8:  # DDMMYYYY format
                start_date_display = f"{start_date_str[:2]}-{start_date_str[2:4]}-{start_date_str[4:]}"
            else:
                start_date_display = start_date_str
                
            if len(end_date_str) == 8:  # DDMMYYYY format
                end_date_display = f"{end_date_str[:2]}-{end_date_str[2:4]}-{end_date_str[4:]}"
            else:
                end_date_display = end_date_str
            
            key = (start_date_str, end_date_str, currency)
            if key not in available_data:
                available_data[key] = {
                    "start_date_display": start_date_display,
                    "end_date_display": end_date_display,
                    "currency": currency,
                    "strategies": []
                }
            
            available_data[key]["strategies"].append(strategy)
    
    # Sidebar for selecting available data
    st.sidebar.header("Available Optimization Results")
    
    # Group the options by date range and currency
    date_options = []
    for key, data in available_data.items():
        date_option = f"{data['start_date_display']} to {data['end_date_display']} ({data['currency']})"
        date_options.append((date_option, key))
    
    # Sort by most recent first
    date_options.sort(key=lambda x: x[1][1], reverse=True)
    
    selected_option = st.sidebar.selectbox(
        "Select Date Range & Currency",
        options=[option[0] for option in date_options],
        index=0
    )
    
    # Find the selected key
    selected_key = None
    for option in date_options:
        if option[0] == selected_option:
            selected_key = option[1]
            break
    
    if selected_key:
        start_date_str, end_date_str, currency = selected_key
        available_strategies = available_data[selected_key]["strategies"]
        
        # Display the strategies available for this date range
        st.sidebar.header("Available Strategies")
        
        # Create checkboxes for each available strategy
        strategy_selections = {}
        for strategy in ["dca", "maco", "rsi", "volatility"]:
            if strategy in available_strategies:
                strategy_selections[strategy] = st.sidebar.checkbox(
                    strategy.upper() if strategy != "maco" else "MACO", 
                    value=True
                )
        
        # Get selected strategies
        selected_strategies = [s for s, selected in strategy_selections.items() if selected]
        
        # Display explanatory text
        with st.expander("How Strategies Were Optimized"):
            st.markdown("""
            ### Bayesian Optimization
            
            The parameters for each strategy were optimized using Bayesian optimization, which:
            
            1. Builds a probabilistic model of the objective function
            2. Efficiently searches the parameter space for optimal results
            3. Maximizes Bitcoin accumulation while considering exchange fees
            
            The optimizer found parameters that maximize Bitcoin accumulation, considering:
            - Exchange fees and loyalty discounts
            - Strategy-specific parameters
            - Weekly investment amounts
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
            return None
        
        # Automatically load results for selected strategies
        if selected_strategies:
            all_results = {}
            
            # Load results for each selected strategy
            for strategy in selected_strategies:
                if optimization_exists(strategy):
                    result = load_optimization_results(strategy)
                    if result is not None:
                        all_results[strategy] = result
            
            # Display results
            if len(all_results) == 1:
                # Single strategy optimization
                strategy = list(all_results.keys())[0]
                display_optimization_results(all_results[strategy], single_strategy=True)
            elif len(all_results) > 1:
                # Multi-strategy optimization
                # Find best strategy among optimized ones
                best_strategy_name = max(
                    all_results.items(),
                    key=lambda x: x[1]["performance"]["final_btc"]
                )[0]
                
                display_optimization_results(all_results, best_strategy_name, single_strategy=False)
            else:
                st.error("No optimization results could be loaded.")
        else:
            st.warning("Please select at least one strategy to view results.")
    else:
        st.error("Error: Could not determine selected date range.")


def display_optimization_results(results, best_strategy_name=None, single_strategy=True):
    """Display the optimization results in the Streamlit interface"""
    
    st.header("Optimization Results")
    
    if single_strategy:
        # Display results for a single strategy
        strategy_name = results["strategy"].upper()
        best_params = results["best_params"]
        best_performance = results["performance"]
        
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
                    st.info(f"**Weekly Investment:** {value:.2f}")
                else:
                    # Strategy-specific parameters
                    formatted_param = param.replace("_", " ").title()
                    st.info(f"**{formatted_param}:** {value}")
        
        with col2:
            st.markdown("### Performance")
            st.metric(
                "Final BTC Holdings",
                f"{best_performance['final_btc']:.8f} BTC"
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
            comparison_data.append({
                "Strategy": strategy_name.upper(),
                "Exchange": result["best_params"]["exchange_id"],
                "Weekly Investment": f"{result['best_params']['weekly_investment']:.2f}",
                "BTC Accumulated": f"{result['performance']['final_btc']:.8f}",
                "Use Discount": "Yes" if result["best_params"]["use_discount"] else "No"
            })
        
        # Convert to DataFrame for display
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Highlight the best strategy
        if best_strategy_name is not None:
            strategy_display = best_strategy_name.upper()
        else:
            strategy_display = "UNKNOWN"
        st.success(f"The best performing strategy is **{strategy_display}** with these parameters:")
        
        best_result = results[best_strategy_name]
        best_params = best_result["best_params"]
        
        # Show best strategy details
        for param, value in best_params.items():
            # Format nicely
            if param == "exchange_id":
                st.info(f"**Exchange:** {value}")
            elif param == "use_discount":
                st.info(f"**Use Loyalty Discount:** {'Yes' if value else 'No'}")
            elif param == "weekly_investment":
                st.info(f"**Weekly Investment:** {value:.2f}")
            else:
                # Strategy-specific parameters
                formatted_param = param.replace("_", " ").title()
                st.info(f"**{formatted_param}:** {value}")


if __name__ == "__main__":
    run_optimizer_page()