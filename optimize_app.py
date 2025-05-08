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
    This tool uses Bayesian Optimization to find the best parameters for each strategy,
    including the optimal exchange to use based on fee structures.
    
    **Note:** Optimizations are saved to disk as Arrow files to avoid re-running them each time.
    """)
    
    # Sidebar for result parameters
    st.sidebar.header("Select Parameters")
    
    # Date range selection 
    today = datetime.date.today()
    ten_years_ago = today.replace(year=today.year - 10)
    default_start_date = today - timedelta(days=365)  # 1 year ago by default
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        min_value=ten_years_ago,  # Up to 10 years ago
        max_value=today - timedelta(days=30)  # At least 30 days of data
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=today,
        min_value=start_date + timedelta(days=30),
        max_value=today
    )
    
    # Currency selection
    currency = st.sidebar.selectbox("Currency", ["AUD", "USD"])
    
    # Strategy selection
    st.sidebar.header("Strategies to View")
    optimize_dca = st.sidebar.checkbox("Dollar Cost Averaging", value=True)
    optimize_maco = st.sidebar.checkbox("Moving Average Crossover", value=True)
    optimize_rsi = st.sidebar.checkbox("RSI-Based Strategy", value=True)
    optimize_volatility = st.sidebar.checkbox("Volatility-Based Strategy", value=True)
    
    # Button to load optimization results
    load_button = st.sidebar.button("Load Optimization Results", type="primary")
    
    # Display explanatory text
    with st.expander("How Bayesian Optimization Works"):
        st.markdown("""
        ### Bayesian Optimization
        
        Bayesian optimization is a machine learning technique that efficiently searches for the
        optimal parameters by:
        
        1. Building a probabilistic model of the objective function
        2. Using this model to select the most promising parameters to evaluate next
        3. Updating the model with new results to improve future selections
        
        This approach is much more efficient than grid search or random search, especially
        for computationally expensive evaluations like backtesting strategies over long periods.
        
        The optimizer will search for parameters that maximize Bitcoin accumulation, considering:
        - Exchange fees and loyalty discounts
        - Strategy-specific parameters
        - Weekly investment amounts
        """)
    
    # Format dates for file paths and function calls
    start_date_str = start_date.strftime("%d-%m-%Y")
    end_date_str = end_date.strftime("%d-%m-%Y")
    
    selected_strategies = []
    if optimize_dca:
        selected_strategies.append("dca")
    if optimize_maco:
        selected_strategies.append("maco")
    if optimize_rsi:
        selected_strategies.append("rsi")
    if optimize_volatility:
        selected_strategies.append("volatility")
    
    # Function to get file path for a strategy
    def get_optimization_file_path(strategy):
        filename = f"{strategy}_{start_date_str.replace('-', '')}_{end_date_str.replace('-', '')}_{currency}.arrow"
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

    # Handle load button click - load existing results
    if load_button:
        if not selected_strategies:
            st.error("Please select at least one strategy to load results for.")
        else:
            try:
                all_results = {}
                missing_results = []
                
                # Try to load results for each selected strategy
                for strategy in selected_strategies:
                    if optimization_exists(strategy):
                        result = load_optimization_results(strategy)
                        if result is not None:
                            all_results[strategy] = result
                            st.success(f"Loaded optimization results for {strategy.upper()}")
                    else:
                        missing_results.append(strategy)
                
                # Warn about missing results
                if missing_results:
                    missing_str = ", ".join([s.upper() for s in missing_results])
                    st.warning(f"No saved results found for: {missing_str}. Pre-calculated optimization results may not be available for these strategies.")
                
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
                    st.error("No optimization results were loaded. Pre-calculated optimization results may not be available for the selected strategies and date range.")
            
            except Exception as e:
                st.error(f"An error occurred while loading results: {str(e)}")
    else:
        # Show existing optimization files
        optimization_files = list(Path(OPTIMIZATION_DIR).glob("*.arrow")) + list(Path(OPTIMIZATION_DIR).glob("*.ipc"))
        if optimization_files:
            st.info("Saved optimization results exist. Click 'Load Optimization Results' to view them.")
            
            # Show a table of available optimization files
            file_data = []
            for file_path in optimization_files:
                file_name = file_path.name
                parts = file_name.split('_')
                if len(parts) >= 4:
                    strategy = parts[0].upper()
                    # Extract dates from filename
                    start = parts[1]
                    end = parts[2].split('.')[0] if '.' in parts[2] else parts[2]
                    # Format dates for display
                    if len(start) == 8:  # DDMMYYYY format
                        start = f"{start[:2]}-{start[2:4]}-{start[4:]}"
                    if len(end) == 8:  # DDMMYYYY format
                        end = f"{end[:2]}-{end[2:4]}-{end[4:]}"
                    
                    file_data.append({
                        "Strategy": strategy,
                        "Start Date": start,
                        "End Date": end,
                        "File": file_name
                    })
            
            if file_data:
                st.subheader("Available Optimization Results")
                st.dataframe(pd.DataFrame(file_data))
        else:
            # Display instructions when page first loads
            st.info("👈 Select your optimization parameters and click 'Load Optimization Results' to view saved optimization results.")
            
            st.markdown("""
            ## Optimization Results Show:
            
            1. **The best exchange** for each strategy based on fee structures
            2. **Optimal parameters** for each trading strategy
            3. **Ideal weekly investment** amount for maximum BTC accumulation
            4. **Comparative performance** showing which strategy works best
            
            The optimization results are based on historical Bitcoin price data simulation
            of strategy performance under different parameter combinations.
            """)


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