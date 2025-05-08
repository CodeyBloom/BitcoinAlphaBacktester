"""
Strategy optimization page for the Bitcoin Strategy Backtester.

This module provides a Streamlit interface for optimizing trading strategies
using Bayesian optimization to find the best parameters and exchanges.
"""

import streamlit as st
import datetime
import plotly.graph_objects as go
import pandas as pd
import polars as pl
from datetime import timedelta

from optimize_strategies import (
    optimize_dca_strategy,
    optimize_maco_strategy,
    optimize_rsi_strategy,
    optimize_volatility_strategy,
    optimize_all_strategies
)
from fee_models import load_exchange_profiles


def run_optimizer_page():
    """Run the strategy optimizer page"""
    
    st.title("Bitcoin Strategy Optimizer")
    st.markdown("""
    This tool uses Bayesian Optimization to find the best parameters for each strategy,
    including the optimal exchange to use based on fee structures.
    
    The optimization process can take several minutes depending on the date range and number of iterations.
    """)
    
    # Sidebar for optimizer parameters
    st.sidebar.header("Optimization Parameters")
    
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
    
    # Optimization iterations
    n_calls = st.sidebar.slider(
        "Optimization Iterations", 
        min_value=10, 
        max_value=100, 
        value=30,
        help="More iterations give better results but take longer"
    )
    
    # Strategy selection
    st.sidebar.header("Strategies to Optimize")
    optimize_dca = st.sidebar.checkbox("Dollar Cost Averaging", value=True)
    optimize_maco = st.sidebar.checkbox("Moving Average Crossover", value=True)
    optimize_rsi = st.sidebar.checkbox("RSI-Based Strategy", value=True)
    optimize_volatility = st.sidebar.checkbox("Volatility-Based Strategy", value=True)
    
    # Button to start optimization
    optimize_button = st.sidebar.button("Run Optimization", type="primary")
    
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
    
    if optimize_button:
        # Format dates for API
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
        
        if not selected_strategies:
            st.error("Please select at least one strategy to optimize.")
        else:
            try:
                with st.spinner("Running optimization... This may take several minutes."):
                    if len(selected_strategies) == 1:
                        # Optimize single strategy
                        strategy = selected_strategies[0]
                        if strategy == "dca":
                            result = optimize_dca_strategy(start_date_str, end_date_str, currency, n_calls)
                        elif strategy == "maco":
                            result = optimize_maco_strategy(start_date_str, end_date_str, currency, n_calls)
                        elif strategy == "rsi":
                            result = optimize_rsi_strategy(start_date_str, end_date_str, currency, n_calls)
                        elif strategy == "volatility":
                            result = optimize_volatility_strategy(start_date_str, end_date_str, currency, n_calls)
                        
                        if result is not None:
                            display_optimization_results(result, single_strategy=True)
                        else:
                            st.error("Optimization failed. Please try again with different parameters.")
                    else:
                        # Optimize subset of strategies or all
                        all_results = {}
                        
                        for strategy in selected_strategies:
                            st.info(f"Optimizing {strategy.upper()} strategy...")
                            if strategy == "dca":
                                result = optimize_dca_strategy(start_date_str, end_date_str, currency, n_calls)
                            elif strategy == "maco":
                                result = optimize_maco_strategy(start_date_str, end_date_str, currency, n_calls)
                            elif strategy == "rsi":
                                result = optimize_rsi_strategy(start_date_str, end_date_str, currency, n_calls)
                            elif strategy == "volatility":
                                result = optimize_volatility_strategy(start_date_str, end_date_str, currency, n_calls)
                            
                            # Only add result if not None
                            if result is not None:
                                all_results[strategy] = result
                        
                        # Check if we have any valid results
                        if all_results:
                            # Find best strategy among the optimized ones
                            best_strategy_name = max(
                                all_results.items(),
                                key=lambda x: x[1]["performance"]["final_btc"]
                            )[0]
                            
                            best_strategy = all_results[best_strategy_name]
                            
                            display_optimization_results(all_results, best_strategy_name, single_strategy=False)
                        else:
                            st.error("No valid optimization results were found. Please try different parameters or strategies.")
            except Exception as e:
                st.error(f"An error occurred during optimization: {str(e)}")
    else:
        # Display instructions when page first loads
        st.info("👈 Select your optimization parameters and click 'Run Optimization' to start.")
        
        st.markdown("""
        ## What the optimizer will find:
        
        1. **The best exchange** for each strategy based on fee structures
        2. **Optimal parameters** for each trading strategy
        3. **Ideal weekly investment** amount for maximum BTC accumulation
        4. **Comparative performance** showing which strategy works best
        
        The optimization uses historical Bitcoin price data from the selected date range to simulate 
        strategy performance under different parameter combinations.
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
        st.success(f"The best performing strategy is **{best_strategy_name.upper()}** with these parameters:")
        
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