import streamlit as st
import polars as pl
import datetime
import plotly.graph_objects as go
import os
import sys
from datetime import timedelta

# Import from the refactored modules
from data_fetcher import fetch_bitcoin_price_data
from strategies import (
    dca_strategy, 
    value_averaging_strategy, 
    maco_strategy,
    rsi_strategy, 
    volatility_strategy,
    xgboost_strategy
)
from metrics import calculate_max_drawdown, calculate_sortino_ratio
from visualizations import (
    plot_cumulative_bitcoin, 
    plot_max_drawdown, 
    plot_sortino_ratio
)
# Import fee models for exchange selection
from fee_models import load_exchange_profiles, get_optimal_exchange_for_strategy
# Import optimizer page
from optimize_app import run_optimizer_page

def run_selected_strategies(df, strategy_selections, strategy_params, 
                           weekly_investment, exchange_id, use_exchange_discount):
    """
    Pure function to run selected strategies on price data.
    
    Args:
        df (polars.DataFrame): Price data with date, price, day_of_week, is_sunday, returns columns
        strategy_selections (dict): Dictionary of strategy names and boolean selections
        strategy_params (dict): Dictionary of strategy names and their parameters
        weekly_investment (float): Weekly investment amount
        exchange_id (str, optional): Exchange identifier for fee calculation
        use_exchange_discount (bool): Whether to apply exchange discounts
        
    Returns:
        tuple: (results, metrics)
            - results: Dictionary of strategy names and their result DataFrames
            - metrics: Dictionary of strategy names and their performance metrics
    """
    # Check if data is empty
    if df.height == 0:
        return {}, {}
    
    # Initialize results
    results = {}
    
    # Always run DCA as the baseline
    if strategy_selections.get("dca", True):
        results["DCA (Baseline)"] = dca_strategy(
            df.clone(), 
            weekly_investment, 
            exchange_id, 
            use_exchange_discount
        )
    
    # Run Value Averaging if selected
    if strategy_selections.get("value_avg", False):
        target_growth_rate = strategy_params.get("value_avg", {}).get("target_growth_rate", 0.01)
        results["Value Averaging"] = value_averaging_strategy(
            df.clone(), 
            weekly_investment, 
            target_growth_rate
        )
    
    # Run MACO if selected
    if strategy_selections.get("maco", False):
        maco_params = strategy_params.get("maco", {})
        short_window = maco_params.get("short_window", 20)
        long_window = maco_params.get("long_window", 100)
        results["MACO"] = maco_strategy(
            df.clone(), 
            weekly_investment,
            short_window,
            long_window
        )
    
    # Run RSI if selected
    if strategy_selections.get("rsi", False):
        rsi_params = strategy_params.get("rsi", {})
        rsi_period = rsi_params.get("rsi_period", 14)
        oversold_threshold = rsi_params.get("oversold_threshold", 30)
        overbought_threshold = rsi_params.get("overbought_threshold", 70)
        results["RSI"] = rsi_strategy(
            df.clone(), 
            weekly_investment,
            rsi_period,
            oversold_threshold,
            overbought_threshold
        )
    
    # Run Volatility if selected
    if strategy_selections.get("volatility", False):
        vol_params = strategy_params.get("volatility", {})
        vol_window = vol_params.get("vol_window", 14)
        vol_threshold = vol_params.get("vol_threshold", 1.5)
        results["Volatility"] = volatility_strategy(
            df.clone(), 
            weekly_investment,
            vol_window,
            vol_threshold
        )
    
    # Run XGBoost if selected
    if strategy_selections.get("xgboost", False):
        xgb_params = strategy_params.get("xgboost", {})
        prediction_horizon = xgb_params.get("prediction_horizon", 7)
        training_days = xgb_params.get("training_days", 365)
        max_investment_factor = xgb_params.get("max_investment_factor", 2.0)
        min_investment_factor = xgb_params.get("min_investment_factor", 0.5)
        results["XGBoost ML"] = xgboost_strategy(
            df.clone(),
            weekly_investment,
            prediction_horizon,
            training_days,
            max_investment_factor,
            min_investment_factor,
            exchange_id,
            use_exchange_discount
        )
    
    # Calculate metrics for each strategy
    metrics = {}
    for strategy_name, strategy_df in results.items():
        max_drawdown = calculate_max_drawdown(strategy_df)
        sortino = calculate_sortino_ratio(strategy_df)
        final_btc = strategy_df["cumulative_btc"].tail(1).item()
        
        metrics[strategy_name] = {
            "final_btc": final_btc,
            "max_drawdown": max_drawdown,
            "sortino_ratio": sortino
        }
    
    return results, metrics

def run_strategies_with_parameters(df, strategies_with_params):
    """
    Pure function to run strategies with specific parameters.
    
    Args:
        df (polars.DataFrame): Price data frame
        strategies_with_params (dict): Dictionary of strategy names and their parameters
            {
                "Strategy Name": {
                    "strategy": strategy_id,
                    "parameters": {param1: value1, ...}
                }
            }
    
    Returns:
        tuple: (results, metrics) 
            - results: Dictionary of strategy names and their result DataFrames
            - metrics: Dictionary of strategy names and their performance metrics
    """
    if df.height == 0:
        return {}, {}
        
    results = {}
    
    for strategy_name, config in strategies_with_params.items():
        strategy_id = config["strategy"]
        params = config["parameters"]
        
        if strategy_id == "dca":
            weekly_investment = params.get("weekly_investment", 100.0)
            exchange_id = params.get("exchange_id")
            use_discount = params.get("use_discount", False)
            
            results[strategy_name] = dca_strategy(
                df.clone(), 
                weekly_investment, 
                exchange_id, 
                use_discount
            )
            
        elif strategy_id == "value_avg":
            weekly_investment = params.get("weekly_investment", 100.0)
            target_growth_rate = params.get("target_growth_rate", 0.01)
            
            results[strategy_name] = value_averaging_strategy(
                df.clone(), 
                weekly_investment, 
                target_growth_rate
            )
            
        elif strategy_id == "maco":
            weekly_investment = params.get("weekly_investment", 100.0)
            short_window = params.get("short_window", 20)
            long_window = params.get("long_window", 100)
            
            results[strategy_name] = maco_strategy(
                df.clone(), 
                weekly_investment,
                short_window,
                long_window
            )
            
        elif strategy_id == "rsi":
            weekly_investment = params.get("weekly_investment", 100.0)
            rsi_period = params.get("rsi_period", 14)
            oversold_threshold = params.get("oversold_threshold", 30)
            overbought_threshold = params.get("overbought_threshold", 70)
            
            results[strategy_name] = rsi_strategy(
                df.clone(), 
                weekly_investment,
                rsi_period,
                oversold_threshold,
                overbought_threshold
            )
            
        elif strategy_id == "volatility":
            weekly_investment = params.get("weekly_investment", 100.0)
            vol_window = params.get("vol_window", 14)
            vol_threshold = params.get("vol_threshold", 1.5)
            
            results[strategy_name] = volatility_strategy(
                df.clone(), 
                weekly_investment,
                vol_window,
                vol_threshold
            )
        
        elif strategy_id == "xgboost":
            weekly_investment = params.get("weekly_investment", 100.0)
            prediction_horizon = params.get("prediction_horizon", 7)
            training_days = params.get("training_days", 365)
            max_investment_factor = params.get("max_investment_factor", 2.0)
            min_investment_factor = params.get("min_investment_factor", 0.5)
            exchange_id = params.get("exchange_id")
            use_discount = params.get("use_discount", False)
            window_sizes = params.get("window_sizes", [7, 14, 30])
            
            results[strategy_name] = xgboost_strategy(
                df.clone(), 
                weekly_investment,
                prediction_horizon,
                training_days,
                max_investment_factor,
                min_investment_factor,
                exchange_id,
                use_discount,
                window_sizes
            )
    
    # Calculate performance metrics
    metrics = {}
    for strategy_name, strategy_df in results.items():
        max_drawdown = calculate_max_drawdown(strategy_df)
        sortino = calculate_sortino_ratio(strategy_df)
        final_btc = strategy_df["cumulative_btc"].tail(1).item()
        
        metrics[strategy_name] = {
            "final_btc": final_btc,
            "max_drawdown": max_drawdown,
            "sortino_ratio": sortino
        }
    
    return results, metrics

def get_strategy_parameters(strategy_name):
    """
    Pure function to get default parameters for a strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        
    Returns:
        dict: Dictionary of parameters for the strategy
    """
    # Parameter dictionary for each strategy
    if strategy_name == "dca":
        return {
            "weekly_investment": 100.0,
            "exchange_id": None,
            "use_discount": False
        }
    elif strategy_name == "value_avg":
        return {
            "weekly_investment": 100.0,
            "exchange_id": None,
            "target_growth_rate": 0.01  # 1% monthly
        }
    elif strategy_name == "maco":
        return {
            "weekly_investment": 100.0,
            "exchange_id": None,
            "short_window": 20,
            "long_window": 100
        }
    elif strategy_name == "rsi":
        return {
            "weekly_investment": 100.0,
            "exchange_id": None,
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70
        }
    elif strategy_name == "volatility":
        return {
            "weekly_investment": 100.0,
            "exchange_id": None,
            "vol_window": 14,
            "vol_threshold": 1.5
        }
    elif strategy_name == "xgboost":
        return {
            "weekly_investment": 100.0,
            "exchange_id": None,
            "use_discount": False,
            "prediction_horizon": 7,
            "training_days": 365,
            "max_investment_factor": 2.0,
            "min_investment_factor": 0.5,
            "window_sizes": [7, 14, 30]
        }
    else:
        # Return empty dict for unknown strategies
        return {}

def get_optimization_files(period=None, strategies=None, currency="AUD"):
    """
    Get optimization files for a specific time period.
    
    Args:
        period (str, optional): Time period ("1 Year", "5 Years", "10 Years")
        strategies (list, optional): List of strategies to include
        currency (str, optional): Currency code (default: "AUD")
    
    Returns:
        dict: Dictionary of strategy names and their optimization files
    """
    import os
    import glob
    from scripts.generate_optimizations_for_periods import OPTIMIZATION_DIR
    
    # Define period mappings
    period_years = {
        "1 Year": 1,
        "5 Years": 5,
        "10 Years": 10
    }
    
    # Default values
    if strategies is None:
        strategies = ["dca", "maco", "rsi", "volatility"]
    
    # Get today's date
    today = datetime.date.today()
    
    # Build wildcard pattern
    if period:
        years = period_years.get(period)
        if years:
            start_date = today.replace(year=today.year - years)
            start_date_str = start_date.strftime("%d%m%Y")
            end_date_str = today.strftime("%d%m%Y")
            pattern = f"*_{start_date_str}_{end_date_str}_{currency}.arrow"
        else:
            pattern = f"*_{currency}.arrow"
    else:
        pattern = f"*_{currency}.arrow"
    
    # Find all matching files
    search_path = os.path.join(OPTIMIZATION_DIR, pattern)
    optimization_files = glob.glob(search_path)
    
    # Filter by strategies if specified
    results = {}
    for file_path in optimization_files:
        file_name = os.path.basename(file_path)
        strategy = file_name.split('_')[0]
        
        if strategy in strategies:
            if strategy not in results:
                results[strategy] = []
            results[strategy].append(file_path)
    
    return results

# Ensure sample optimization data exists
def ensure_optimization_data_exists():
    """Check if optimization data exists and generate it if necessary"""
    try:
        # Get required constants
        import os
        from scripts.generate_optimizations_for_periods import OPTIMIZATION_DIR, TIME_PERIODS
        
        # Make sure the optimization directory exists
        if not os.path.exists(OPTIMIZATION_DIR):
            os.makedirs(OPTIMIZATION_DIR)
        
        # Define time periods and strategies
        periods = {"1 Year": 1, "5 Years": 5, "10 Years": 10}
        strategies = ["dca", "maco", "rsi", "volatility"]
        currency = "AUD"  # Only use AUD as requested
        
        # Check if we have at least some optimization files for each time period
        missing_files = []
        
        for period_name, years in periods.items():
            # Check for at least one strategy file for this time period
            for strategy in strategies:
                today = datetime.date.today()
                start_date = today.replace(year=today.year - years)
                start_date_str = start_date.strftime("%d%m%Y")
                end_date_str = today.strftime("%d%m%Y")
                
                filename = f"{strategy}_{start_date_str}_{end_date_str}_{currency}.arrow"
                file_path = os.path.join(OPTIMIZATION_DIR, filename)
                
                if not os.path.exists(file_path):
                    missing_files.append((strategy, years))
        
        # If any time period is missing files, run the script to generate all optimization files
        if missing_files:
            st.info("Generating optimization data for all time periods...")
            
            # Import and run the function to generate optimization files for all periods
            from scripts.generate_optimizations_for_periods import main as generate_optimizations
            generate_optimizations()
            
            st.success("Optimization data generated successfully!")
                    
        return True
    except Exception as e:
        st.error(f"Error ensuring optimization data exists: {str(e)}")
        return False

# Set page configuration
st.set_page_config(
    page_title="Bitcoin Strategy Backtester",
    page_icon="📊",
    layout="wide"
)

# Ensure we have optimization data before proceeding
ensure_optimization_data_exists()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Backtest Strategies", "Optimize Strategies"])

if page == "Optimize Strategies":
    # Run the optimizer page
    run_optimizer_page()
else:  # "Backtest Strategies"
    # Title and description for the backtester page
    st.title("Bitcoin Strategy Backtesting Tool")
    st.markdown("""
    This tool allows you to compare different investment strategies for Bitcoin against the baseline
    Dollar Cost Averaging (DCA) strategy. See which approaches might generate alpha over a simple DCA approach.
    """)
    
    # Sidebar for date selection and strategy parameters
    st.sidebar.header("Backtest Parameters")
    st.sidebar.markdown("""
    💾 Historical Bitcoin data is stored locally for faster and more extensive backtesting.
    If data for your selected date range isn't available locally, it will be fetched from CoinGecko.
    """)

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

    # Investment amount - only AUD as per requirements
    investment_currency = "AUD"
    weekly_investment = st.sidebar.number_input(
        f"Weekly Investment ({investment_currency})",
        min_value=10,
        max_value=10000,
        value=100,
        step=10
    )

    # Exchange selection for fee comparison
    st.sidebar.header("Exchange Options")
    # Load available exchanges from fee models
    try:
        exchange_profiles = load_exchange_profiles()
        available_exchanges = list(exchange_profiles.keys())
        
        # Add "None" option to indicate no exchange fees
        available_exchanges = ["None"] + available_exchanges
        
        # Let user select exchange
        selected_exchange = st.sidebar.selectbox(
            "Exchange (for fee calculation)",
            available_exchanges,
            index=0,
            help="Select an exchange to account for transaction fees in strategy performance calculations."
        )
        
        # Option for exchange loyalty discounts
        use_exchange_discount = False
        if selected_exchange != "None":
            use_exchange_discount = st.sidebar.checkbox(
                "Use exchange loyalty discounts",
                value=False,
                help="Apply exchange loyalty token discounts (if available)"
            )
            
            # Show optimal exchange recommendation
            st.sidebar.markdown("##### Optimal Exchange Recommendation")
            if st.sidebar.button("Find optimal exchange"):
                optimal_exchange, est_fee = get_optimal_exchange_for_strategy("dca", weekly_investment, investment_currency)
                st.sidebar.success(f"For DCA strategy with {weekly_investment} {investment_currency}/week:\n\n"
                                f"**Recommended exchange:** {optimal_exchange}\n\n"
                                f"**Estimated annual fee:** {est_fee:.2f} {investment_currency}")
    except Exception as e:
        st.sidebar.warning(f"Could not load exchange profiles: {str(e)}")
        selected_exchange = "None"
        use_exchange_discount = False

    # Strategy selection
    st.sidebar.header("Strategies to Compare")
    use_dca = st.sidebar.checkbox("Dollar Cost Averaging (Baseline)", value=True, disabled=True)

    # Additional strategies
    # Select which strategies to compare
    use_value_avg = st.sidebar.checkbox("Value Averaging", value=False)
    use_maco = st.sidebar.checkbox("Moving Average Crossover", value=False)
    use_rsi = st.sidebar.checkbox("RSI-Based Strategy", value=False)
    use_volatility = st.sidebar.checkbox("Volatility-Based Strategy", value=False)
    use_xgboost = st.sidebar.checkbox("XGBoost ML Strategy", value=False)

    # Strategy parameters (only show if strategy is selected)
    strategy_params = {}

    # Value Averaging parameters
    if use_value_avg:
        st.sidebar.subheader("Value Averaging Parameters")
        strategy_params["value_avg"] = {
            "target_growth_rate": st.sidebar.slider("Target Monthly Growth (%)", 1, 20, 5) / 100
        }

    # MACO parameters
    if use_maco:
        st.sidebar.subheader("Moving Average Crossover Parameters")
        strategy_params["maco"] = {
            "short_window": st.sidebar.slider("Short Window (days)", 5, 50, 20),
            "long_window": st.sidebar.slider("Long Window (days)", 50, 200, 100)
        }

    # RSI parameters
    if use_rsi:
        st.sidebar.subheader("RSI Strategy Parameters")
        strategy_params["rsi"] = {
            "rsi_period": st.sidebar.slider("RSI Period", 7, 21, 14),
            "oversold_threshold": st.sidebar.slider("Oversold Threshold", 20, 40, 30),
            "overbought_threshold": st.sidebar.slider("Overbought Threshold", 60, 80, 70)
        }

    # Volatility parameters
    if use_volatility:
        st.sidebar.subheader("Volatility Strategy Parameters")
        strategy_params["volatility"] = {
            "vol_window": st.sidebar.slider("Volatility Window (days)", 5, 30, 14),
            "vol_threshold": st.sidebar.slider("Volatility Threshold Multiplier", 0.5, 3.0, 1.5, 0.1)
        }
    
    # XGBoost parameters
    if use_xgboost:
        st.sidebar.subheader("XGBoost ML Strategy Parameters")
        strategy_params["xgboost"] = {
            "prediction_horizon": st.sidebar.slider("Prediction Horizon (days)", 1, 14, 7),
            "training_days": st.sidebar.slider("Training Days", 60, 500, 365, 30),
            "max_investment_factor": st.sidebar.slider("Max Investment Factor", 1.0, 3.0, 2.0, 0.1),
            "min_investment_factor": st.sidebar.slider("Min Investment Factor", 0.1, 1.0, 0.5, 0.1)
        }

    # We've removed Lump Sum and Buy the Dip parameters in the refactored version

    # Load data when user clicks the button
    run_button = st.sidebar.button("Run Backtest", type="primary")

    # Information section
    with st.expander("Strategy Descriptions"):
        st.markdown("""
        ### Dollar Cost Averaging (DCA)
        The baseline strategy where you invest a fixed amount regularly (weekly on Sundays) regardless of price.
        
        ### Value Averaging
        Adjusts investment amounts to maintain a predetermined growth path for your portfolio value.
        
        ### Moving Average Crossover (MACO)
        Uses short and long-term moving averages to time entry points. Buys when short-term MA crosses above long-term MA.
        
        ### RSI-Based Strategy
        Uses the Relative Strength Index to buy more when the market is oversold and less when overbought.
        
        ### Volatility-Based Strategy
        Increases investment during periods of high volatility to capture potential upswings.
        """)

    with st.expander("Performance Metrics Explained"):
        st.markdown("""
        ### Cumulative Bitcoin
        The total amount of Bitcoin accumulated through the investment strategy over time.
        
        ### Maximum Drawdown
        The largest percentage decline in Bitcoin holdings from a peak to a subsequent trough.
        Lower values indicate less risk of significant reductions in holdings.
        
        ### Sortino Ratio
        A variation of the Sharpe ratio that only penalizes downside volatility. 
        It measures the risk-adjusted return of an investment, with higher values indicating better 
        performance per unit of downside risk.
        """)

    # Main content area
    if run_button:
        with st.spinner("Fetching Bitcoin price data..."):
            # Format dates for API
            start_date_str = start_date.strftime("%d-%m-%Y")
            end_date_str = end_date.strftime("%d-%m-%Y")
            
            # Fetch data
            try:
                df = fetch_bitcoin_price_data(start_date_str, end_date_str, investment_currency)
                
                if df is None or len(df) == 0:
                    st.error("Failed to fetch data or no data available for the selected date range.")
                else:
                    st.success(f"Successfully fetched {len(df)} days of Bitcoin price data.")
                    
                    # Create a dictionary to store strategy results
                    strategy_results = {}
                    
                    # Get exchange parameters
                    exchange_id = None if selected_exchange == "None" else selected_exchange
                    
                    # Add exchange information to the sidebar if selected
                    if exchange_id:
                        st.sidebar.info(f"Running strategies with **{exchange_id}** exchange fees" + 
                                     (" with loyalty discounts" if use_exchange_discount else ""))
                    
                    # Collect strategy selections
                    strategy_selections = {
                        "dca": True,  # Always run as baseline
                        "value_avg": use_value_avg,
                        "maco": use_maco,
                        "rsi": use_rsi,
                        "volatility": use_volatility,
                        "xgboost": use_xgboost
                    }
                    
                    # Run the strategies using the pure function
                    with st.spinner("Running selected strategies..."):
                        strategy_results, performance_metrics = run_selected_strategies(
                            df, 
                            strategy_selections, 
                            strategy_params, 
                            weekly_investment,
                            exchange_id,
                            use_exchange_discount
                        )
                            
                    # Show exchange information if used
                    if exchange_id:
                        # Add exchange fee info to metrics
                        try:
                            from fee_models import get_exchange_fee, TransactionType
                            fee_percentage = get_exchange_fee(
                                exchange_id, 
                                TransactionType.BUY, 
                                use_discount=use_exchange_discount
                            )
                            st.info(f"Exchange **{exchange_id}** applied with {fee_percentage*100:.2f}% fees" +
                                  (" (including loyalty discounts)" if use_exchange_discount else ""))
                        except Exception as e:
                            st.warning(f"Could not display exchange fee information: {str(e)}")
                    
                    # The Lump Sum and Buy the Dip strategies have been removed in the refactored version
                    
                    # Enhance the metrics with additional information for display
                    for strategy_name, strategy_df in strategy_results.items():
                        if strategy_name in performance_metrics:
                            total_invested = strategy_df["cumulative_investment"].tail(1).item()
                            performance_metrics[strategy_name]["total_invested"] = total_invested
                            
                            # Calculate BTC per currency (efficiency metric)
                            final_btc = performance_metrics[strategy_name]["final_btc"]
                            performance_metrics[strategy_name]["btc_per_currency"] = final_btc / total_invested * weekly_investment
                    
                    # Display summary of results
                    st.header("Performance Summary")
                    
                    metric_cols = st.columns(len(strategy_results))
                    for i, (strategy_name, metrics) in enumerate(performance_metrics.items()):
                        with metric_cols[i]:
                            st.subheader(strategy_name)
                            st.metric(
                                "Final BTC Holdings", 
                                f"{metrics['final_btc']:.8f} BTC"
                            )
                            st.metric(
                                f"BTC per {weekly_investment} {investment_currency}", 
                                f"{metrics['btc_per_currency']:.8f} BTC"
                            )
                            st.metric(
                                "Max Drawdown", 
                                f"{metrics['max_drawdown']*100:.2f}%"
                            )
                            st.metric(
                                "Sortino Ratio", 
                                f"{metrics['sortino_ratio']:.2f}"
                            )
                    
                    # Plot the results
                    st.header("Visualization of Results")
                    
                    # Cumulative Bitcoin
                    st.subheader("Cumulative Bitcoin Holdings")
                    cumulative_btc_fig = plot_cumulative_bitcoin(strategy_results)
                    st.plotly_chart(cumulative_btc_fig, use_container_width=True)
                    
                    # Max Drawdown over time
                    st.subheader("Maximum Drawdown Over Time")
                    max_drawdown_fig = plot_max_drawdown(strategy_results)
                    st.plotly_chart(max_drawdown_fig, use_container_width=True)
                    
                    # Sortino Ratio
                    st.subheader("Sortino Ratio Comparison")
                    sortino_fig = plot_sortino_ratio(performance_metrics)
                    st.plotly_chart(sortino_fig, use_container_width=True)
                    
                    # Display final comparisons against DCA
                    st.header("Strategy Comparison Against DCA Baseline")
                    
                    # Get DCA metrics as baseline
                    dca_metrics = performance_metrics["DCA (Baseline)"]
                    
                    comparison_data = []
                    for strategy_name, metrics in performance_metrics.items():
                        if strategy_name != "DCA (Baseline)":
                            btc_alpha = (metrics["final_btc"] - dca_metrics["final_btc"]) / dca_metrics["final_btc"] * 100
                            drawdown_improvement = (dca_metrics["max_drawdown"] - metrics["max_drawdown"]) * 100
                            sortino_improvement = metrics["sortino_ratio"] - dca_metrics["sortino_ratio"]
                            
                            comparison_data.append({
                                "Strategy": strategy_name,
                                "BTC Alpha (%)": round(btc_alpha, 2),
                                "Drawdown Improvement (pp)": round(drawdown_improvement, 2),
                                "Sortino Improvement": round(sortino_improvement, 2)
                            })
                    
                    if comparison_data:
                        comparison_df = pl.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Highlight the best strategy
                        if len(comparison_data) > 0:
                            # Find the row with the maximum BTC Alpha
                            best_row = comparison_df.filter(
                                pl.col("BTC Alpha (%)") == pl.col("BTC Alpha (%)").max()
                            ).row(0)
                            best_strategy = best_row[0]  # First column is Strategy
                            best_alpha = best_row[1]     # Second column is BTC Alpha (%)
                            
                            if best_alpha > 0:
                                st.success(f"The best performing strategy is **{best_strategy}** with a **{best_alpha}%** BTC alpha over DCA.")
                            else:
                                st.info("None of the tested strategies outperformed the DCA baseline in this period.")
                    else:
                        st.info("No additional strategies were selected for comparison.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    else:
        # Display instructions when app first loads
        st.info("👈 Select your parameters and click 'Run Backtest' to start the analysis.")
        
        # Default explanatory text
        st.markdown("""
        ## How to use this tool
        
        1. Set your backtest parameters in the sidebar
        2. Select which strategies you want to compare
        3. Adjust strategy-specific parameters if needed
        4. Click "Run Backtest" to see the results
        
        The tool will compare different investment strategies against the baseline Dollar Cost Averaging (DCA) approach 
        to see if any can generate alpha (outperformance) in terms of Bitcoin accumulation.
        """)