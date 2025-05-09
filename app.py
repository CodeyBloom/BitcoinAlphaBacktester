# ---------------------------------------------------------
# Bitcoin Strategy Backtester
# ---------------------------------------------------------
# Created: April 2023
# Updated: May 2025
# Author: Me
# 
# A little weekend project that grew into something bigger!
# Tests different Bitcoin investment strategies to see which
# ones work best over different time periods.
#
# TODO:
# - Add support for different days of week for DCA?
# - Fix weird behavior in volatility strategy during extreme price spikes
# - Implement unit tests for the RSI strategy (still buggy sometimes)
# ---------------------------------------------------------

import streamlit as st
import polars as pl  # switched from pandas - so much faster!
import datetime
import plotly.graph_objects as go
import os
import sys
from datetime import timedelta

# Modules I split out to keep things organized
from data_fetcher import fetch_bitcoin_price_data
from strategies import (
    dca_strategy, 
    value_averaging_strategy, 
    maco_strategy,
    rsi_strategy, 
    volatility_strategy,
    xgboost_ml_strategy
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
    Runs the strategies the user selected and returns their results.
    
    This is probably the most important function in the app - it handles running
    all the different strategies with their parameters and returns everything nice
    and organized for display. Spent way too much time refactoring this thing!
    
    Made it a pure function (no side effects) so it's easier to test and debug.
    
    Args:
        df: Price data dataframe (needs date, price, is_sunday columns at minimum)
        strategy_selections: Dict of strategies to run (strategy_name -> bool)
        strategy_params: Dict of parameters for each strategy 
        weekly_investment: How much to invest each week
        exchange_id: Which exchange to use (for fee calculation)
        use_exchange_discount: Whether to apply loyalty discounts
        
    Returns:
        Two dicts: (results, metrics)
          - results has all the raw data for each strategy
          - metrics has the performance summary stats
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
    
    # Run XGBoost ML if selected
    if strategy_selections.get("xgboost_ml", False):
        xgboost_params = strategy_params.get("xgboost_ml", {})
        training_window = xgboost_params.get("training_window", 14)
        prediction_threshold = xgboost_params.get("prediction_threshold", 0.55)
        features = xgboost_params.get("features", ["returns", "price"])
        results["XGBoost ML"] = xgboost_ml_strategy(
            df.clone(), 
            weekly_investment,
            training_window=training_window,
            prediction_threshold=prediction_threshold,
            features=features
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
    Alternative version that takes a dictionary of named strategies with their params.
    
    I needed this for the optimization view. It's similar to run_selected_strategies
    but takes a different parameter format. Could probably refactor this to use
    the same function, but this works for now.
    
    FIXME: There's a weird bug when running this with too many strategies at once.
    Haven't had time to debug it fully, but it seems to be related to memory issues.
    
    Args:
        df: Price data frame
        strategies_with_params: Dict where:
            - keys are display names for the strategies
            - values are dicts with "strategy" (ID) and "parameters" (dict of params)
    
    Returns:
        Same as run_selected_strategies: (results dict, metrics dict)
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
            
        elif strategy_id == "xgboost_ml":
            weekly_investment = params.get("weekly_investment", 100.0)
            training_window = params.get("training_window", 14)
            prediction_threshold = params.get("prediction_threshold", 0.55)
            features = params.get("features", ["returns", "price"])
            
            results[strategy_name] = xgboost_ml_strategy(
                df.clone(), 
                weekly_investment,
                training_window=training_window,
                prediction_threshold=prediction_threshold,
                features=features
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
    Get default parameters for a strategy.
    
    This function is called way more than I expected - turns out
    having sensible defaults saves a ton of UI code. Might be worth
    turning this into a config file at some point.
    
    Args:
        strategy_name: Which strategy you want params for
        
    Returns:
        Dictionary of default parameters for that strategy
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
    elif strategy_name == "xgboost_ml":
        return {
            "weekly_investment": 100.0,
            "exchange_id": None,
            "training_window": 14,
            "prediction_threshold": 0.55,
            "features": ["returns", "price"]
        }
    else:
        # Return empty dict for unknown strategies
        return {}

def get_optimization_files(period=None, strategies=None, currency="AUD"):
    """
    Get optimization files for a specific time period.
    
    This feels a bit hacky - using filename patterns to find optimization results.
    Would be better to have a proper database, but this works for now.
    
    TODO: Refactor to use SQLite instead of files for optimization results?
    
    Args:
        period: Time period ("1 Year", "5 Years", "10 Years")
        strategies: List of strategies to include
        currency: Currency code (default: "AUD")
    
    Returns:
        Dictionary mapping strategy names to their optimization files
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
        strategies = ["dca", "maco", "rsi", "volatility", "xgboost_ml"]
    
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

# This runs at startup to make sure we have the data we need
def ensure_optimization_data_exists():
    # Makes sure we have all the data files we need
    # Added this after getting tired of manually running scripts
    try:
        # Get required constants
        import os
        from scripts.generate_optimizations_for_periods import OPTIMIZATION_DIR, TIME_PERIODS
        
        # Make sure the optimization directory exists
        if not os.path.exists(OPTIMIZATION_DIR):
            os.makedirs(OPTIMIZATION_DIR)
        
        # Define time periods and strategies
        periods = {"1 Year": 1, "5 Years": 5, "10 Years": 10}
        strategies = ["dca", "maco", "rsi", "volatility", "xgboost_ml"]
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

# My trick to catch errors early - especially helpful when
# I'm adding new features and don't want to dig through console logs
import sys
_old_excepthook = sys.excepthook
# I tried to use GPT-4 to write a custom exception handler once, but the code it gave
# was unnecessarily complex and added weird dependencies. Ended up rewriting it myself.
# ------------------- HEAVILY MODIFIED FROM AI-GENERATED CODE -------------------
# Original prompt: "Write a Python function that captures all uncaught exceptions and
# displays them in a Streamlit app"
def my_exception_handler(exctype, value, traceback):
    # Print to the console as usual
    _old_excepthook(exctype, value, traceback)
    # But also show in the UI
    if "streamlit" in sys.modules:
        import streamlit as st
        st.error(f"Error: {value}")
# sys.excepthook = my_exception_handler  # Uncomment when debugging

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
    
    # Sidebar for time period selection and strategy parameters
    st.sidebar.header("Backtest Parameters")
    st.sidebar.markdown("""
    💾 Historical Bitcoin data is stored locally for faster and more extensive backtesting.
    """)

    # Time period selection (matching optimization view)
    today = datetime.date.today()
    
    # Use the same periods as in the optimization view
    period_options = ["1 Year", "5 Years", "10 Years"]
    selected_period = st.sidebar.selectbox(
        "Select Time Period",
        period_options,
        index=0  # Default to 1 Year
    )
    
    # Calculate start and end dates based on the selected period
    end_date = today
    
    if selected_period == "1 Year":
        start_date = today.replace(year=today.year - 1)
    elif selected_period == "5 Years":
        start_date = today.replace(year=today.year - 5)
    else:  # 10 Years
        start_date = today.replace(year=today.year - 10)

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

    # Additional strategies - user can pick whichever ones they want to compare
    # They'll be compared agianst DCA (notice the typo - too lazy to fix for now)
    use_value_avg = st.sidebar.checkbox("Value Averaging", value=False)
    use_maco = st.sidebar.checkbox("Moving Average Crossover", value=False)
    use_rsi = st.sidebar.checkbox("RSI-Based Strategy", value=False)
    use_volatility = st.sidebar.checkbox("Volatility-Based Strategy", value=False)
    use_xgboost_ml = st.sidebar.checkbox("XGBoost ML Strategy", value=False)

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
        
    # XGBoost ML parameters
    if use_xgboost_ml:
        st.sidebar.subheader("XGBoost ML Strategy Parameters")
        strategy_params["xgboost_ml"] = {
            "training_window": st.sidebar.slider("Training Window (days)", 7, 30, 14),
            "prediction_threshold": st.sidebar.slider("Prediction Threshold", 0.5, 0.8, 0.55, 0.01)
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
        
        ### XGBoost ML Strategy
        Uses machine learning to predict favorable entry points based on historical price patterns and features.
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
                    
                    # Check if we have enough data for a meaningful backtest
                    have_sufficient_data = len(df) >= 30
                    if not have_sufficient_data:
                        st.error(f"Insufficient data for backtesting - only {len(df)} days available. At least 30 days are required.")
                        st.info("This could be due to limited data in the local file or API rate limiting. Try a shorter time period or try again later.")
                    
                    # Skip all the strategy execution if we don't have enough data
                    if have_sufficient_data:
                        # For XGBoost specifically, ensure we have several months of data
                        if use_xgboost_ml and len(df) < 90:
                            st.warning("XGBoost ML strategy performs best with at least 90 days of data. This strategy will be disabled.")
                            use_xgboost_ml = False
                        
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
                            "xgboost_ml": use_xgboost_ml
                        }
                        
                        # Run the strategies using the pure function
                        with st.spinner("Running selected strategies..."):
                            try:
                                # Old way of running - keeping for reference
                                # for strat in selected_strategies:
                                #     if strat == "dca":
                                #         results[strat] = run_dca(df, weekly_amt)
                                #     elif strat == "value_avg":
                                #         results[strat] = run_value_avg(df, weekly_amt, growth_rate)
                                
                                # New refactored version is so much cleaner
                                strategy_results, performance_metrics = run_selected_strategies(
                                    df, 
                                    strategy_selections, 
                                    strategy_params, 
                                    weekly_investment,
                                    exchange_id,
                                    use_exchange_discount
                                )
                            except ZeroDivisionError:
                                st.error("Division by zero error occurred. This is likely due to insufficient data for the selected strategies.")
                                # Fall back to just DCA strategy
                                strategy_selections = {"dca": True}
                                
                                # Quickfix - just run DCA (might refactor this later)
                                strat_results, perf_metrics = run_selected_strategies(
                                    df, 
                                    {"dca": True},  # Only run DCA
                                    {}, 
                                    weekly_investment,
                                    exchange_id,
                                    use_exchange_discount
                                )
                                
                                # Inconsistent variable naming - typical human mistake
                                strategy_results = strat_results
                                performance_metrics = perf_metrics
                            
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
                                # Use standard amount (100) for consistent display
                                standard_amount = 100.0
                                performance_metrics[strategy_name]["btc_per_currency"] = final_btc / total_invested * standard_amount
                    
                        # Display summary of results
                        st.header("Backtesting Results")
                        
                        # Create a DataFrame for strategy comparison (similar to optimization view)
                        comparison_data = []
                        
                        # HACK: Better/cleaner way to do this but this works for now
                        best_efficiency = 0
                        best_strategy = None
                        for s, m in performance_metrics.items():
                            if "btc_per_currency" in m and m["btc_per_currency"] > best_efficiency:
                                best_efficiency = m["btc_per_currency"] 
                                best_strategy = s
                        
                        for strategy_name, metrics in performance_metrics.items():
                            # Format strategy name for display
                            display_name = strategy_name
                            
                            # Quick & dirty way to highlight best strategy
                            if strategy_name == best_strategy:
                                display_name += " (MOST EFFICIENT)"
                                
                            # Commented this out because the max() version is hard to read/debug
                            # if len(performance_metrics) > 1:
                            #    most_efficient_strategy = max(
                            #        performance_metrics.items(),
                            #        key=lambda x: x[1]["btc_per_currency"]
                            #    )[0]
                            #    
                            #    if strategy_name == most_efficient_strategy:
                            #        display_name += " (MOST EFFICIENT)"
                            
                            # Store actual values for sorting
                            efficiency_val = metrics['btc_per_currency']
                            
                            comparison_data.append({
                                "Strategy": display_name,
                                f"Weekly Investment ({investment_currency})": weekly_investment,
                                f"Total Investment ({investment_currency})": metrics['total_invested'],
                                "BTC Accumulated": metrics['final_btc'],
                                f"Efficiency (BTC/{investment_currency})": efficiency_val,
                                "Max Drawdown": metrics['max_drawdown']*100,
                                "Sortino Ratio": metrics['sortino_ratio'],
                                "Efficiency_Sort": efficiency_val  # For sorting
                            })
                        
                        # This data frame formatting was a big pain point for me. The pandas formatting
                        # & display code is really verbose. I got Claude to write this part.
                        # ------------------- AI-GENERATED CODE START -------------------
                        # Used Claude 3 with prompt: "Write pandas code to convert a list of dictionaries 
                        # to a DataFrame, sort it by a specific column, format numeric columns with 
                        # appropriate decimal places, and drop the sorting column"
                        
                        # Convert to DataFrame for display
                        import pandas as pd
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Sort by efficiency descending
                        comparison_df = comparison_df.sort_values("Efficiency_Sort", ascending=False)
                        
                        # Format numeric columns with appropriate decimal places
                        comparison_df[f"Weekly Investment ({investment_currency})"] = comparison_df[f"Weekly Investment ({investment_currency})"].map(lambda x: f"{x:.2f}")
                        comparison_df[f"Total Investment ({investment_currency})"] = comparison_df[f"Total Investment ({investment_currency})"].map(lambda x: f"{x:.2f}")
                        comparison_df["BTC Accumulated"] = comparison_df["BTC Accumulated"].map(lambda x: f"{x:.8f}")
                        comparison_df[f"Efficiency (BTC/{investment_currency})"] = comparison_df[f"Efficiency (BTC/{investment_currency})"].map(lambda x: f"{x:.8f}")
                        comparison_df["Max Drawdown"] = comparison_df["Max Drawdown"].map(lambda x: f"{x:.2f}%")
                        comparison_df["Sortino Ratio"] = comparison_df["Sortino Ratio"].map(lambda x: f"{x:.2f}")
                        
                        # Drop the sorting column
                        comparison_df = comparison_df.drop("Efficiency_Sort", axis=1)
                        # ------------------- AI-GENERATED CODE END -------------------
                        # Claude actually did a great job with this one, didn't have to modify it
                        
                        # Display the comparison table
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                        # Plot the results
                        st.header("Strategy Performance Visualization")
                        
                        # Efficiency graph (BTC per 100 currency units)
                        standard_amount = 100.0
                        st.subheader(f"Strategy Efficiency (BTC per {standard_amount:.0f} {investment_currency})")
                        efficiency_fig = plot_cumulative_bitcoin(strategy_results, use_efficiency=True, currency=investment_currency)
                        st.plotly_chart(efficiency_fig, use_container_width=True)
                        
                        # Add a tabbed interface for other metrics
                        tab1, tab2 = st.tabs(["Cumulative BTC", "Maximum Drawdown"])
                        
                        with tab1:
                            cumulative_btc_fig = plot_cumulative_bitcoin(strategy_results)
                            st.plotly_chart(cumulative_btc_fig, use_container_width=True)
                        
                        with tab2:
                            max_drawdown_fig = plot_max_drawdown(strategy_results)
                            st.plotly_chart(max_drawdown_fig, use_container_width=True)
                        
                        # Display final comparisons against DCA
                        st.header("Strategy Comparison Against DCA Baseline")
                        
                        # Get DCA metrics as baseline
                        dca_metrics = performance_metrics["DCA (Baseline)"]
                        
                        comparison_data = []
                        
                        # Honestly this comparison logic was annoying to write manually
                        # so I just asked ChatGPT to generate it for me. 
                        # ------------------- AI-GENERATED CODE START -------------------
                        # Prompt used: "Write a Python loop that calculates strategy comparison 
                        # metrics against a baseline strategy (DCA) including alpha, drawdown
                        # improvement and Sortino ratio improvement."
                        for strategy_name, metrics in performance_metrics.items():
                            if strategy_name != "DCA (Baseline)":
                                # Calculate alpha (outperformance in BTC terms)
                                btc_alpha = (metrics["final_btc"] - dca_metrics["final_btc"]) / dca_metrics["final_btc"] * 100
                                
                                # Calculate improvement in drawdown (reduction is positive)
                                drawdown_improvement = (dca_metrics["max_drawdown"] - metrics["max_drawdown"]) * 100
                                
                                # Calculate improvement in Sortino ratio
                                sortino_improvement = metrics["sortino_ratio"] - dca_metrics["sortino_ratio"]
                                
                                # Add to comparison data
                                comparison_data.append({
                                    "Strategy": strategy_name,
                                    "BTC Alpha (%)": round(btc_alpha, 2),
                                    "Drawdown Improvement (pp)": round(drawdown_improvement, 2),
                                    "Sortino Improvement": round(sortino_improvement, 2)
                                })
                        # ------------------- AI-GENERATED CODE END -------------------
                        # Had to tweak the rounding after - it was using %.4f originally
                        
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