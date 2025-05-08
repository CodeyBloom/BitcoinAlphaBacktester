import streamlit as st
import polars as pl
import datetime
import plotly.graph_objects as go
from datetime import timedelta

# Import from the refactored modules
from data_fetcher import fetch_bitcoin_price_data
from strategies import (
    dca_strategy, 
    value_averaging_strategy, 
    maco_strategy,
    rsi_strategy, 
    volatility_strategy
)
from metrics import calculate_max_drawdown, calculate_sortino_ratio
from visualizations import (
    plot_cumulative_bitcoin, 
    plot_max_drawdown, 
    plot_sortino_ratio
)
# Import fee models for exchange selection
from fee_models import load_exchange_profiles, get_optimal_exchange_for_strategy

# Set page configuration
st.set_page_config(
    page_title="Bitcoin Strategy Backtester",
    page_icon="📊",
    layout="wide"
)

# Title and description
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

# Investment amount
investment_currency = st.sidebar.selectbox("Investment Currency", ["AUD", "USD"])
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
                
                # Run baseline DCA strategy
                with st.spinner("Running DCA strategy..."):
                    dca_result = dca_strategy(df.clone(), weekly_investment, exchange_id, use_exchange_discount)
                    strategy_results["DCA (Baseline)"] = dca_result
                
                # Run Value Averaging if selected
                if use_value_avg:
                    with st.spinner("Running Value Averaging strategy..."):
                        va_result = value_averaging_strategy(
                            df.clone(), 
                            weekly_investment, 
                            strategy_params["value_avg"]["target_growth_rate"]
                        )
                        strategy_results["Value Averaging"] = va_result
                
                # Run MACO if selected
                if use_maco:
                    with st.spinner("Running Moving Average Crossover strategy..."):
                        maco_result = maco_strategy(
                            df.clone(), 
                            weekly_investment,
                            strategy_params["maco"]["short_window"],
                            strategy_params["maco"]["long_window"]
                        )
                        strategy_results["MACO"] = maco_result
                
                # Run RSI if selected
                if use_rsi:
                    with st.spinner("Running RSI-based strategy..."):
                        rsi_result = rsi_strategy(
                            df.clone(), 
                            weekly_investment,
                            strategy_params["rsi"]["rsi_period"],
                            strategy_params["rsi"]["oversold_threshold"],
                            strategy_params["rsi"]["overbought_threshold"]
                        )
                        strategy_results["RSI"] = rsi_result
                
                # Run Volatility if selected
                if use_volatility:
                    with st.spinner("Running Volatility-based strategy..."):
                        vol_result = volatility_strategy(
                            df.clone(), 
                            weekly_investment,
                            strategy_params["volatility"]["vol_window"],
                            strategy_params["volatility"]["vol_threshold"]
                        )
                        strategy_results["Volatility"] = vol_result
                        
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
                
                # Calculate performance metrics
                performance_metrics = {}
                
                for strategy_name, strategy_df in strategy_results.items():
                    max_drawdown = calculate_max_drawdown(strategy_df)
                    sortino = calculate_sortino_ratio(strategy_df)
                    final_btc = strategy_df["cumulative_btc"].tail(1).item()
                    total_invested = strategy_df["cumulative_investment"].tail(1).item()
                    
                    performance_metrics[strategy_name] = {
                        "final_btc": final_btc,
                        "total_invested": total_invested,
                        "max_drawdown": max_drawdown,
                        "sortino_ratio": sortino,
                        "btc_per_currency": final_btc / total_invested * weekly_investment
                    }
                
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
