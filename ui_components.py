"""
UI components for the Bitcoin Strategy Backtester.

This module provides reusable UI components for the Bitcoin Strategy Backtester
application. These components follow a consistent design pattern and make the
main application code more modular and testable.

Components include:
- Strategy cards for displaying strategy information
- Efficiency charts for visualizing strategy performance
- Metrics dashboards for overview information
- Strategy comparison views
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import polars as pl


def strategy_card(strategy_name, efficiency, params, is_most_efficient=False, currency="AUD"):
    """
    Display strategy information in a card format.
    
    Args:
        strategy_name (str): Name of the strategy
        efficiency (float): Efficiency value (BTC per currency unit)
        params (dict): Strategy parameters
        is_most_efficient (bool): Whether this is the most efficient strategy
        currency (str): Currency code
        
    Returns:
        st.container: The container with the strategy card
    """
    with st.container():
        # Create a card-like container with border styling
        if is_most_efficient:
            st.success(f"### {strategy_name.upper()}")
            st.success("**MOST EFFICIENT STRATEGY**")
        else:
            st.info(f"### {strategy_name.upper()}")
        
        # Display key metrics
        st.info(f"**Efficiency:** {efficiency:.8f} BTC/{currency}")
        
        # Display parameters in a clean format
        st.markdown("**Parameters:**")
        for param, value in params.items():
            # Format parameter names for better display
            param_display = param.replace("_", " ").title()
            
            # Special formatting for specific parameters
            if param == "weekly_investment":
                st.info(f"**{param_display}:** {value:.2f} {currency}")
            elif param == "exchange_id":
                st.info(f"**Exchange:** {value}")
            elif param == "use_discount":
                st.info(f"**Use Loyalty Discount:** {'Yes' if value else 'No'}")
            elif isinstance(value, float):
                st.info(f"**{param_display}:** {value:.4f}")
            else:
                st.info(f"**{param_display}:** {value}")
        
        return st.container()


def efficiency_chart(results, most_efficient, currency="AUD"):
    """
    Display an interactive efficiency chart with enhanced features.
    
    Args:
        results (dict): Dictionary of strategy results
        most_efficient (str): Name of the most efficient strategy
        currency (str): Currency code
        
    Returns:
        go.Figure: The Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add each strategy as a trace
    for strategy_name, result in results.items():
        dates = result.get("dates", [])
        efficiency_curve = result.get("efficiency_curve", [])
        
        # Format the strategy name for display
        display_name = strategy_name.upper()
        if strategy_name == most_efficient:
            display_name += " (MOST EFFICIENT)"
        
        # Custom line styles and width for better visibility
        line_width = 3 if strategy_name == most_efficient else 2
        dash_style = None if strategy_name == most_efficient else "dash"
        
        # Enhanced hover template with more information
        hover_template = (
            f'<b>{display_name}</b><br>' +
            f'Date: %{{x|%d %b %Y}}<br>' +
            f'Efficiency: %{{y:.8f}} BTC/{currency}<br>' +
            f'<extra></extra>'
        )
        
        # Add line to plot with enhanced styling
        fig.add_trace(go.Scatter(
            x=dates, 
            y=efficiency_curve,
            mode='lines',
            name=display_name,
            line=dict(width=line_width, dash=dash_style),
            hovertemplate=hover_template
        ))
    
    # Add reference line for buy-and-hold baseline if available
    if "buy_and_hold" in results:
        bh_result = results["buy_and_hold"]
        bh_dates = bh_result.get("dates", [])
        bh_efficiency = bh_result.get("efficiency_curve", [])
        
        fig.add_trace(go.Scatter(
            x=bh_dates,
            y=bh_efficiency,
            mode='lines',
            name='BUY & HOLD BASELINE',
            line=dict(color='gray', width=1.5, dash='dot'),
            hovertemplate='Buy & Hold: %{y:.8f} BTC/' + currency + '<br>%{x|%d %b %Y}<extra></extra>'
        ))
    
    # Enhanced layout with better styling
    fig.update_layout(
        title_text='',  # Empty string instead of None
        xaxis_title="Date",
        yaxis_title=f"Efficiency (BTC/{currency})",
        legend_title="Strategies",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=10, l=10, r=10, b=10)
    )
    
    # Add range selector for time periods
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Show the figure
    st.plotly_chart(fig, use_container_width=True)
    
    return fig


def metrics_dashboard(results, most_efficient, currency="AUD"):
    """
    Display a dashboard of key performance metrics for all strategies.
    
    Args:
        results (dict): Dictionary of strategy results
        most_efficient (str): Name of the most efficient strategy
        currency (str): Currency code
        
    Returns:
        st.container: The container with the metrics dashboard
    """
    with st.container():
        st.markdown("## Strategy Performance Overview")
        
        # Create a summary dataframe for comparison
        summary_data = []
        for strategy_name, result in results.items():
            performance = result.get("performance", {})
            params = result.get("best_params", {})
            
            final_btc = performance.get("final_btc", 0)
            total_invested = performance.get("total_invested", 0)
            efficiency = final_btc / total_invested if total_invested > 0 else 0
            
            summary_data.append({
                "Strategy": strategy_name.upper() + (" (MOST EFFICIENT)" if strategy_name == most_efficient else ""),
                f"Efficiency (BTC/{currency})": efficiency,
                "Final BTC": final_btc,
                f"Total Invested ({currency})": total_invested,
                "Max Drawdown (%)": performance.get("max_drawdown", 0) * 100,
                "Sortino Ratio": performance.get("sortino_ratio", 0)
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Sort by efficiency (descending)
            summary_df = summary_df.sort_values(f"Efficiency (BTC/{currency})", ascending=False)
            
            # Format the numeric columns
            summary_df[f"Efficiency (BTC/{currency})"] = summary_df[f"Efficiency (BTC/{currency})"].map("{:.8f}".format)
            summary_df["Final BTC"] = summary_df["Final BTC"].map("{:.8f}".format)
            summary_df[f"Total Invested ({currency})"] = summary_df[f"Total Invested ({currency})"].map("{:.2f}".format)
            summary_df["Max Drawdown (%)"] = summary_df["Max Drawdown (%)"].map("{:.2f}".format)
            summary_df["Sortino Ratio"] = summary_df["Sortino Ratio"].map("{:.2f}".format)
            
            # Display the summary dataframe with styling
            st.dataframe(summary_df, use_container_width=True)
            
            # Create metrics cards for top strategies
            st.markdown("### Top Strategy Metrics")
            
            # Display up to 3 top strategies in metrics cards
            cols = st.columns(min(3, len(summary_data)))
            for i, (col, strategy_data) in enumerate(zip(cols, summary_data[:3])):
                with col:
                    strategy_name = strategy_data["Strategy"]
                    is_top = "(MOST EFFICIENT)" in strategy_name
                    
                    if is_top:
                        st.success(f"**{strategy_name.replace(' (MOST EFFICIENT)', '')}**")
                        st.success("🏆 MOST EFFICIENT")
                    else:
                        st.info(f"**{strategy_name}**")
                        st.info(f"🥇 Rank #{i+1}")
                    
                    st.metric(
                        f"Efficiency (BTC/{currency})", 
                        strategy_data[f"Efficiency (BTC/{currency})"]
                    )
                    
                    st.metric(
                        "Max Drawdown (%)", 
                        strategy_data["Max Drawdown (%)"]
                    )
        
        return st.container()


def strategy_comparison(strategy1, strategy2, results, currency="AUD"):
    """
    Display a side-by-side comparison of two strategies.
    
    Args:
        strategy1 (str): Name of the first strategy
        strategy2 (str): Name of the second strategy
        results (dict): Dictionary of strategy results
        currency (str): Currency code
        
    Returns:
        st.container: The container with the comparison
    """
    with st.container():
        st.markdown(f"## Strategy Comparison")
        st.markdown(f"### {strategy1.upper()} vs {strategy2.upper()}")
        
        # Get data for both strategies
        strategy1_data = results.get(strategy1, {})
        strategy2_data = results.get(strategy2, {})
        
        strategy1_performance = strategy1_data.get("performance", {})
        strategy2_performance = strategy2_data.get("performance", {})
        
        strategy1_params = strategy1_data.get("best_params", {})
        strategy2_params = strategy2_data.get("best_params", {})
        
        # Calculate efficiency
        strategy1_final_btc = strategy1_performance.get("final_btc", 0)
        strategy1_invested = strategy1_performance.get("total_invested", 0)
        strategy1_efficiency = strategy1_final_btc / strategy1_invested if strategy1_invested > 0 else 0
        
        strategy2_final_btc = strategy2_performance.get("final_btc", 0)
        strategy2_invested = strategy2_performance.get("total_invested", 0)
        strategy2_efficiency = strategy2_final_btc / strategy2_invested if strategy2_invested > 0 else 0
        
        # Calculate percentage difference
        efficiency_diff = ((strategy1_efficiency / strategy2_efficiency) - 1) * 100 if strategy2_efficiency > 0 else 0
        
        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {strategy1.upper()}")
            st.metric(
                f"Efficiency (BTC/{currency})", 
                f"{strategy1_efficiency:.8f}",
                f"{efficiency_diff:+.2f}%" if strategy2_efficiency > 0 else None
            )
            st.metric("Final BTC", f"{strategy1_final_btc:.8f}")
            st.metric(f"Total Invested ({currency})", f"{strategy1_invested:.2f}")
            
            if "max_drawdown" in strategy1_performance:
                st.metric("Max Drawdown (%)", f"{strategy1_performance['max_drawdown']*100:.2f}%")
            
            if "sortino_ratio" in strategy1_performance:
                st.metric("Sortino Ratio", f"{strategy1_performance['sortino_ratio']:.2f}")
                
        with col2:
            st.markdown(f"### {strategy2.upper()}")
            st.metric(
                f"Efficiency (BTC/{currency})", 
                f"{strategy2_efficiency:.8f}",
                f"{-efficiency_diff:+.2f}%" if strategy1_efficiency > 0 else None
            )
            st.metric("Final BTC", f"{strategy2_final_btc:.8f}")
            st.metric(f"Total Invested ({currency})", f"{strategy2_invested:.2f}")
            
            if "max_drawdown" in strategy2_performance:
                st.metric("Max Drawdown (%)", f"{strategy2_performance['max_drawdown']*100:.2f}%")
            
            if "sortino_ratio" in strategy2_performance:
                st.metric("Sortino Ratio", f"{strategy2_performance['sortino_ratio']:.2f}")
        
        # Compare parameters
        st.markdown("### Parameter Comparison")
        
        # Find all unique parameters
        all_params = set(strategy1_params.keys()) | set(strategy2_params.keys())
        
        # Create comparison table
        param_data = []
        for param in all_params:
            param_display = param.replace("_", " ").title()
            param_data.append({
                "Parameter": param_display,
                f"{strategy1.upper()}": strategy1_params.get(param, "N/A"),
                f"{strategy2.upper()}": strategy2_params.get(param, "N/A")
            })
        
        param_df = pd.DataFrame(param_data)
        st.dataframe(param_df, use_container_width=True)
        
        return st.container()


def implementation_guide(strategy_name, params, currency="AUD"):
    """
    Display an implementation guide for a specific strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        params (dict): Strategy parameters
        currency (str): Currency code
        
    Returns:
        st.container: The container with the implementation guide
    """
    with st.container():
        st.markdown("## Implementation Guide")
        
        # Different tutorial based on strategy
        if strategy_name == "dca":
            st.markdown("""
            ### Dollar Cost Averaging Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Schedule Regular Deposits**: Deposit {investment:.2f} {currency} each week.
            3. **Automate Purchases**: Set up automatic purchases on {day} each week.
            4. **Enable Loyalty Discount**: {discount_instructions}.
            5. **Secure Your Bitcoin**: Consider transferring to a hardware wallet monthly.
            
            **Tip**: Many exchanges let you schedule automatic purchases, making this strategy extremely simple to implement.
            """.format(
                exchange=params.get("exchange_id", "Binance").capitalize(),
                investment=params.get("weekly_investment", 0),
                currency=currency,
                day=params.get("day_of_week", "Sunday"),
                discount_instructions="Enable the exchange's token discount program" if params.get("use_discount", False) else "No discount required for this strategy"
            ))
        
        elif strategy_name == "maco":
            st.markdown("""
            ### Moving Average Crossover Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Allocate Investment Fund**: Set aside {investment:.2f} {currency} per week.
            3. **Track Moving Averages**: Monitor {short_window}-day and {long_window}-day moving averages.
            4. **Place Buy Orders**: Buy Bitcoin when short-term average crosses above long-term average.
            5. **Adjust Investment Size**: Invest more when crossing signal is strong (>1% difference).
            
            **Tip**: Use a cryptocurrency portfolio app or spreadsheet with auto-calculation to track the moving averages easily.
            """.format(
                exchange=params.get("exchange_id", "Binance").capitalize(),
                investment=params.get("weekly_investment", 0),
                currency=currency,
                short_window=params.get("short_window", 20),
                long_window=params.get("long_window", 50)
            ))
        
        elif strategy_name == "rsi":
            st.markdown("""
            ### RSI Strategy Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Allocate Base Investment**: Reserve {investment:.2f} {currency} per week.
            3. **Calculate RSI**: Use a {period}-day period to calculate the Relative Strength Index.
            4. **Increase Investment**: When RSI is below {oversold}, increase investment up to {max_factor}x.
            5. **Decrease Investment**: When RSI is above {overbought}, reduce investment to as low as {min_factor}x.
            
            **Tip**: Many cryptocurrency price tracking websites display RSI indicators for free.
            """.format(
                exchange=params.get("exchange_id", "Binance").capitalize(),
                investment=params.get("weekly_investment", 0),
                currency=currency,
                period=params.get("rsi_period", 14),
                oversold=params.get("oversold_threshold", 30),
                overbought=params.get("overbought_threshold", 70),
                max_factor=params.get("max_increase_factor", "2.5"),
                min_factor=params.get("min_decrease_factor", "0.5")
            ))
        
        elif strategy_name == "volatility":
            st.markdown("""
            ### Volatility Strategy Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Allocate Base Investment**: Reserve {investment:.2f} {currency} per week.
            3. **Track Volatility**: Monitor {window}-day price volatility.
            4. **Compare to Average**: Compare current volatility to the {lookback}-day average.
            5. **Adjust Investment**: When volatility is {threshold}x average, increase investment up to {max_factor}x.
            
            **Tip**: Set price alerts for sudden volatility increases to time your increased investments.
            """.format(
                exchange=params.get("exchange_id", "Binance").capitalize(),
                investment=params.get("weekly_investment", 0),
                currency=currency,
                window=params.get("vol_window", 14),
                threshold=params.get("vol_threshold", 1.5),
                lookback=params.get("lookback_period", "90"),
                max_factor=params.get("max_increase_factor", "3.0")
            ))
        
        elif strategy_name == "xgboost_ml":
            st.markdown("""
            ### XGBoost ML Strategy Implementation:
            1. **Set Up Exchange Account**: Create an account on {exchange}.
            2. **Allocate Base Investment**: Reserve {investment:.2f} {currency} per week.
            3. **Gather Historical Data**: Collect {features} data points for the ML model.
            4. **Train ML Model**: Use a {training_window}-day window to train XGBoost model.
            5. **Make Investment Decisions**: Invest when prediction confidence exceeds {threshold}.
            
            **Tip**: Retraining your model periodically (e.g., monthly) can help it adapt to changing market conditions.
            """.format(
                exchange=params.get("exchange_id", "Binance").capitalize(),
                investment=params.get("weekly_investment", 0),
                currency=currency,
                features=params.get("feature_set", "price,returns").replace(",", ", "),
                training_window=params.get("training_window", 30),
                threshold=params.get("prediction_threshold", 0.55)
            ))
        
        # Add download button (simulated for now)
        if st.button(f"Download {strategy_name.upper()} Strategy Guide"):
            st.info("Strategy guide download feature coming soon!")
        
        return st.container()