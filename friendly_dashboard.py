"""
Friendly dashboard for Bitcoin Strategy Backtester.

This module provides a simplified, easy-to-understand presentation of 
backtest results for non-experts.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime

def run_friendly_dashboard(strategy_results, performance_metrics, investment_currency="AUD"):
    """
    Display a friendly dashboard of backtest results.
    
    Args:
        strategy_results (dict): Dictionary of strategy results DataFrames
        performance_metrics (dict): Dictionary of strategy performance metrics
        investment_currency (str): Currency code
    """
    if not strategy_results or not performance_metrics:
        st.info("Run a backtest to see results.")
        return
    
    # Get the DCA (baseline) results for reference
    dca_df = strategy_results.get("DCA (Baseline)")
    dca_metrics = performance_metrics.get("DCA (Baseline)")
    
    if not dca_df or not dca_metrics:
        st.error("DCA (Baseline) results not found.")
        return
    
    # Title and intro
    st.title("🔍 Your Bitcoin Strategy Results")
    st.markdown("""
    Here's a simple breakdown of how each strategy performed in plain language.
    """)
    
    # Find the best strategy based on BTC per currency
    best_strategy = "DCA (Baseline)"
    best_efficiency = dca_metrics.get("btc_per_currency", 0)
    
    for name, metrics in performance_metrics.items():
        if name != "DCA (Baseline)" and metrics.get("btc_per_currency", 0) > best_efficiency:
            best_strategy = name
            best_efficiency = metrics.get("btc_per_currency", 0)
    
    # Create a summary banner for the best strategy
    if best_strategy != "DCA (Baseline)":
        improvement = (best_efficiency / dca_metrics.get("btc_per_currency", 1) - 1) * 100
        st.success(f"### Best strategy: {best_strategy} (got you {improvement:.1f}% more Bitcoin)")
    else:
        st.info("### Regular weekly buying (DCA) worked best in this period")
    
    # Show "What's The Story?" timeline
    st.header("📊 What Happened")
    
    # Create a price chart with strategy buy points
    fig = create_simple_timeline(strategy_results, performance_metrics, best_strategy)
    st.plotly_chart(fig, use_container_width=True)
    
    # Create "Worth the Effort?" card
    st.header("⚖️ Was It Worth the Effort?")
    
    # Create columns for comparison cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("How strategies compare to simple weekly buying")
        
        # Create a simple comparison table
        comparison_data = []
        for name, metrics in performance_metrics.items():
            if name != "DCA (Baseline)":
                btc_ratio = metrics.get("btc_per_currency", 0) / dca_metrics.get("btc_per_currency", 1)
                extra_percent = (btc_ratio - 1) * 100
                
                if extra_percent > 0:
                    worth_it = "✅" if extra_percent > 5 else "⚖️"
                    assessment = "Much better" if extra_percent > 10 else "Slightly better"
                else:
                    worth_it = "❌"
                    assessment = "Worse"
                
                comparison_data.append({
                    "Strategy": name,
                    "Compared to DCA": f"{assessment} ({extra_percent:+.1f}%)",
                    "Worth it?": worth_it
                })
        
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), hide_index=True)
        else:
            st.info("No other strategies were selected for comparison")
    
    with col2:
        st.subheader("Sleep Well Factor")
        
        # Create a sleep well factor visualization
        for name, metrics in performance_metrics.items():
            max_drawdown = metrics.get("max_drawdown", 0)
            
            # Calculate a simple 1-5 sleep well score
            # Lower drawdown = better sleep
            sleep_score = 5 - max_drawdown * 10  # 20% drawdown = 3, 40% = 1
            sleep_score = max(1, min(5, sleep_score))  # Clamp between 1-5
            
            # Display as a gauge
            st.write(f"**{name}**: {'😴' * int(round(sleep_score))}")
            
            # Add a simple explanation
            if max_drawdown < 0.1:
                st.write("Very smooth ride")
            elif max_drawdown < 0.2:
                st.write("Some bumps, but mostly smooth")
            elif max_drawdown < 0.3:
                st.write("Moderate ups and downs")
            else:
                st.write("Bumpy ride with big drops")
            
            # Add divider between strategies
            st.markdown("---")
    
    # "Bottom Line" section
    st.header("💡 The Bottom Line")
    
    # Create a more concrete example of how much could be made
    example_weekly_amount = 100
    example_years = (dca_df["date"].max() - dca_df["date"].min()).days / 365
    
    # Calculate total Bitcoin for strategies
    total_investment = example_weekly_amount * 52 * example_years
    
    results_data = []
    for name, metrics in performance_metrics.items():
        btc_per_currency = metrics.get("btc_per_currency", 0)
        total_btc = btc_per_currency * total_investment / example_weekly_amount
        
        # Get final Bitcoin price
        final_price = dca_df["price"].tail(1).item()
        
        # Calculate value
        final_value = total_btc * final_price
        
        results_data.append({
            "Strategy": name,
            f"If you invested ${example_weekly_amount}/week for {example_years:.1f} years": f"{total_btc:.8f} BTC (${final_value:,.2f})"
        })
    
    st.dataframe(pd.DataFrame(results_data), hide_index=True)
    
    # Create a "Try It Yourself" section
    st.header("🚀 How to Use This Strategy")
    
    # Show implementation steps for the best strategy
    if best_strategy in strategy_results:
        display_implementation_steps(best_strategy, strategy_results[best_strategy])
    
    # Disclaimer
    st.caption("""
    **Disclaimer**: Past performance is not indicative of future results. This tool provides educational 
    information, not financial advice. Always do your own research before investing.
    """)

def create_simple_timeline(strategy_results, performance_metrics, best_strategy="DCA (Baseline)"):
    """
    Create a simple timeline visualization of strategy performance.
    
    Args:
        strategy_results (dict): Dictionary of strategy results DataFrames
        performance_metrics (dict): Dictionary of strategy performance metrics
        best_strategy (str): Name of the best performing strategy
        
    Returns:
        plotly.graph_objects.Figure: Timeline visualization
    """
    # Get price data from any strategy
    first_strategy = next(iter(strategy_results.values()))
    dates = first_strategy["date"].to_numpy()
    prices = first_strategy["price"].to_numpy()
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            name="Bitcoin Price",
            line=dict(color="#2C3E50", width=1.5),
            hovertemplate="$%{y:,.2f}<br>%{x|%d %b %Y}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add investment points for each strategy
    colors = {
        "DCA (Baseline)": "#3498db",
        "Value Averaging": "#e74c3c",
        "MACO": "#2ecc71",
        "RSI": "#f39c12", 
        "Volatility": "#9b59b6",
        "XGBoost ML": "#1abc9c"
    }
    
    for strategy_name, df in strategy_results.items():
        # Skip if this is price data
        if strategy_name == "Price Data":
            continue
        
        # Get strategy data
        strategy_dates = df["date"].to_numpy()
        investments = df["investment"].to_numpy()
        cumulative_btc = df["cumulative_btc"].to_numpy()
        
        # Only plot points where investment was made
        buy_mask = investments > 0
        buy_dates = strategy_dates[buy_mask]
        buy_prices = prices[buy_mask]
        buy_sizes = investments[buy_mask] / investments[buy_mask].max() * 15 + 5
        
        # Add buy points
        if len(buy_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    name=f"{strategy_name} Buys",
                    marker=dict(
                        color=colors.get(strategy_name, "#7f8c8d"),
                        size=buy_sizes,
                        line=dict(width=1, color="white")
                    ),
                    hovertemplate="Buy: $%{marker.size:.2f}<br>%{x|%d %b %Y}<extra></extra>"
                ),
                secondary_y=False
            )
        
        # Add cumulative BTC line
        fig.add_trace(
            go.Scatter(
                x=strategy_dates,
                y=cumulative_btc,
                name=f"{strategy_name} BTC",
                line=dict(
                    color=colors.get(strategy_name, "#7f8c8d"),
                    width=2,
                    dash="dot" if strategy_name != best_strategy else None
                ),
                hovertemplate="%{y:.8f} BTC<br>%{x|%d %b %Y}<extra></extra>"
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title="Bitcoin Price and Strategy Purchases",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Bitcoin Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative BTC", secondary_y=True)
    
    return fig

def display_implementation_steps(strategy_name, strategy_df):
    """
    Display simple implementation steps for a strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        strategy_df (polars.DataFrame): Strategy results DataFrame
    """
    # Analyze the strategy behavior to inform implementation steps
    investments = strategy_df["investment"].to_numpy()
    non_zero_investments = investments[investments > 0]
    
    if len(non_zero_investments) == 0:
        st.info("This strategy made no investments in the period.")
        return
    
    # Calculate investment patterns
    avg_investment = np.mean(non_zero_investments)
    max_investment = np.max(non_zero_investments)
    min_investment = np.min(non_zero_investments)
    
    # Create implementation steps based on strategy type
    if strategy_name == "DCA (Baseline)":
        steps = [
            {
                "title": "Set up a recurring purchase",
                "description": f"Create a weekly automatic buy of about ${avg_investment:.0f} on your exchange of choice."
            },
            {
                "title": "Stick to the schedule",
                "description": "The key is consistency - don't try to time the market, just buy regularly."
            },
            {
                "title": "Store your Bitcoin securely",
                "description": "Consider moving your Bitcoin to a secure wallet periodically."
            }
        ]
    
    elif "MACO" in strategy_name:
        steps = [
            {
                "title": "Watch for crossovers",
                "description": "Once a week, check if the short-term average price crossed above the long-term average."
            },
            {
                "title": "Buy on positive crossings",
                "description": f"When the short-term average goes above the long-term average, buy around ${avg_investment:.0f}."
            },
            {
                "title": "Be patient during downtrends",
                "description": "Don't buy when the short-term average is below the long-term average."
            }
        ]
    
    elif "RSI" in strategy_name:
        steps = [
            {
                "title": "Check the RSI indicator weekly",
                "description": "The RSI shows if Bitcoin is potentially oversold (good time to buy) or overbought."
            },
            {
                "title": "Buy more during oversold periods",
                "description": f"When RSI drops below 30, consider buying more (around ${max_investment:.0f})."
            },
            {
                "title": "Buy less during normal/high periods",
                "description": f"When RSI is high, make smaller purchases (around ${min_investment:.0f}) or wait."
            }
        ]
    
    elif "Volatility" in strategy_name:
        steps = [
            {
                "title": "Monitor price volatility",
                "description": "Check once a week if Bitcoin's price is fluctuating more than usual."
            },
            {
                "title": "Buy more during high volatility",
                "description": f"When prices are swinging wildly, invest more (around ${max_investment:.0f})."
            },
            {
                "title": "Stick to regular amounts otherwise",
                "description": f"During calmer periods, make standard purchases (around ${min_investment:.0f})."
            }
        ]
    
    elif "Value" in strategy_name:
        steps = [
            {
                "title": "Set a growth target",
                "description": "Decide how much you want your Bitcoin holdings to grow each month (e.g., 5%)."
            },
            {
                "title": "Adjust weekly investments",
                "description": "Invest more when prices drop and less when prices rise to stay on your growth path."
            },
            {
                "title": "Recalculate monthly",
                "description": "At the start of each month, check if you're on target and plan your investments."
            }
        ]
    
    elif "XGBoost" in strategy_name or "ML" in strategy_name:
        steps = [
            {
                "title": "Use a price prediction service",
                "description": "Find a service that uses machine learning to predict short-term Bitcoin price movements."
            },
            {
                "title": "Follow the signals",
                "description": f"Buy around ${avg_investment:.0f} when the prediction indicates a likely price increase."
            },
            {
                "title": "Be disciplined",
                "description": "Stick to the signals even if your gut feeling says otherwise."
            }
        ]
    
    else:
        # Generic steps for other strategies
        steps = [
            {
                "title": "Establish a weekly routine",
                "description": f"Set aside time each week to check Bitcoin prices and make decisions."
            },
            {
                "title": "Follow the strategy rules",
                "description": f"Invest around ${avg_investment:.0f} according to the strategy's signals."
            },
            {
                "title": "Be consistent",
                "description": "The key to any strategy is consistency - stick with it through ups and downs."
            }
        ]
    
    # Display steps
    for i, step in enumerate(steps):
        with st.expander(f"Step {i+1}: {step['title']}"):
            st.write(step["description"])