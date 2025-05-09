"""
Streamlit component for the Strategy Story Dashboard.

This module contains the Streamlit interface components that render the
Strategy Story Dashboard using the pure functions from strategy_story.py.

This follows the functional programming pattern of separating:
- Pure calculations (in strategy_story.py)
- Streamlit UI rendering (in this file)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from strategy_story import (
    calculate_sleep_well_factor,
    detect_market_conditions,
    identify_key_events,
    compare_to_savings,
    create_strategy_timeline,
    simplify_recommendation,
    create_implementation_steps
)

def render_strategy_story_dashboard(strategy_results, performance_metrics, investment_currency="AUD"):
    """
    Action function to render the Strategy Story Dashboard in Streamlit.
    
    Args:
        strategy_results (dict): Dictionary of strategy results DataFrames
        performance_metrics (dict): Dictionary of strategy performance metrics
        investment_currency (str, optional): Currency code (default: "AUD")
    """
    st.header("Strategy Story Dashboard")
    
    # Introduction
    st.markdown("""
    This simplified dashboard tells the story of how each strategy performed in plain language,
    showing when they worked well and when they struggled.
    """)
    
    # If no results, show message and return
    if not strategy_results or not performance_metrics:
        st.info("Run a backtest to see the strategy story dashboard.")
        return
    
    # Section 1: Strategy Timeline
    st.subheader("What Happened Timeline")
    
    timeline_fig = create_strategy_timeline(strategy_results, performance_metrics)
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    st.markdown("""
    This timeline shows when each strategy bought Bitcoin. Colored areas show bull markets (green),
    bear markets (red), and sideways markets (gray). The bottom chart shows how much was invested on each date.
    """)
    
    # Section 2: Simple Comparison Cards
    st.subheader("Simple Comparison")
    
    # Calculate sleep well factors
    sleep_well_factors = {}
    for strategy_name, metrics in performance_metrics.items():
        if "max_drawdown" in metrics and "sortino_ratio" in metrics:
            sleep_well_factors[strategy_name] = calculate_sleep_well_factor(
                metrics["max_drawdown"], 
                metrics["sortino_ratio"]
            )
    
    # Compare to savings
    savings_comparisons = compare_to_savings(performance_metrics)
    
    # Create three columns for comparison cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Sleep Well Factor")
        
        # Create a visual representation of sleep well factor
        for strategy, factor in sleep_well_factors.items():
            # Display as a progress bar with label
            st.write(f"**{strategy}**")
            
            # Convert to 0-100 scale for progress bar
            progress_value = factor / 5 * 100
            
            # Choose color based on value
            if factor >= 4:
                bar_color = "green"
            elif factor >= 3:
                bar_color = "blue" 
            elif factor >= 2:
                bar_color = "orange"
            else:
                bar_color = "red"
                
            # Custom progress bar with HTML
            st.markdown(f"""
            <div style="margin-bottom: 10px;">
                <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px;">
                    <div style="width: {progress_value}%; height: 24px; background-color: {bar_color}; 
                          border-radius: 5px; text-align: center; color: white; line-height: 24px;">
                        {factor:.1f}/5
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add interpretation
            if factor >= 4:
                st.markdown("Very smooth ride 😴")
            elif factor >= 3:
                st.markdown("Relatively smooth 😌")
            elif factor >= 2:
                st.markdown("Some bumpy periods 😬")
            else:
                st.markdown("Quite bumpy ride 😰")
                
            st.markdown("---")
    
    with col2:
        st.markdown("### Better Than Savings?")
        
        for strategy, comparison in savings_comparisons.items():
            ratio = comparison["performance_ratio"]
            
            st.write(f"**{strategy}**")
            
            if ratio > 1:
                st.markdown(f"**{ratio:.1f}x** better than a savings account")
                
                # Visualize as multiple piggy banks
                piggy_count = min(5, int(ratio))
                st.markdown("".join(["🐖" for _ in range(piggy_count)]))
                
                extra_value = comparison["extra_value"]
                if extra_value > 0:
                    st.markdown(f"You earned **${extra_value:,.2f}** more than you would have in savings.")
            else:
                st.markdown(f"**{ratio:.1f}x** worse than a savings account")
                st.markdown("💸")
                
                if "extra_value" in comparison:
                    lost_value = -comparison["extra_value"]
                    if lost_value > 0:
                        st.markdown(f"You would have made **${lost_value:,.2f}** more in a savings account.")
            
            st.markdown("---")
    
    with col3:
        st.markdown("### Worth The Effort?")
        
        # Compare to DCA baseline
        dca_metrics = performance_metrics.get("DCA (Baseline)")
        
        if dca_metrics:
            dca_btc_per_currency = dca_metrics.get("btc_per_currency", 0)
            
            for strategy, metrics in performance_metrics.items():
                if strategy != "DCA (Baseline)" and "btc_per_currency" in metrics:
                    st.write(f"**{strategy}**")
                    
                    btc_per_currency = metrics["btc_per_currency"]
                    difference = (btc_per_currency / dca_btc_per_currency - 1) * 100
                    
                    if difference > 10:
                        st.markdown(f"**{difference:.1f}%** better than simple DCA")
                        st.markdown("✅ Definitely worth it")
                    elif difference > 3:
                        st.markdown(f"**{difference:.1f}%** better than simple DCA")
                        st.markdown("✓ Probably worth it")
                    elif difference > 0:
                        st.markdown(f"**{difference:.1f}%** better than simple DCA")
                        st.markdown("⚖️ Marginally better")
                    else:
                        st.markdown(f"**{abs(difference):.1f}%** worse than simple DCA")
                        st.markdown("❌ Not worth the extra work")
                    
                    st.markdown("---")
    
    # Section 3: Bottom Line
    st.subheader("Bottom Line")
    
    # Create recommendation
    recommendation = simplify_recommendation(performance_metrics)
    st.markdown(f"### {recommendation}")
    
    # Section 4: Try It Yourself
    st.subheader("Try It Yourself")
    
    # Find best strategy
    best_strategy = None
    best_efficiency = 0
    
    for strategy_name, metrics in performance_metrics.items():
        if "btc_per_currency" in metrics and metrics["btc_per_currency"] > best_efficiency:
            best_efficiency = metrics["btc_per_currency"]
            best_strategy = strategy_name
    
    # If we found a best strategy, show implementation steps
    if best_strategy and best_strategy in strategy_results:
        st.markdown(f"#### How to implement {best_strategy}:")
        
        # Get implementation steps
        steps = create_implementation_steps(best_strategy, strategy_results[best_strategy])
        
        # Display steps
        for i, step in enumerate(steps):
            expander = st.expander(f"Step {i+1}: {step['title']}")
            with expander:
                st.markdown(step["description"])
    
    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer**: Past performance is not indicative of future results. This tool provides educational
    information, not financial advice. Always do your own research before investing.
    """)

# Additional utility function for app.py integration
def get_last_price_from_results(strategy_results):
    """
    Pure function to get the last Bitcoin price from strategy results.
    
    Args:
        strategy_results (dict): Dictionary of strategy results DataFrames
    
    Returns:
        float or None: Last Bitcoin price or None if not available
    """
    for strategy_name, df in strategy_results.items():
        if "price" in df.columns:
            return df["price"].tail(1).item()
    return None