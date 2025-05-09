"""
Strategy Story Dashboard module for Bitcoin investment strategy visualization.

This module provides components for a user-friendly dashboard that explains
strategy performance in simple, everyday language with intuitive visualizations.

This module follows functional programming principles from "Grokking Simplicity":
- Functions are categorized as calculations (pure functions) or actions (with side effects)
- Data is treated as immutable
- Complex operations are composed of smaller, reusable functions
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from datetime import datetime, timedelta

# ===== UTILITY FUNCTIONS (PURE) =====

def calculate_sleep_well_factor(max_drawdown, sortino_ratio):
    """
    Pure function to calculate a "Sleep Well Factor" on a scale of 1-5.
    
    This is a simplified risk assessment based on maximum drawdown and Sortino ratio.
    
    Args:
        max_drawdown (float): Maximum drawdown (0-1 scale)
        sortino_ratio (float): Sortino ratio
    
    Returns:
        float: Sleep Well Factor on a scale of 1-5 (higher is better)
    """
    # Invert max_drawdown so higher is better
    drawdown_component = 5 * (1 - max_drawdown)
    
    # Scale sortino ratio to 0-5 range
    # Typical sortino ratios range from -1 to 2, so we'll scale accordingly
    sortino_component = min(5, max(0, sortino_ratio + 1)) * 2.5 / 3
    
    # Combine with more weight on drawdown for "sleep well" factor
    sleep_well_factor = 0.7 * drawdown_component + 0.3 * sortino_component
    
    # Ensure we stay within 1-5 range
    return max(1, min(5, sleep_well_factor))

def detect_market_conditions(df):
    """
    Pure function to detect bull, bear, and sideways market conditions.
    
    This function identifies periods of significant trends in Bitcoin price.
    
    Args:
        df (polars.DataFrame): DataFrame with price data
    
    Returns:
        dict: Dictionary with market condition periods
            {
                "bull": [(start_idx, end_idx, description), ...],
                "bear": [(start_idx, end_idx, description), ...],
                "sideways": [(start_idx, end_idx, description), ...]
            }
    """
    # Convert to numpy for calculations
    prices = df["price"].to_numpy()
    dates = df["date"].to_numpy()
    
    # Initialize results
    bull_periods = []
    bear_periods = []
    sideways_periods = []
    
    # Configuration
    window = min(30, len(prices) // 10)  # 30 days or 1/10 of series
    threshold_bull = 0.15  # 15% increase
    threshold_bear = -0.15  # 15% decrease
    
    # Detect trends
    i = window
    while i < len(prices):
        # Calculate percentage change over window
        change = (prices[i] - prices[i-window]) / prices[i-window]
        
        # Detect significant trends
        if change > threshold_bull:
            # Find end of bull run (when price stops increasing significantly)
            end = i
            for j in range(i+1, min(i+window*2, len(prices))):
                if prices[j] > prices[end]:
                    end = j
                elif prices[j] < prices[end] * 0.95:  # 5% drop from peak
                    break
            
            # Add bull period with simple description
            start_date = dates[i-window].strftime("%b %Y")
            end_date = dates[end].strftime("%b %Y")
            gain = (prices[end] / prices[i-window] - 1) * 100
            desc = f"Bitcoin rose {gain:.0f}% from {start_date} to {end_date}"
            bull_periods.append((i-window, end, desc))
            
            # Skip to end of this period
            i = end + 1
            
        elif change < threshold_bear:
            # Find end of bear market (when price stops decreasing significantly)
            end = i
            for j in range(i+1, min(i+window*2, len(prices))):
                if prices[j] < prices[end]:
                    end = j
                elif prices[j] > prices[end] * 1.05:  # 5% recovery from bottom
                    break
            
            # Add bear period with simple description
            start_date = dates[i-window].strftime("%b %Y")
            end_date = dates[end].strftime("%b %Y")
            loss = (1 - prices[end] / prices[i-window]) * 100
            desc = f"Bitcoin fell {loss:.0f}% from {start_date} to {end_date}"
            bear_periods.append((i-window, end, desc))
            
            # Skip to end of this period
            i = end + 1
            
        else:
            # Potential sideways market, check volatility
            std_dev = np.std(prices[i-window:i+1]) / np.mean(prices[i-window:i+1])
            if std_dev < 0.05:  # Low volatility
                # Find end of sideways market
                end = i
                for j in range(i+1, min(i+window*2, len(prices))):
                    new_std = np.std(prices[i-window:j+1]) / np.mean(prices[i-window:j+1])
                    if new_std > 0.08:  # Volatility increases
                        break
                    end = j
                
                # Only record if sufficient length (at least window days)
                if end - (i-window) >= window:
                    start_date = dates[i-window].strftime("%b %Y")
                    end_date = dates[end].strftime("%b %Y")
                    desc = f"Bitcoin traded sideways from {start_date} to {end_date}"
                    sideways_periods.append((i-window, end, desc))
                
                i = end + 1
            else:
                i += 1
        
    return {
        "bull": bull_periods,
        "bear": bear_periods,
        "sideways": sideways_periods
    }

def identify_key_events(results):
    """
    Pure function to identify key events in strategy performance.
    
    Finds significant buying/selling points and their impact.
    
    Args:
        results (dict): Dictionary of strategy results DataFrames
    
    Returns:
        list: List of event dictionaries
            [
                {
                    "date": datetime,
                    "description": str,
                    "strategy": str,
                    "impact": float  # Positive or negative impact
                },
                ...
            ]
    """
    events = []
    
    for strategy_name, df in results.items():
        # Skip if not a real strategy
        if strategy_name == "Price Data":
            continue
            
        # Convert to numpy for calculations
        dates = df["date"].to_numpy()
        investments = df["investment"].to_numpy()
        btc_bought = df["btc_bought"].to_numpy()
        prices = df["price"].to_numpy()
        
        # Find significant investment days (top 10% of investments)
        if np.sum(investments > 0) > 0:  # Only if there are investments
            threshold = np.percentile(investments[investments > 0], 90)
            for i in range(len(investments)):
                if investments[i] > threshold:
                    # Get percentage of monthly investment
                    last_month = max(0, i - 30)
                    monthly_investment = np.sum(investments[last_month:i])
                    if monthly_investment > 0:
                        pct_of_monthly = investments[i] / monthly_investment * 100
                        
                        # Only include if significant percentage of monthly
                        if pct_of_monthly > 25:  # More than 25% of monthly investment
                            # Describe the event
                            price_change = 0
                            if i > 0:
                                price_change = (prices[i] / prices[i-1] - 1) * 100
                                
                            if price_change < -5:
                                desc = f"Bought {btc_bought[i]:.8f} BTC during a {abs(price_change):.1f}% price drop"
                            elif price_change > 5:
                                desc = f"Bought {btc_bought[i]:.8f} BTC during a {price_change:.1f}% price increase"
                            else:
                                desc = f"Made a large purchase of {btc_bought[i]:.8f} BTC"
                                
                            events.append({
                                "date": dates[i],
                                "description": desc,
                                "strategy": strategy_name,
                                "impact": 1.0  # Positive impact
                            })
        
        # Find missed opportunities (for strategies other than DCA)
        if strategy_name != "DCA (Baseline)" and "DCA (Baseline)" in results:
            dca_df = results["DCA (Baseline)"]
            dca_btc_bought = dca_df["btc_bought"].to_numpy()
            
            # Find where DCA bought significantly more than this strategy
            for i in range(len(btc_bought)):
                if dca_btc_bought[i] > 0 and (btc_bought[i] / dca_btc_bought[i] < 0.5 if btc_bought[i] > 0 else True):
                    # Check if price dropped significantly after this
                    if i < len(prices) - 7:  # At least 7 days left
                        future_change = (prices[i+7] / prices[i] - 1) * 100
                        if future_change > 10:  # Price increased by 10%+
                            desc = f"Missed buying opportunity before a {future_change:.1f}% price increase"
                            events.append({
                                "date": dates[i],
                                "description": desc,
                                "strategy": strategy_name,
                                "impact": -0.5  # Negative impact
                            })
    
    # Sort events by date
    events.sort(key=lambda x: x["date"])
    
    return events

def compare_to_savings(metrics, savings_rate=0.03):
    """
    Pure function to compare strategy performance to traditional savings.
    
    Args:
        metrics (dict): Dictionary of strategy performance metrics
        savings_rate (float): Annual interest rate for savings account (default: 3%)
    
    Returns:
        dict: Dictionary of savings comparison results
            {
                "Strategy Name": {
                    "performance_ratio": float,  # How many times better than savings
                    "better_than_savings": bool,
                    "extra_value": float  # Additional value compared to savings
                },
                ...
            }
    """
    results = {}
    
    for strategy_name, strategy_metrics in metrics.items():
        # Get investment details
        total_invested = strategy_metrics["total_invested"]
        final_btc = strategy_metrics["final_btc"]
        btc_price = None
        
        # Find the final Bitcoin price from any strategy result
        for name, data in metrics.items():
            if "final_price" in data:
                btc_price = data["final_price"]
                break
        
        # If price not available, assume current holdings equal investment
        # (this is a fallback, but the better approach is to include final_price in metrics)
        if btc_price is None:
            btc_price = total_invested / final_btc
        
        # Calculate strategy end value in currency
        strategy_end_value = final_btc * btc_price
        
        # Calculate what the same investments would be worth in a savings account
        # This is a simplification assuming weekly equal investments
        weeks = total_invested / 100  # Assuming $100/week standard
        weekly_rate = savings_rate / 52
        
        # Calculate future value of periodic investment
        # FV = PMT * ((1 + r)^n - 1) / r
        if weekly_rate > 0:
            savings_end_value = 100 * ((1 + weekly_rate) ** weeks - 1) / weekly_rate
        else:
            savings_end_value = 100 * weeks
        
        # Calculate performance ratio
        if savings_end_value > 0:
            performance_ratio = strategy_end_value / savings_end_value
        else:
            performance_ratio = 1.0  # Default to 1 if no savings value
            
        # Record results
        results[strategy_name] = {
            "performance_ratio": performance_ratio,
            "better_than_savings": performance_ratio > 1,
            "extra_value": strategy_end_value - savings_end_value
        }
    
    return results

def create_strategy_timeline(results, metrics):
    """
    Pure function to create a visual timeline of strategy performance.
    
    This timeline shows price, buy/sell points, and market conditions.
    
    Args:
        results (dict): Dictionary of strategy results DataFrames
        metrics (dict): Dictionary of strategy performance metrics
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object with the timeline
    """
    # Get any strategy result to extract dates and prices
    first_strategy = next(iter(results.values()))
    dates = first_strategy["date"].to_numpy()
    prices = first_strategy["price"].to_numpy()
    
    # Create subplot with two rows
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=["Bitcoin Price with Key Events", "Investment Timeline"]
    )
    
    # Add price line to top plot
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=prices,
            mode='lines',
            name='Bitcoin Price',
            line=dict(color='#2C3E50', width=1)
        ),
        row=1, col=1
    )
    
    # Detect market conditions
    market_conditions = detect_market_conditions(first_strategy)
    
    # Add market condition shadings
    for condition, periods in market_conditions.items():
        for start_idx, end_idx, desc in periods:
            color = "#c8e6c9" if condition == "bull" else "#ffcdd2" if condition == "bear" else "#e0e0e0"
            opacity = 0.3
            
            # Add shaded area
            fig.add_vrect(
                x0=dates[start_idx],
                x1=dates[end_idx],
                fillcolor=color,
                opacity=opacity,
                layer="below",
                line_width=0,
                row="all", col="all"
            )
            
            # Add annotation at the top of the shading
            mid_idx = (start_idx + end_idx) // 2
            fig.add_annotation(
                x=dates[mid_idx],
                y=max(prices[start_idx:end_idx+1]) * 1.05,
                text=desc,
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#c7c7c7",
                borderwidth=1,
                borderpad=4,
                row=1, col=1
            )
    
    # Add buy points for each strategy
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    color_idx = 0
    
    for strategy_name, df in results.items():
        # Skip price data trace
        if strategy_name == "Price Data":
            continue
            
        # Get buy data
        strategy_dates = df["date"].to_numpy()
        investments = df["investment"].to_numpy()
        btc_bought = df["btc_bought"].to_numpy()
        
        # Find buy points (non-zero investments)
        buy_dates = []
        buy_prices = []
        buy_sizes = []
        
        for i in range(len(investments)):
            if investments[i] > 0:
                buy_dates.append(strategy_dates[i])
                buy_prices.append(prices[i])
                
                # Scale marker size based on BTC bought relative to strategy's max
                max_btc = max(btc_bought) if max(btc_bought) > 0 else 1
                normalized_size = btc_bought[i] / max_btc * 15 + 5  # Scale to 5-20 range
                buy_sizes.append(normalized_size)
        
        # Add buy points to price chart
        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    name=f'{strategy_name} Buys',
                    marker=dict(
                        color=colors[color_idx % len(colors)],
                        size=buy_sizes,
                        line=dict(width=1, color='#FFFFFF')
                    ),
                    hovertemplate='%{x|%d %b %Y}<br>Bought BTC at $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add investment timeline to bottom chart
        fig.add_trace(
            go.Scatter(
                x=strategy_dates,
                y=investments,
                mode='lines',
                name=f'{strategy_name} Investments',
                line=dict(color=colors[color_idx % len(colors)], width=1),
                fill='tozeroy',
                fillcolor=f'rgba(0, 0, 0, 0.1)',
                hovertemplate='%{x|%d %b %Y}<br>Invested: $%{y:,.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        color_idx += 1
    
    # Identify key events
    events = identify_key_events(results)
    
    # Add key events as annotations on the price chart
    for i, event in enumerate(events):
        if i >= 5:  # Limit to top 5 events to avoid cluttering
            break
            
        fig.add_annotation(
            x=event["date"],
            y=prices[np.where(dates == event["date"])[0][0]] * 0.95,
            text=event["description"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#636363",
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Bitcoin Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Investment Amount", row=2, col=1)
    
    return fig

def simplify_recommendation(metrics):
    """
    Pure function to create a simple, plain-language recommendation.
    
    Args:
        metrics (dict): Dictionary of strategy performance metrics
    
    Returns:
        str: Plain language recommendation
    """
    # Find the strategy with the most BTC per currency
    best_strategy = None
    best_efficiency = 0
    
    for strategy_name, strategy_metrics in metrics.items():
        if "btc_per_currency" in strategy_metrics:
            if strategy_metrics["btc_per_currency"] > best_efficiency:
                best_efficiency = strategy_metrics["btc_per_currency"]
                best_strategy = strategy_name
    
    # Check if we found a best strategy
    if not best_strategy:
        return "Based on the data, no clear winner emerged among the tested strategies."
    
    # Check if best strategy is DCA
    if best_strategy == "DCA (Baseline)":
        return ("The simplest approach (regular weekly buying) worked best in this period. "
                "No need to overthink it - just set up a recurring purchase each week.")
    
    # For other strategies, create a more detailed recommendation
    recommendation = f"{best_strategy} performed best during this period. "
    
    # Add comparison to DCA if available
    if "DCA (Baseline)" in metrics:
        dca_efficiency = metrics["DCA (Baseline)"]["btc_per_currency"]
        improvement = (best_efficiency / dca_efficiency - 1) * 100
        
        if improvement > 5:
            recommendation += f"It got you {improvement:.1f}% more Bitcoin than regular weekly buying. "
        else:
            recommendation += "It performed only slightly better than regular weekly buying. "
    
    # Add risk assessment
    if "max_drawdown" in metrics[best_strategy]:
        max_drawdown = metrics[best_strategy]["max_drawdown"]
        if max_drawdown > 0.25:
            recommendation += "Keep in mind it had some big drops along the way. "
        elif max_drawdown < 0.15:
            recommendation += "It also had smoother performance with fewer big drops. "
    
    # Complete with actionable next step
    recommendation += f"Consider using the {best_strategy} approach if you're comfortable with the steps shown below."
    
    return recommendation

def create_implementation_steps(strategy_name, strategy_df):
    """
    Pure function to create simple implementation steps for a strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        strategy_df (polars.DataFrame): Strategy results DataFrame
    
    Returns:
        list: List of step dictionaries with titles and descriptions
    """
    steps = []
    
    # Analyze the strategy behavior to inform implementation steps
    investments = strategy_df["investment"].to_numpy()
    non_zero_investments = investments[investments > 0]
    
    if len(non_zero_investments) == 0:
        return [{"title": "No investments", "description": "This strategy made no investments in the period."}]
    
    # Calculate investment patterns
    avg_investment = np.mean(non_zero_investments)
    max_investment = np.max(non_zero_investments)
    min_investment = np.min(non_zero_investments)
    
    # Determine frequency
    investment_days = np.where(investments > 0)[0]
    if len(investment_days) > 1:
        intervals = np.diff(investment_days)
        avg_interval = np.mean(intervals)
    else:
        avg_interval = 0
    
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
    
    # Keep only the first 3 steps to keep it simple
    return steps[:3]