import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import numpy as np
from metrics import calculate_drawdown_over_time

def plot_cumulative_bitcoin(strategy_results):
    """
    Plot cumulative Bitcoin holdings for different strategies
    
    Args:
        strategy_results (dict): Dictionary of strategy names and their DataFrames
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    for strategy_name, df in strategy_results.items():
        # Convert Polars dataframe columns to lists for Plotly
        dates = df["date"].to_list()
        btc_values = df["cumulative_btc"].to_list()
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=btc_values,
                mode="lines",
                name=strategy_name
            )
        )
    
    # Add customization to the layout
    fig.update_layout(
        title="Cumulative Bitcoin Holdings by Strategy",
        xaxis_title="Date",
        yaxis_title="Bitcoin Holdings (BTC)",
        legend_title="Strategies",
        hovermode="x unified"
    )
    
    # Add grid and improve appearance
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    
    return fig

def plot_max_drawdown(strategy_results):
    """
    Plot maximum drawdown over time for different strategies
    
    Args:
        strategy_results (dict): Dictionary of strategy names and their DataFrames
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    for strategy_name, df in strategy_results.items():
        # Calculate drawdown over time
        drawdown = calculate_drawdown_over_time(df)
        
        # Convert Polars series to lists for Plotly
        dates = df["date"].to_list()
        drawdown_values = (drawdown * 100).to_list()  # Convert to percentage
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown_values,
                mode="lines",
                name=strategy_name
            )
        )
    
    # Add customization to the layout
    fig.update_layout(
        title="Maximum Drawdown Over Time (in BTC terms)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        legend_title="Strategies",
        hovermode="x unified"
    )
    
    # Add grid and improve appearance
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    
    return fig

def plot_sortino_ratio(performance_metrics):
    """
    Plot Sortino ratio for different strategies
    
    Args:
        performance_metrics (dict): Dictionary of strategy names and their metrics
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Extract strategy names and Sortino ratios
    strategies = list(performance_metrics.keys())
    sortino_ratios = [metrics["sortino_ratio"] for metrics in performance_metrics.values()]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=strategies,
            y=sortino_ratios,
            marker_color="royalblue"
        )
    )
    
    # Add customization to the layout
    fig.update_layout(
        title="Sortino Ratio Comparison (in BTC terms)",
        xaxis_title="Strategy",
        yaxis_title="Sortino Ratio",
        hovermode="x"
    )
    
    # Add grid and improve appearance
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    
    return fig
