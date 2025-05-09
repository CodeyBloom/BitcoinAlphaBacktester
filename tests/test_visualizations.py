"""
Tests for the visualizations module of the Bitcoin Strategy Backtester.
"""

import pytest
import polars as pl
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import the module to test
from visualizations import (
    plot_cumulative_bitcoin,
    plot_max_drawdown,
    plot_sortino_ratio
)

@pytest.fixture
def sample_data():
    """Create sample data for testing visualizations"""
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    # Create sample data for two strategies
    dca_data = pl.DataFrame({
        "date": dates,
        "cumulative_btc": [0.01 * i for i in range(30)]
    })
    
    maco_data = pl.DataFrame({
        "date": dates,
        "cumulative_btc": [0.009 * i * (i % 3 + 1) for i in range(30)]
    })
    
    # Return as a dictionary of strategy results
    return {
        "DCA": dca_data,
        "MACO": maco_data
    }

@pytest.fixture
def performance_metrics():
    """Create sample performance metrics"""
    return {
        "DCA": {
            "final_btc": 0.29,
            "max_drawdown": 0.1,
            "sortino_ratio": 1.5
        },
        "MACO": {
            "final_btc": 0.32,
            "max_drawdown": 0.15,
            "sortino_ratio": 1.8
        },
        "RSI": {
            "final_btc": 0.28,
            "max_drawdown": 0.12,
            "sortino_ratio": 1.4
        }
    }

def test_plot_cumulative_bitcoin(sample_data):
    """Test plotting cumulative Bitcoin holdings"""
    # Generate the plot
    fig = plot_cumulative_bitcoin(sample_data)
    
    # Check that the figure was created
    assert isinstance(fig, go.Figure)
    
    # Check that it has the right number of traces (one per strategy)
    assert len(fig.data) == len(sample_data)
    
    # Verify trace names match strategy names
    data_traces = list(fig.data)
    strategy_names = list(sample_data.keys())
    for i, trace in enumerate(data_traces):
        assert trace.name in strategy_names
    
    # Check the layout has appropriate titles
    assert "Cumulative Bitcoin Holdings" in fig.layout.title.text
    assert "Date" in fig.layout.xaxis.title.text
    assert "Bitcoin Holdings" in fig.layout.yaxis.title.text

def test_plot_max_drawdown(sample_data):
    """Test plotting maximum drawdown over time"""
    # Generate the plot
    fig = plot_max_drawdown(sample_data)
    
    # Check that the figure was created
    assert isinstance(fig, go.Figure)
    
    # Check that it has the right number of traces (one per strategy)
    assert len(fig.data) == len(sample_data)
    
    # Check that each trace has the right name
    strategy_names = set(sample_data.keys())
    trace_names = {trace.name for trace in fig.data}
    assert strategy_names == trace_names
    
    # Check the layout has appropriate titles
    assert "Maximum Drawdown" in fig.layout.title.text
    assert "Date" in fig.layout.xaxis.title.text
    assert "Drawdown" in fig.layout.yaxis.title.text

def test_plot_sortino_ratio(performance_metrics):
    """Test plotting Sortino ratio comparison"""
    # Generate the plot
    fig = plot_sortino_ratio(performance_metrics)
    
    # Check that the figure was created
    assert isinstance(fig, go.Figure)
    
    # Check that it has one trace (bar chart)
    assert len(fig.data) == 1
    
    # Check that the bar chart has the right number of bars
    bar_trace = fig.data[0]
    assert len(bar_trace.x) == len(performance_metrics)
    assert len(bar_trace.y) == len(performance_metrics)
    
    # Check the layout has appropriate titles
    assert "Sortino Ratio" in fig.layout.title.text
    assert "Strategy" in fig.layout.xaxis.title.text
    assert "Sortino Ratio" in fig.layout.yaxis.title.text
    
    # Check the values match the performance metrics
    for i, strategy in enumerate(bar_trace.x):
        assert bar_trace.y[i] == performance_metrics[strategy]["sortino_ratio"]

def test_plot_empty_data():
    """Test plotting with empty data"""
    # Create empty strategy results
    empty_results = {}
    
    # Test with empty data
    fig1 = plot_cumulative_bitcoin(empty_results)
    assert isinstance(fig1, go.Figure)
    assert len(fig1.data) == 0
    
    fig2 = plot_max_drawdown(empty_results)
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) == 0
    
    # Empty performance metrics
    empty_metrics = {}
    fig3 = plot_sortino_ratio(empty_metrics)
    assert isinstance(fig3, go.Figure)
    assert len(fig3.data) == 1  # Bar chart with no bars
    assert len(fig3.data[0].x) == 0
    assert len(fig3.data[0].y) == 0