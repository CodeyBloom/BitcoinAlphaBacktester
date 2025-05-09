"""
Tests for UI components used in the Bitcoin Strategy Backtester.
"""
import pytest
import streamlit as st
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import patch, MagicMock

# Import the modules we'll be testing
import app
import optimize_app
from ui_components import strategy_card, efficiency_chart, metrics_dashboard


def test_strategy_card_displays_correct_metrics():
    """Test that strategy cards display key metrics correctly"""
    # Mock strategy data
    strategy_name = "test_strategy"
    efficiency = 0.00001234
    params = {"param1": 10, "param2": 20}
    is_most_efficient = True
    currency = "AUD"
    
    # Create a test streamlit container
    with patch("streamlit.container") as mock_container:
        # Create a mock context manager
        mock_cm = MagicMock()
        mock_container.return_value.__enter__.return_value = mock_cm
        
        # Call the component with test data
        result = strategy_card(
            strategy_name=strategy_name, 
            efficiency=efficiency, 
            params=params, 
            is_most_efficient=is_most_efficient,
            currency=currency
        )
        
        # Check that the container was created
        mock_container.assert_called_once()
        
        # Verify that the component displays the efficiency with proper formatting
        # We're expecting 8 decimal places for BTC values
        mock_cm.info.assert_any_call(f"**Efficiency:** {efficiency:.8f} BTC/{currency}")
        
        # Verify that the component highlights when it's the most efficient
        mock_cm.success.assert_called_once()


def test_efficiency_chart_shows_correct_metrics():
    """Test that efficiency chart shows the correct values"""
    # Mock results data structure
    mock_results = {
        "strategy1": {
            "efficiency_curve": np.array([0.0001, 0.0002, 0.0003]),
            "dates": [datetime.now() - timedelta(days=i) for i in range(3, 0, -1)],
            "performance": {"final_btc": 0.0001, "total_invested": 100},
            "best_params": {"weekly_investment": 10}
        },
        "strategy2": {
            "efficiency_curve": np.array([0.00005, 0.00015, 0.00025]),
            "dates": [datetime.now() - timedelta(days=i) for i in range(3, 0, -1)],
            "performance": {"final_btc": 0.00008, "total_invested": 100},
            "best_params": {"weekly_investment": 10}
        }
    }
    
    most_efficient = "strategy1"
    currency = "AUD"
    
    # Create a test streamlit container
    with patch("streamlit.plotly_chart") as mock_plotly_chart:
        # Call the component with test data
        result = efficiency_chart(
            results=mock_results,
            most_efficient=most_efficient,
            currency=currency
        )
        
        # Check that plotly chart was called
        mock_plotly_chart.assert_called_once()
        
        # Extract the figure from the call
        fig = mock_plotly_chart.call_args[0][0]
        
        # Check that we have the correct number of traces
        assert len(fig.data) == len(mock_results), "Chart should have one trace per strategy"
        
        # Check that the most efficient strategy is labeled correctly
        assert any("MOST EFFICIENT" in trace.name for trace in fig.data), "Most efficient strategy should be labeled"
        
        # Verify y-axis title includes currency
        assert f"Efficiency (BTC/{currency})" in fig.layout.yaxis.title.text, "Y-axis should show efficiency with currency"


def test_metrics_dashboard_shows_key_information():
    """Test that metrics dashboard shows key performance information"""
    # Create mock optimization results
    mock_results = {
        "strategy1": {
            "performance": {
                "final_btc": 0.1,
                "total_invested": 1000,
                "max_drawdown": 0.3,
                "sortino_ratio": 1.5
            },
            "best_params": {"weekly_investment": 100}
        },
        "strategy2": {
            "performance": {
                "final_btc": 0.08,
                "total_invested": 1000,
                "max_drawdown": 0.25,
                "sortino_ratio": 1.2
            },
            "best_params": {"weekly_investment": 100}
        }
    }
    
    most_efficient = "strategy1"
    currency = "AUD"
    
    # Create a test streamlit container
    with patch("streamlit.container") as mock_container:
        # Create a mock context manager
        mock_cm = MagicMock()
        mock_container.return_value.__enter__.return_value = mock_cm
        
        # Call the component with test data
        result = metrics_dashboard(
            results=mock_results,
            most_efficient=most_efficient,
            currency=currency
        )
        
        # Check that container was created
        mock_container.assert_called_once()
        
        # Verify that the dashboard shows the most efficient strategy
        mock_cm.markdown.assert_any_call("## Strategy Performance Overview")