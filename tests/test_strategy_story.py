"""
Tests for the Strategy Story Dashboard components.

Following TDD principles, these tests define the expected behavior
of the new dashboard components before implementation.
"""
import pytest
import polars as pl
import numpy as np
from datetime import date, timedelta
import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy_story import (
    calculate_sleep_well_factor,
    detect_market_conditions,
    identify_key_events,
    compare_to_savings,
    create_strategy_timeline,
    simplify_recommendation,
    create_implementation_steps
)

# ===== TEST DATA FIXTURES =====

@pytest.fixture
def sample_strategy_results():
    """Generate sample strategy results for testing."""
    # Create sample dates
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    # Create price data with a realistic pattern (including a crash and recovery)
    # Use numpy array for consistent typing
    prices = np.zeros(len(dates))
    prices[0] = 10000
    
    for i in range(1, len(dates)):
        # Add some realistic price movements
        change = prices[i-1] * np.random.normal(0.001, 0.03)  # Small drift, some volatility
        # Add a crash in the middle
        if i == len(dates) // 3:
            change = -prices[i-1] * 0.2  # 20% crash
        # Add a rally near the end
        if i == int(len(dates) * 0.7):
            change = prices[i-1] * 0.15  # 15% rally
        new_price = prices[i-1] + change
        if new_price < 1000:
            new_price = 1000.0  # Ensure price doesn't go too low
        prices[i] = new_price
    
    # Create test data frames for DCA and another strategy
    dca_df = pl.DataFrame({
        "date": dates,
        "price": prices,
        "investment": [100 if i % 7 == 0 else 0 for i in range(len(dates))],  # Weekly on Sundays
        "btc_bought": [100 / prices[i] if i % 7 == 0 else 0 for i in range(len(dates))],
        "cumulative_btc": np.cumsum([100 / prices[i] if i % 7 == 0 else 0 for i in range(len(dates))]),
        "cumulative_investment": np.cumsum([100 if i % 7 == 0 else 0 for i in range(len(dates))])
    })
    
    # Create a slightly different pattern for the test strategy
    test_investments = []
    test_btc_bought = []
    for i in range(len(dates)):
        if i % 7 == 0:  # Weekly base
            inv = 100
        elif prices[i] < prices[i-1] * 0.95:  # Buy more on 5% dips
            inv = 150
        else:
            inv = 0
        test_investments.append(inv)
        test_btc_bought.append(inv / prices[i] if inv > 0 else 0)
    
    test_df = pl.DataFrame({
        "date": dates,
        "price": prices,
        "investment": test_investments,
        "btc_bought": test_btc_bought,
        "cumulative_btc": np.cumsum(test_btc_bought),
        "cumulative_investment": np.cumsum(test_investments)
    })
    
    return {"DCA (Baseline)": dca_df, "Test Strategy": test_df}

@pytest.fixture
def sample_metrics():
    """Generate sample performance metrics for testing."""
    return {
        "DCA (Baseline)": {
            "final_btc": 0.05,
            "max_drawdown": 0.2,
            "sortino_ratio": 0.8,
            "total_invested": 5200,
            "btc_per_currency": 0.00000962
        },
        "Test Strategy": {
            "final_btc": 0.055,
            "max_drawdown": 0.18,
            "sortino_ratio": 0.9,
            "total_invested": 5500,
            "btc_per_currency": 0.00001000
        }
    }

# ===== TESTS =====

def test_calculate_sleep_well_factor():
    """Test the sleep well factor calculation."""
    # High drawdown should result in a lower sleep well factor
    assert calculate_sleep_well_factor(0.4, 0.7) < calculate_sleep_well_factor(0.1, 0.7)
    
    # Higher sortino ratio should result in a higher sleep well factor
    assert calculate_sleep_well_factor(0.2, 1.5) > calculate_sleep_well_factor(0.2, 0.5)
    
    # Factor should be on a 1-5 scale
    sleep_well_factor1 = calculate_sleep_well_factor(0.5, 0.1)
    sleep_well_factor2 = calculate_sleep_well_factor(0.1, 2.0)
    assert 1 <= sleep_well_factor1 <= 5
    assert 1 <= sleep_well_factor2 <= 5

def test_detect_market_conditions(sample_strategy_results):
    """Test market condition detection."""
    dca_df = sample_strategy_results["DCA (Baseline)"]
    
    # Function should return dict with bull, bear, sideways keys
    market_conditions = detect_market_conditions(dca_df)
    assert "bull" in market_conditions
    assert "bear" in market_conditions
    assert "sideways" in market_conditions
    
    # Each condition should be a list of tuples (start_index, end_index, description)
    assert isinstance(market_conditions["bull"], list)
    if market_conditions["bull"]:  # If any bull markets found
        assert isinstance(market_conditions["bull"][0], tuple)
        assert len(market_conditions["bull"][0]) == 3

def test_identify_key_events(sample_strategy_results):
    """Test key event identification."""
    results = sample_strategy_results
    
    # Should return a list of events with dates and descriptions
    events = identify_key_events(results)
    assert isinstance(events, list)
    
    # Each event should have date, description, and strategy fields
    if events:
        assert "date" in events[0]
        assert "description" in events[0]
        assert "strategy" in events[0]

def test_compare_to_savings(sample_metrics):
    """Test savings comparison calculation."""
    metrics = sample_metrics
    
    # Should return a dict with comparison results for each strategy
    savings_comparison = compare_to_savings(metrics, 0.04)  # 4% annual yield
    assert "DCA (Baseline)" in savings_comparison
    assert "Test Strategy" in savings_comparison
    
    # Each strategy should have a performance_ratio > 0
    assert savings_comparison["DCA (Baseline)"]["performance_ratio"] > 0
    assert savings_comparison["Test Strategy"]["performance_ratio"] > 0

def test_create_strategy_timeline(sample_strategy_results, sample_metrics):
    """Test strategy timeline creation."""
    results = sample_strategy_results
    metrics = sample_metrics
    
    # Should return a figure object
    fig = create_strategy_timeline(results, metrics)
    assert fig is not None
    
    # Figure should have data and layout attributes
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")

def test_simplify_recommendation(sample_metrics):
    """Test recommendation simplification."""
    metrics = sample_metrics
    
    # Should return a string recommendation
    recommendation = simplify_recommendation(metrics)
    assert isinstance(recommendation, str)
    assert len(recommendation) > 0
    
    # Should mention the best strategy
    assert "Test Strategy" in recommendation

def test_create_implementation_steps(sample_strategy_results):
    """Test implementation steps creation."""
    results = sample_strategy_results
    
    # Should return a list of step dicts
    strategy_name = "Test Strategy"
    steps = create_implementation_steps(strategy_name, results[strategy_name])
    assert isinstance(steps, list)
    
    # Each step should have a title and description
    if steps:
        assert "title" in steps[0]
        assert "description" in steps[0]
        
    # Steps should be simple (3 or fewer)
    assert len(steps) <= 3