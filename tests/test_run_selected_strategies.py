"""
Tests for the run_selected_strategies function in app.py.

Following TDD principles, we will test various scenarios for the run_selected_strategies
function to ensure it correctly processes strategies based on user selections.
"""

import pytest
import polars as pl
from datetime import datetime, date, timedelta
import os
import sys
import numpy as np

# Import functions from app.py
from app import run_selected_strategies

# ===== FIXTURES =====

@pytest.fixture
def sample_price_data():
    """Create a sample price dataset for testing strategies."""
    # Generate 120 days of sample data
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(120)]
    
    # Create weekday indicators (using ISO weekday: 1=Monday, 7=Sunday)
    weekdays = [(base_date + timedelta(days=i)).isoweekday() for i in range(120)]
    is_sunday = [day == 7 for day in weekdays]
    
    # Create price series with some volatility to trigger strategy conditions
    base_price = 20000.0
    prices = []
    for i in range(120):
        # Add both trend and random variations
        if i < 40:  # Initial uptrend
            trend = i * 50
        elif i < 70:  # Downtrend
            trend = 2000 - (i - 40) * 100
        else:  # Recovery
            trend = -1000 + (i - 70) * 75
            
        # Add some randomness
        np.random.seed(i)  # For reproducibility
        random_change = np.random.normal(0, 200)
        
        price = max(base_price + trend + random_change, 10000)  # Ensure price doesn't go too low
        prices.append(price)
    
    # Calculate returns
    returns = [0.0]  # First day has no return
    for i in range(1, len(prices)):
        returns.append((prices[i] - prices[i-1]) / prices[i-1])
    
    # Create DataFrame
    df = pl.DataFrame({
        "date": dates,
        "price": prices,
        "day_of_week": weekdays,
        "is_sunday": is_sunday,
        "returns": returns
    })
    
    return df

@pytest.fixture
def strategy_selections():
    """Sample strategy selections for testing."""
    return {
        "dca": True,
        "value_averaging": False,
        "maco": True,
        "rsi": False,
        "volatility": True
    }

@pytest.fixture
def strategy_params():
    """Sample strategy parameters for testing."""
    return {
        "dca": {},
        "value_averaging": {"target_growth_rate": 0.05},
        "maco": {"short_window": 10, "long_window": 30},
        "rsi": {"period": 14, "oversold": 30, "overbought": 70},
        "volatility": {"window": 20, "threshold": 1.5}
    }

# ===== TESTS =====

def test_run_selected_strategies_empty_data():
    """Test handling of empty DataFrame input."""
    empty_df = pl.DataFrame({"date": [], "price": [], "day_of_week": [], "is_sunday": [], "returns": []})
    strategy_selections = {"dca": True}
    strategy_params = {"dca": {}}
    
    results, metrics = run_selected_strategies(
        empty_df, strategy_selections, strategy_params, 100, None, False
    )
    
    assert results == {}
    assert metrics == {}

def test_run_selected_strategies_no_selection(sample_price_data):
    """Test when no strategies are selected."""
    strategy_selections = {"dca": False, "maco": False, "rsi": False, "volatility": False, "value_averaging": False}
    strategy_params = {"dca": {}, "maco": {}, "rsi": {}, "volatility": {}, "value_averaging": {}}
    
    results, metrics = run_selected_strategies(
        sample_price_data, strategy_selections, strategy_params, 100, None, False
    )
    
    assert results == {}
    assert metrics == {}

def test_run_selected_strategies_single_strategy(sample_price_data):
    """Test running a single strategy (DCA)."""
    strategy_selections = {"dca": True, "maco": False, "rsi": False, "volatility": False, "value_averaging": False}
    strategy_params = {"dca": {}, "maco": {}, "rsi": {}, "volatility": {}, "value_averaging": {}}
    weekly_investment = 100
    
    results, metrics = run_selected_strategies(
        sample_price_data, strategy_selections, strategy_params, weekly_investment, None, False
    )
    
    # Check if DCA strategy results are present (could be "DCA" or "DCA (Baseline)")
    dca_key = next((k for k in results.keys() if k.startswith("DCA")), None)
    assert dca_key is not None, "No DCA strategy results found"
    assert dca_key in metrics, f"No metrics found for {dca_key}"
    
    # Verify result structure
    dca_result = results[dca_key]
    assert isinstance(dca_result, pl.DataFrame)
    assert set(dca_result.columns).issuperset({"date", "price", "investment", "btc_bought", "cumulative_investment", "cumulative_btc"})
    
    # Verify investment logic - DCA invests weekly on Sundays
    sunday_investments = dca_result.filter(pl.col("is_sunday")).select("investment")
    # Check each investment value
    for i in range(len(sunday_investments)):
        assert sunday_investments[i, 0] == weekly_investment
    
    # Verify metrics structure
    dca_metrics = metrics[dca_key]
    assert "total_invested" in dca_metrics
    assert "total_btc" in dca_metrics
    assert "final_btc_value" in dca_metrics
    assert "efficiency" in dca_metrics
    
    # Verify basic metrics values
    assert dca_metrics["total_invested"] > 0
    assert dca_metrics["total_btc"] > 0
    assert dca_metrics["final_btc_value"] > 0

def test_run_selected_strategies_multiple_strategies(sample_price_data, strategy_selections, strategy_params):
    """Test running multiple strategies simultaneously."""
    weekly_investment = 100
    
    results, metrics = run_selected_strategies(
        sample_price_data, strategy_selections, strategy_params, weekly_investment, None, False
    )
    
    # Verify we have three strategies run (some variant of DCA, MACO, and Volatility)
    assert len(results) == 3, f"Expected 3 strategies, got {len(results)}: {list(results.keys())}"
    assert len(metrics) == 3, f"Expected 3 metrics, got {len(metrics)}: {list(metrics.keys())}"
    
    # Find actual strategy names (they might be "DCA (Baseline)" instead of "DCA" etc.)
    dca_key = next((k for k in results.keys() if "DCA" in k), None)
    maco_key = next((k for k in results.keys() if "MACO" in k), None)
    volatility_key = next((k for k in results.keys() if "Volatility" in k), None)
    
    assert dca_key is not None, "No DCA strategy found"
    assert maco_key is not None, "No MACO strategy found"
    assert volatility_key is not None, "No Volatility strategy found"
    
    # Verify each strategy has the correct structure
    for strategy_name in [dca_key, maco_key, volatility_key]:
        strategy_result = results[strategy_name]
        assert isinstance(strategy_result, pl.DataFrame)
        assert set(strategy_result.columns).issuperset({"date", "price", "investment", "btc_bought", "cumulative_investment", "cumulative_btc"})
        
        strategy_metrics = metrics[strategy_name]
        assert "total_invested" in strategy_metrics
        assert "total_btc" in strategy_metrics
        assert "final_btc_value" in strategy_metrics
        assert "efficiency" in strategy_metrics
    
    # Verify the strategies behave differently
    dca_efficiency = metrics["DCA"]["efficiency"]
    maco_efficiency = metrics["MACO"]["efficiency"]
    volatility_efficiency = metrics["Volatility"]["efficiency"]
    
    # In a realistic scenario, efficiencies should differ
    assert len({dca_efficiency, maco_efficiency, volatility_efficiency}) > 1

def test_run_selected_strategies_with_exchange_fees(sample_price_data):
    """Test running strategies with exchange fee calculations."""
    strategy_selections = {"dca": True}
    strategy_params = {"dca": {}}
    weekly_investment = 100
    exchange_id = "binance"  # Using a known exchange
    
    # Run with and without exchange fees
    results_with_fees, metrics_with_fees = run_selected_strategies(
        sample_price_data, strategy_selections, strategy_params, weekly_investment, exchange_id, False
    )
    
    results_no_fees, metrics_no_fees = run_selected_strategies(
        sample_price_data, strategy_selections, strategy_params, weekly_investment, None, False
    )
    
    # With fees, we should get less BTC
    assert metrics_with_fees["DCA"]["total_btc"] < metrics_no_fees["DCA"]["total_btc"]
    
    # Test with exchange discount
    results_with_discount, metrics_with_discount = run_selected_strategies(
        sample_price_data, strategy_selections, strategy_params, weekly_investment, exchange_id, True
    )
    
    # With discount, we should get more BTC than without discount
    assert metrics_with_discount["DCA"]["total_btc"] > metrics_with_fees["DCA"]["total_btc"]