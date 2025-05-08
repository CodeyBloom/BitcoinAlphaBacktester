"""
Tests for the domain module.

This file contains tests for the pure functions in the domain module.
"""

import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from domain import (
    calculate_investment_for_dca,
    calculate_btc_bought,
    calculate_moving_average,
    calculate_rsi,
    calculate_volatility,
    calculate_price_drop,
    is_dip,
    determine_rsi_investment_factor,
    determine_volatility_investment_factor,
    forward_fill_cumulative_values,
    find_sundays,
    apply_dca_strategy,
    apply_value_averaging_strategy,
    apply_maco_strategy,
    apply_rsi_strategy,
    apply_volatility_strategy
)

# ===== FIXTURES =====

@pytest.fixture
def sample_prices():
    """Sample price data for testing."""
    return np.array([100.0, 102.0, 99.0, 105.0, 108.0, 106.0, 110.0, 112.0, 115.0, 113.0])

@pytest.fixture
def sample_returns():
    """Sample returns data calculated from sample prices."""
    prices = np.array([100.0, 102.0, 99.0, 105.0, 108.0, 106.0, 110.0, 112.0, 115.0, 113.0])
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return returns

@pytest.fixture
def sample_dates():
    """Sample dates for testing, starting on a Monday."""
    start_date = datetime(2023, 1, 2)  # Monday
    return np.array([start_date + timedelta(days=i) for i in range(14)])

@pytest.fixture
def sample_dataframe(sample_dates, sample_prices):
    """Sample DataFrame with price data."""
    # Extend prices to match dates
    extended_prices = np.concatenate([sample_prices, sample_prices[:4]])
    
    # Calculate returns
    returns = np.zeros_like(extended_prices)
    returns[1:] = (extended_prices[1:] - extended_prices[:-1]) / extended_prices[:-1]
    
    # Find Sundays (day of week 6)
    is_sunday = np.array([date.weekday() == 6 for date in sample_dates])
    
    return pl.DataFrame({
        "date": sample_dates,
        "price": extended_prices,
        "returns": returns,
        "is_sunday": is_sunday,
        "day_of_week": [date.weekday() for date in sample_dates]
    })

# ===== TESTS FOR PURE CALCULATION FUNCTIONS =====

def test_calculate_investment_for_dca():
    """Test the calculate_investment_for_dca function."""
    assert calculate_investment_for_dca(True, 100.0) == 100.0
    assert calculate_investment_for_dca(False, 100.0) == 0.0

def test_calculate_btc_bought():
    """Test the calculate_btc_bought function."""
    assert calculate_btc_bought(100.0, 10000.0) == 0.01
    assert calculate_btc_bought(100.0, 20000.0) == 0.005
    assert calculate_btc_bought(0.0, 10000.0) == 0.0
    assert calculate_btc_bought(100.0, 0.0) == 0.0

def test_calculate_moving_average(sample_prices):
    """Test the calculate_moving_average function."""
    window = 3
    ma = calculate_moving_average(sample_prices, window)
    
    # First (window-1) values should be 0
    assert all(ma[:window-1] == 0)
    
    # Check calculated values
    assert ma[2] == (100.0 + 102.0 + 99.0) / 3
    assert ma[3] == (102.0 + 99.0 + 105.0) / 3
    assert ma[4] == (99.0 + 105.0 + 108.0) / 3

def test_calculate_rsi(sample_prices):
    """Test the calculate_rsi function."""
    period = 3
    rsi = calculate_rsi(sample_prices, period)
    
    # First period values should be 0
    assert all(rsi[:period] == 0)
    
    # RSI should be between 0 and 100
    assert all((rsi[period:] >= 0) & (rsi[period:] <= 100))

def test_calculate_volatility(sample_returns):
    """Test the calculate_volatility function."""
    window = 3
    vol = calculate_volatility(sample_returns, window)
    
    # First (window-1) values should be 0
    assert all(vol[:window-1] == 0)
    
    # Volatility should be positive
    assert all(vol[window-1:] >= 0)

def test_calculate_price_drop(sample_prices):
    """Test the calculate_price_drop function."""
    lookback_period = 3
    price_drop = calculate_price_drop(sample_prices, lookback_period)
    
    # First (lookback_period-1) values should be 0
    assert all(price_drop[:lookback_period-1] == 0)
    
    # Price drop at index 3 should be (max(prices[1:4]) - prices[3]) / max(prices[1:4])
    max_price = max(sample_prices[1:4])
    expected_drop = (max_price - sample_prices[3]) / max_price
    assert price_drop[3] == expected_drop

def test_is_dip():
    """Test the is_dip function."""
    assert is_dip(0.1, 0.05) == True
    assert is_dip(0.03, 0.05) == False
    assert is_dip(0.05, 0.05) == True

def test_determine_rsi_investment_factor():
    """Test the determine_rsi_investment_factor function."""
    oversold = 30
    overbought = 70
    
    # Test oversold condition
    assert determine_rsi_investment_factor(20, oversold, overbought) == 2.0
    assert determine_rsi_investment_factor(30, oversold, overbought) == 2.0
    
    # Test overbought condition
    assert determine_rsi_investment_factor(80, oversold, overbought) == 0.5
    assert determine_rsi_investment_factor(70, oversold, overbought) == 0.5
    
    # Test middle range (should scale linearly)
    assert determine_rsi_investment_factor(50, oversold, overbought) == 1.25
    assert determine_rsi_investment_factor(40, oversold, overbought) == 1.625
    assert determine_rsi_investment_factor(60, oversold, overbought) == 0.875

def test_determine_volatility_investment_factor():
    """Test the determine_volatility_investment_factor function."""
    threshold = 1.5
    
    # Calculate expected values based on our implementation
    vol_ratio_exact = 1.5
    vol_ratio_high = 3.0  # 0.45/0.15 = 3.0, which is 2x threshold
    vol_ratio_low1 = 2/3  # 0.1/0.15 = 2/3
    vol_ratio_low2 = 0.5  # 0.075/0.15 = 0.5
    
    # Test exactly at threshold (ratio is threshold)
    result1 = determine_volatility_investment_factor(0.15 * threshold, 0.15, threshold)
    assert result1 == 1.5
    
    # Test high volatility (ratio > threshold)
    result2 = determine_volatility_investment_factor(0.45, 0.15, threshold)
    assert round(result2, 4) == 2.0  # Should be capped at 2.0
    
    # Test very high volatility
    result3 = determine_volatility_investment_factor(0.6, 0.15, threshold)
    assert result3 == 2.0  # Definitely at cap
    
    # Test low volatility (ratio < threshold)
    result4 = determine_volatility_investment_factor(0.1, 0.15, threshold)
    expected4 = 0.5 + (vol_ratio_low1 / threshold)
    assert abs(result4 - expected4) < 0.0001
    
    # Test very low volatility
    result5 = determine_volatility_investment_factor(0.075, 0.15, threshold)
    expected5 = 0.5 + (vol_ratio_low2 / threshold)
    assert abs(result5 - expected5) < 0.0001
    
    # Test edge cases
    assert determine_volatility_investment_factor(0.15, 0.0, threshold) == 1.0  # avoid div by zero
    assert determine_volatility_investment_factor(0.0, 0.15, threshold) == 0.5  # zero volatility

def test_forward_fill_cumulative_values():
    """Test the forward_fill_cumulative_values function."""
    dates = np.array([1, 2, 3, 4, 5])
    values = np.array([10.0, 0.0, 20.0, 0.0, 0.0])
    
    result = forward_fill_cumulative_values(dates, values)
    expected = np.array([10.0, 10.0, 20.0, 20.0, 20.0])
    
    assert np.array_equal(result, expected)

def test_find_sundays(sample_dates):
    """Test the find_sundays function."""
    sundays = find_sundays(sample_dates)
    
    # Check that the correct days are identified as Sundays
    for i, date in enumerate(sample_dates):
        if date.weekday() == 6:  # Sunday
            assert sundays[i] == True
        else:
            assert sundays[i] == False

# ===== TESTS FOR STRATEGY APPLICATION FUNCTIONS =====

def test_apply_dca_strategy(sample_dataframe):
    """Test the apply_dca_strategy function."""
    weekly_investment = 100.0
    
    result = apply_dca_strategy(sample_dataframe, weekly_investment)
    
    # Check that the result has all expected columns
    for col in ["investment", "btc_bought", "cumulative_investment", "cumulative_btc"]:
        assert col in result.columns
    
    # Check that investment only happens on Sundays
    for i in range(len(result)):
        if result["is_sunday"][i]:
            assert result["investment"][i] == weekly_investment
        else:
            assert result["investment"][i] == 0.0
    
    # Check that cumulative values are monotonically increasing
    cum_inv = result["cumulative_investment"].to_numpy()
    cum_btc = result["cumulative_btc"].to_numpy()
    
    for i in range(1, len(cum_inv)):
        assert cum_inv[i] >= cum_inv[i-1]
        assert cum_btc[i] >= cum_btc[i-1]

def test_apply_value_averaging_strategy(sample_dataframe):
    """Test the apply_value_averaging_strategy function."""
    weekly_base_investment = 100.0
    target_growth_rate = 0.01  # 1% monthly
    
    result = apply_value_averaging_strategy(sample_dataframe, weekly_base_investment, target_growth_rate)
    
    # Check that the result has all expected columns
    for col in ["investment", "btc_bought", "cumulative_investment", "cumulative_btc"]:
        assert col in result.columns
    
    # Value averaging can have negative investments, but overall we should see increasing BTC
    cum_btc = result["cumulative_btc"].to_numpy()
    assert cum_btc[-1] > 0  # Final BTC position should be positive

def test_apply_maco_strategy(sample_dataframe):
    """Test the apply_maco_strategy function."""
    weekly_investment = 100.0
    short_window = 2
    long_window = 5
    
    result = apply_maco_strategy(sample_dataframe, weekly_investment, short_window, long_window)
    
    # Check that the result has all expected columns
    for col in ["short_ma", "long_ma", "signal", "investment", "btc_bought", "cumulative_investment", "cumulative_btc"]:
        assert col in result.columns
    
    # Moving averages should not be calculated until sufficient data points
    short_ma_values = result["short_ma"].to_numpy()
    long_ma_values = result["long_ma"].to_numpy()
    assert all(short_ma_values[0:short_window-1] == 0)
    assert all(long_ma_values[0:long_window-1] == 0)
    
    # With our sample data, we should see some valid moving averages after sufficient data points
    assert any(short_ma_values[short_window:] > 0)
    assert any(long_ma_values[long_window:] > 0)
    
    # Verify cumulative values are non-negative and properly accumulated
    cumulative_investment = result["cumulative_investment"].to_numpy()
    cumulative_btc = result["cumulative_btc"].to_numpy()
    
    assert all(cumulative_investment >= 0)
    assert all(cumulative_btc >= 0)
    
    # Ensure final investment has occurred (all funds used)
    last_idx = len(result) - 1
    assert cumulative_investment[last_idx] > 0
    assert cumulative_btc[last_idx] > 0

def test_apply_rsi_strategy(sample_dataframe):
    """Test the apply_rsi_strategy function."""
    weekly_investment = 100.0
    rsi_period = 3
    
    result = apply_rsi_strategy(sample_dataframe, weekly_investment, rsi_period)
    
    # Check that the result has all expected columns
    for col in ["rsi", "investment", "btc_bought", "cumulative_investment", "cumulative_btc"]:
        assert col in result.columns
    
    # RSI should not be calculated until sufficient data points
    rsi_values = result["rsi"].to_numpy()
    assert all(rsi_values[:rsi_period] == 0)
    
    # RSI values should be between 0 and 100
    assert all((rsi_values[rsi_period:] >= 0) & (rsi_values[rsi_period:] <= 100))

def test_apply_volatility_strategy(sample_dataframe):
    """Test the apply_volatility_strategy function."""
    weekly_investment = 100.0
    vol_window = 3
    
    result = apply_volatility_strategy(sample_dataframe, weekly_investment, vol_window)
    
    # Check that the result has all expected columns
    for col in ["volatility", "avg_volatility", "investment", "btc_bought", "cumulative_investment", "cumulative_btc"]:
        assert col in result.columns
    
    # Volatility should not be calculated until sufficient data points
    vol_values = result["volatility"].to_numpy()
    assert all(vol_values[:vol_window-1] == 0)
    
    # Volatility should be non-negative
    valid_vols = vol_values[vol_window:]
    valid_vols = valid_vols[~np.isnan(valid_vols)]  # Remove NaN values
    assert all(valid_vols >= 0)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])