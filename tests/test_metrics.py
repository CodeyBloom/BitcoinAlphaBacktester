"""
Tests for the metrics module of the Bitcoin Strategy Backtester.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# Import the module to test
from metrics import (
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_drawdown_over_time
)

@pytest.fixture
def sample_data():
    """Create sample data for testing metrics"""
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Create cumulative BTC values - start with increasing then a drop then recovery
    # This pattern will give us a clear drawdown to measure
    btc_values = []
    for i in range(100):
        if i < 30:
            # Steady growth
            btc_values.append(0.1 + i * 0.01)
        elif i < 50:
            # Drawdown period
            btc_values.append(0.1 + 30 * 0.01 - (i - 30) * 0.005)
        else:
            # Recovery
            btc_values.append(0.1 + 30 * 0.01 - 20 * 0.005 + (i - 50) * 0.007)
    
    # Convert to Polars DataFrame
    df = pl.DataFrame({
        "date": dates,
        "cumulative_btc": btc_values
    })
    
    return df

def test_calculate_max_drawdown(sample_data):
    """Test calculating maximum drawdown"""
    # Calculate max drawdown using our function
    max_dd = calculate_max_drawdown(sample_data)
    
    # Calculate expected drawdown manually
    peak = 0.1 + 30 * 0.01  # 0.4 BTC at peak
    trough = peak - 20 * 0.005  # 0.3 BTC at trough
    expected_max_dd = (peak - trough) / peak  # (0.4 - 0.3) / 0.4 = 0.25
    
    # Test with some numerical tolerance
    assert np.isclose(max_dd, expected_max_dd, rtol=1e-5)

def test_calculate_max_drawdown_empty():
    """Test calculating maximum drawdown with empty data"""
    # Create empty DataFrame
    df = pl.DataFrame({
        "cumulative_btc": []
    })
    
    # Should return 0 for empty data
    assert calculate_max_drawdown(df) == 0.0

def test_calculate_sortino_ratio(sample_data):
    """Test calculating Sortino ratio"""
    # Calculate Sortino ratio using our function
    sortino = calculate_sortino_ratio(sample_data)
    
    # Sortino ratio should be positive for our sample data that has growth
    assert sortino > 0
    
    # Calculate with different parameters
    sortino_high_mar = calculate_sortino_ratio(sample_data, minimum_acceptable_return=0.1)
    # Higher minimum acceptable return should result in lower Sortino ratio
    assert sortino_high_mar < sortino

def test_calculate_sortino_ratio_edge_cases():
    """Test Sortino ratio with edge cases"""
    # Create DataFrame with all positive returns
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    btc_values = [0.1 * (i + 1) for i in range(10)]  # Strictly increasing
    df_positive = pl.DataFrame({
        "date": dates,
        "cumulative_btc": btc_values
    })
    
    # Should return infinity for all positive returns
    assert calculate_sortino_ratio(df_positive) == float('inf')
    
    # Create DataFrame with few data points
    df_small = pl.DataFrame({
        "date": [datetime(2023, 1, 1)],
        "cumulative_btc": [0.1]
    })
    
    # Should return 0 for too few data points
    assert calculate_sortino_ratio(df_small) == 0.0

def test_calculate_drawdown_over_time(sample_data):
    """Test calculating drawdown over time"""
    # Calculate drawdown series
    drawdown_series = calculate_drawdown_over_time(sample_data)
    
    # Check that it's the right length
    assert len(drawdown_series) == len(sample_data)
    
    # First value should be 0 (no drawdown at start)
    assert np.isclose(drawdown_series[0], 0.0)
    
    # Maximum drawdown in the series should match our max_drawdown function
    max_dd = calculate_max_drawdown(sample_data)
    assert np.isclose(max(drawdown_series), max_dd)
    
    # Drawdown should be 0 at peak and positive during drawdown periods
    peak_idx = 30  # Based on our sample data construction
    assert np.isclose(drawdown_series[peak_idx], 0.0)
    for i in range(31, 50):
        assert drawdown_series[i] > 0
        
def test_calculate_drawdown_over_time_empty():
    """Test calculating drawdown over time with empty data"""
    # Create empty DataFrame
    df = pl.DataFrame({
        "cumulative_btc": []
    })
    
    # Should return empty series for empty data
    drawdown_series = calculate_drawdown_over_time(df)
    assert len(drawdown_series) == 0
    assert drawdown_series.name == "drawdown"