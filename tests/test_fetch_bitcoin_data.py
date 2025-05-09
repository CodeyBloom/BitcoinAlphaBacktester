"""
Tests for the fetch_bitcoin_data module.

This module tests the functions in fetch_bitcoin_data.py for fetching Bitcoin price data.
"""

import pytest
from datetime import date, datetime, timedelta
import polars as pl
import httpx
from unittest.mock import patch, MagicMock

# Import the module to test
from fetch_bitcoin_data import (
    fetch_last_year_bitcoin_data,
    parse_api_response,
    process_price_data
)

@pytest.fixture
def mock_successful_response():
    """Fixture for a successful API response"""
    # Sample data format from CoinGecko API
    prices = [
        [1600000000000, 20000.0],  # Unix timestamp (ms), price
        [1600086400000, 20100.0],
        [1600172800000, 19800.0],
        [1600259200000, 20500.0],
        [1600345600000, 20800.0]
    ]
    return {
        "prices": prices,
        "market_caps": [[ts, 1000000000.0] for ts, _ in prices],
        "total_volumes": [[ts, 50000000.0] for ts, _ in prices]
    }

@pytest.fixture
def sample_processed_df():
    """Fixture for a sample processed DataFrame"""
    dates = [
        datetime.fromtimestamp(1600000000),
        datetime.fromtimestamp(1600086400),
        datetime.fromtimestamp(1600172800),
        datetime.fromtimestamp(1600259200),
        datetime.fromtimestamp(1600345600)
    ]
    prices = [20000.0, 20100.0, 19800.0, 20500.0, 20800.0]
    
    # Create DataFrame
    df = pl.DataFrame({
        "date": dates,
        "price": prices
    })
    
    # Add day_of_week, is_sunday, and returns columns
    df = df.with_columns([
        pl.col("date").dt.weekday().alias("day_of_week"),
        (pl.col("date").dt.weekday() == 6).alias("is_sunday")
    ])
    
    # Calculate returns
    df = df.with_columns([
        pl.col("price").pct_change().fill_null(0).alias("returns")
    ])
    
    return df

def test_parse_api_response(mock_successful_response):
    """Test that API responses are parsed correctly"""
    result = parse_api_response(mock_successful_response)
    
    # Verify result structure
    assert isinstance(result, list)
    assert len(result) == 5
    
    # Verify data format
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], datetime)
        assert isinstance(item[1], float)
    
    # Verify data values
    assert result[0][1] == 20000.0
    assert result[4][1] == 20800.0

def test_process_price_data():
    """Test that price data is processed correctly"""
    # Sample data
    data = [
        (datetime.fromtimestamp(1600000000), 20000.0),
        (datetime.fromtimestamp(1600086400), 20100.0),
        (datetime.fromtimestamp(1600172800), 19800.0),
        (datetime.fromtimestamp(1600259200), 20500.0),
        (datetime.fromtimestamp(1600345600), 20800.0)
    ]
    
    # Process data
    df = process_price_data(data)
    
    # Verify DataFrame
    assert isinstance(df, pl.DataFrame)
    assert df.height == 5
    assert set(df.columns) == {"date", "price", "day_of_week", "is_sunday", "returns"}
    
    # Verify column types
    assert df["date"].dtype == pl.Datetime
    assert df["price"].dtype == pl.Float64
    assert df["day_of_week"].dtype == pl.Int32
    assert df["is_sunday"].dtype == pl.Boolean
    assert df["returns"].dtype == pl.Float64
    
    # Verify calculated values
    assert df["returns"][0] == 0.0  # First day should be 0
    assert pytest.approx(df["returns"][1]) == 0.005  # 20100/20000 - 1

@patch("fetch_bitcoin_data.httpx.get")
def test_fetch_last_year_bitcoin_data_success(mock_get, mock_successful_response, sample_processed_df):
    """Test successful data fetching"""
    # Mock the httpx.get response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_successful_response
    mock_get.return_value = mock_response
    
    # Call the function
    result = fetch_last_year_bitcoin_data("AUD")
    
    # Verify the API was called with the right parameters
    mock_get.assert_called_once()
    call_args = mock_get.call_args[1]
    assert "params" in call_args
    params = call_args["params"]
    assert "vs_currency" in params
    assert params["vs_currency"] == "aud"
    
    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) == {"date", "price", "day_of_week", "is_sunday", "returns"}
    assert result.height > 0

@patch("fetch_bitcoin_data.httpx.get")
def test_fetch_last_year_bitcoin_data_error(mock_get):
    """Test error handling in data fetching"""
    # Mock the httpx.get response for an error
    mock_response = MagicMock()
    mock_response.status_code = 429  # Too Many Requests
    mock_get.return_value = mock_response
    
    # Call the function
    result = fetch_last_year_bitcoin_data("AUD")
    
    # Verify the function returns None on error
    assert result is None

@patch("fetch_bitcoin_data.httpx.get")
def test_fetch_last_year_bitcoin_data_exception(mock_get):
    """Test exception handling in data fetching"""
    # Mock the httpx.get to raise an exception
    mock_get.side_effect = httpx.RequestError("Connection error")
    
    # Call the function
    result = fetch_last_year_bitcoin_data("AUD")
    
    # Verify the function returns None on exception
    assert result is None