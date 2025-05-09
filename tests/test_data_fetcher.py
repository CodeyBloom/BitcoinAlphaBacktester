"""
Tests for the data_fetcher module.

This file contains tests for the functions in the data_fetcher module.
"""

import pytest
import polars as pl
from datetime import datetime, date
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from data_fetcher import (
    fetch_bitcoin_price_data, 
    fetch_from_api,
    parse_date_string,
    get_arrow_path,
    format_date_for_api,
    filter_dataframe_by_date_range,
    calculate_day_of_week,
    flag_sundays,
    calculate_returns,
    ensure_row_index,
    read_from_arrow_file
)

# ===== FIXTURES =====

@pytest.fixture
def sample_arrow_file():
    """Create a temporary Arrow file with sample Bitcoin price data."""
    with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as tmp:
        # Create a sample DataFrame
        df = pl.DataFrame({
            "date": [
                datetime(2023, 1, 1), 
                datetime(2023, 1, 2),
                datetime(2023, 1, 3)
            ],
            "price": [50000.0, 51000.0, 49000.0],
            "day_of_week": [6, 0, 1],  # Sunday, Monday, Tuesday
            "is_sunday": [True, False, False],
            "returns": [0.0, 0.02, -0.039]
        })
        
        # Save to Arrow file
        df.write_ipc(tmp.name)
        
        yield tmp.name
    
    # Clean up after test
    if os.path.exists(tmp.name):
        os.remove(tmp.name)

# ===== TESTS =====

def test_fetch_bitcoin_price_data_from_file(sample_arrow_file, monkeypatch):
    """Test fetching Bitcoin price data from a local Arrow file."""
    # Mock the paths to use our sample file
    def mock_get_arrow_path(*args, **kwargs):
        return sample_arrow_file
    
    monkeypatch.setattr("data_fetcher.get_arrow_path", mock_get_arrow_path)
    
    # Mock fetch_from_api to ensure it's not called
    def mock_fetch_from_api(*args, **kwargs):
        pytest.fail("fetch_from_api should not be called when local data is available")
    
    monkeypatch.setattr("data_fetcher.fetch_from_api", mock_fetch_from_api)
    
    # Call the function
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 3)
    result = fetch_bitcoin_price_data(start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y"))
    
    # Verify the result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 3
    assert result["date"][0].date() == start_date
    assert result["date"][2].date() == end_date

def test_fetch_bitcoin_price_data_filtered_range(sample_arrow_file, monkeypatch):
    """Test fetching a subset of data from a local Arrow file."""
    # Mock the paths to use our sample file
    def mock_get_arrow_path(*args, **kwargs):
        return sample_arrow_file
    
    monkeypatch.setattr("data_fetcher.get_arrow_path", mock_get_arrow_path)
    
    # Call the function with a subset date range
    start_date = date(2023, 1, 2)  # Skip first day
    end_date = date(2023, 1, 3)
    result = fetch_bitcoin_price_data(start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y"))
    
    # Verify the result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 2
    assert result["date"][0].date() == start_date

def test_fetch_bitcoin_price_data_fallback_to_api(monkeypatch):
    """Test fallback to API when local data is not available."""
    # Mock get_arrow_path to return a non-existent file
    def mock_get_arrow_path(*args, **kwargs):
        return "/non/existent/path.arrow"
    
    monkeypatch.setattr("data_fetcher.get_arrow_path", mock_get_arrow_path)
    
    # Mock fetch_from_api to return a sample DataFrame
    def mock_fetch_from_api(*args, **kwargs):
        return pl.DataFrame({
            "date": [
                datetime(2023, 1, 1), 
                datetime(2023, 1, 2)
            ],
            "price": [50000.0, 51000.0],
            "day_of_week": [6, 0],
            "is_sunday": [True, False],
            "returns": [0.0, 0.02]
        })
    
    monkeypatch.setattr("data_fetcher.fetch_from_api", mock_fetch_from_api)
    
    # Call the function
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 2)
    result = fetch_bitcoin_price_data(start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y"))
    
    # Verify the result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 2
    assert result["date"][0].date() == start_date

# ===== UTILITY FUNCTION TESTS =====

def test_parse_date_string():
    """Test parsing date strings in DD-MM-YYYY format."""
    result = parse_date_string("01-01-2023")
    assert result == date(2023, 1, 1)
    
    result = parse_date_string("31-12-2022")
    assert result == date(2022, 12, 31)
    
    # Test edge cases
    with pytest.raises(ValueError):
        parse_date_string("invalid-date")

def test_get_arrow_path():
    """Test getting the Arrow file path."""
    # Default currency
    result = get_arrow_path()
    assert result == os.path.join("data", "bitcoin_prices.arrow")
    
    # Specified currency (should still return the same path in the current implementation)
    result = get_arrow_path(currency="USD")
    assert result == os.path.join("data", "bitcoin_prices.arrow")

def test_format_date_for_api():
    """Test formatting dates for API calls."""
    test_date = date(2023, 1, 1)
    result = format_date_for_api(test_date)
    
    # The expected timestamp for 2023-01-01 00:00:00 UTC
    expected = int(datetime(2023, 1, 1, 0, 0, 0).timestamp())
    assert result == expected

def test_filter_dataframe_by_date_range():
    """Test filtering DataFrames by date range."""
    # Create a sample DataFrame
    df = pl.DataFrame({
        "date": [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5)
        ],
        "value": [1, 2, 3, 4, 5]
    })
    
    # Test inclusive range
    start_date = date(2023, 1, 2)
    end_date = date(2023, 1, 4)
    result = filter_dataframe_by_date_range(df, start_date, end_date)
    
    assert len(result) == 3
    assert result["date"][0].date() == date(2023, 1, 2)
    assert result["date"][2].date() == date(2023, 1, 4)
    
    # Test single day
    single_date = date(2023, 1, 3)
    result = filter_dataframe_by_date_range(df, single_date, single_date)
    
    assert len(result) == 1
    assert result["date"][0].date() == date(2023, 1, 3)
    
    # Test no matches
    no_match_date = date(2023, 2, 1)
    result = filter_dataframe_by_date_range(df, no_match_date, no_match_date)
    
    assert len(result) == 0

def test_calculate_day_of_week():
    """Test calculating day of week from dates."""
    df = pl.DataFrame({
        "date": [
            datetime(2023, 1, 1),  # Sunday (ISO weekday is 7)
            datetime(2023, 1, 2),  # Monday (ISO weekday is 1)
            datetime(2023, 1, 7)   # Saturday (ISO weekday is 6)
        ]
    })
    
    result = calculate_day_of_week(df)
    
    assert "day_of_week" in result.columns
    # Polars dt.weekday() returns ISO weekdays where Monday=1 through Sunday=7
    assert result["day_of_week"][0] == 7  # Sunday is 7 in ISO
    assert result["day_of_week"][1] == 1  # Monday is 1 in ISO
    assert result["day_of_week"][2] == 6  # Saturday is 6 in ISO

def test_flag_sundays():
    """Test flagging Sundays in DataFrame."""
    df = pl.DataFrame({
        "day_of_week": [1, 7, 2, 7]  # Monday, Sunday, Tuesday, Sunday
    })
    
    result = flag_sundays(df)
    
    assert "is_sunday" in result.columns
    assert not result["is_sunday"][0]  # Monday
    assert result["is_sunday"][1]      # Sunday
    assert not result["is_sunday"][2]  # Tuesday
    assert result["is_sunday"][3]      # Sunday

def test_calculate_returns():
    """Test calculating returns from prices."""
    df = pl.DataFrame({
        "price": [100.0, 110.0, 99.0, 99.0]
    })
    
    result = calculate_returns(df)
    
    assert "returns" in result.columns
    # Polars pct_change sets first value to None
    assert result["returns"][0] is None       # First entry is None in polars
    assert result["returns"][1] == 0.1        # (110 - 100) / 100 = 0.1
    assert pytest.approx(result["returns"][2], 0.001) == -0.1  # (99 - 110) / 110 = -0.1
    assert result["returns"][3] == 0.0        # (99 - 99) / 99 = 0.0

def test_ensure_row_index():
    """Test ensuring row_index column exists."""
    # DataFrame without row_index
    df = pl.DataFrame({
        "value": [1, 2, 3]
    })
    
    result = ensure_row_index(df)
    
    assert "row_index" in result.columns
    assert result["row_index"][0] == 0
    assert result["row_index"][1] == 1
    assert result["row_index"][2] == 2
    
    # DataFrame with existing row_index
    df_with_index = pl.DataFrame({
        "row_index": [10, 20, 30],
        "value": [1, 2, 3]
    })
    
    result = ensure_row_index(df_with_index)
    
    assert result["row_index"][0] == 10  # Should preserve existing row_index
    assert result["row_index"][1] == 20
    assert result["row_index"][2] == 30

def test_read_from_arrow_file(sample_arrow_file):
    """Test reading from an Arrow file."""
    # Test with existing file
    result = read_from_arrow_file(sample_arrow_file)
    
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 3
    
    # Test with non-existent file
    nonexistent_file = "/non/existent/path.arrow"
    result = read_from_arrow_file(nonexistent_file)
    
    assert result is None
    
    # Test with corrupted file
    with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as tmp:
        # Create a corrupted file (not a valid Arrow file)
        with open(tmp.name, 'w') as f:
            f.write("This is not a valid Arrow file format")
        
        # Redirect stdout to capture print statements
        with patch('builtins.print') as mock_print:
            result = read_from_arrow_file(tmp.name)
            mock_print.assert_called_once()  # Verify error message was printed
        
        assert result is None
        
        # Clean up
        os.remove(tmp.name)

# ===== API FUNCTION TESTS =====

@patch('data_fetcher.httpx.Client')
def test_fetch_from_api_success(mock_client):
    """Test successful API data fetching."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "prices": [
            # Format: [timestamp_ms, price]
            [1672531200000, 16500.0],  # 2023-01-01
            [1672617600000, 16600.0],  # 2023-01-02
            [1672704000000, 16400.0]   # 2023-01-03
        ]
    }
    
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_client.return_value.__enter__.return_value = mock_session
    
    # Test parameters
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 3)
    currency = "AUD"
    
    # Call the function
    with patch('builtins.print') as mock_print:
        result = fetch_from_api(start_date, end_date, currency)
    
    # Verify the API was called with correct parameters
    expected_params = {
        'vs_currency': currency.lower(),
        'from': format_date_for_api(start_date),
        'to': format_date_for_api(end_date)
    }
    mock_session.get.assert_called_once()
    _, kwargs = mock_session.get.call_args
    assert kwargs['params'] == expected_params
    
    # Verify the result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 3
    assert set(result.columns).issuperset({"date", "price", "day_of_week", "is_sunday", "returns", "row_index"})
    assert result["price"][0] == 16500.0

@patch('data_fetcher.httpx.Client')
def test_fetch_from_api_empty_data(mock_client):
    """Test handling empty data from API."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "prices": []  # Empty price list
    }
    
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_client.return_value.__enter__.return_value = mock_session
    
    # Test parameters
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 3)
    
    # Call the function
    with patch('builtins.print') as mock_print:
        result = fetch_from_api(start_date, end_date)
    
    # Verify appropriate error message was printed and None was returned
    mock_print.assert_any_call("No price data returned from API")
    assert result is None

@patch('data_fetcher.httpx.Client')
def test_fetch_from_api_rate_limit(mock_client):
    """Test handling rate limit response from API."""
    # Mock rate limit response followed by successful response
    mock_rate_limit_response = MagicMock()
    mock_rate_limit_response.status_code = 429
    mock_rate_limit_response.headers = {"Retry-After": "1"}  # Short delay for test
    
    mock_success_response = MagicMock()
    mock_success_response.status_code = 200
    mock_success_response.json.return_value = {
        "prices": [
            [1672531200000, 16500.0],  # 2023-01-01
            [1672617600000, 16600.0]   # 2023-01-02
        ]
    }
    
    mock_session = MagicMock()
    # Return rate limit on first call, success on second
    mock_session.get.side_effect = [mock_rate_limit_response, mock_success_response]
    mock_client.return_value.__enter__.return_value = mock_session
    
    # Test parameters
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 2)
    
    # Call the function
    with patch('builtins.print') as mock_print, patch('data_fetcher.time.sleep') as mock_sleep:
        result = fetch_from_api(start_date, end_date)
    
    # Verify retry behavior
    assert mock_session.get.call_count == 2
    mock_print.assert_any_call("Rate limit hit, waiting for 1 seconds...")
    mock_sleep.assert_called_once_with(1)
    
    # Verify result after retry is successful
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 2

@patch('data_fetcher.httpx.Client')
def test_fetch_from_api_http_error(mock_client):
    """Test handling HTTP error response from API."""
    # Mock HTTP error response
    mock_error_response = MagicMock()
    mock_error_response.status_code = 500
    mock_error_response.text = "Internal Server Error"
    
    mock_session = MagicMock()
    mock_session.get.return_value = mock_error_response
    mock_client.return_value.__enter__.return_value = mock_session
    
    # Test parameters
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 2)
    
    # Call the function
    with patch('builtins.print') as mock_print, patch('data_fetcher.time.sleep') as mock_sleep:
        result = fetch_from_api(start_date, end_date)
    
    # Verify error handling
    mock_print.assert_any_call(f"API error: 500 - Internal Server Error")
    assert mock_sleep.call_count == 3  # Called once for each retry attempt
    assert result is None  # Function should return None after all retries fail

@patch('data_fetcher.httpx.Client')
def test_fetch_from_api_exception(mock_client):
    """Test handling exceptions during API call."""
    # Mock session to raise an exception
    mock_session = MagicMock()
    mock_session.get.side_effect = Exception("Connection error")
    mock_client.return_value.__enter__.return_value = mock_session
    
    # Test parameters
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 2)
    
    # Call the function
    with patch('builtins.print') as mock_print, patch('data_fetcher.time.sleep') as mock_sleep:
        result = fetch_from_api(start_date, end_date)
    
    # Verify error handling
    mock_print.assert_any_call("Exception while fetching data: Connection error")
    assert mock_sleep.call_count == 3  # Called once for each retry attempt
    assert result is None  # Function should return None after all retries fail

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])