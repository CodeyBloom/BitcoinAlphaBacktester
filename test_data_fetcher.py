"""
Tests for the data_fetcher module.

This file contains tests for the functions in the data_fetcher module.
"""

import pytest
import polars as pl
from datetime import datetime, date
import os
import tempfile
from data_fetcher import fetch_bitcoin_price_data, fetch_from_api

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

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])