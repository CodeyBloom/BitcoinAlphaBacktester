"""
Tests for the main app of the Bitcoin Strategy Backtester.
"""

import pytest
import os
import sys
import polars as pl
from datetime import datetime, timedelta
import tempfile
import shutil
import subprocess
from importlib import import_module

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app functions to test
from app import (
    ensure_optimization_data_exists,
    get_optimization_files,
    get_strategy_parameters,
    run_strategies_with_parameters
)

# Create a mock Streamlit class similar to the one in test_optimize_app.py
class MockSt:
    """Mock Streamlit for testing"""
    def __init__(self):
        self.metrics = {}
        self.markdown_calls = []
        self.info_calls = []
        self.success_calls = []
        self.error_calls = []
        self.warning_calls = []
        self.title_calls = []
        self.header_calls = []
        self.subheader_calls = []
        self.columns_calls = []
        self.dataframe_calls = []
        self.expander_calls = []
        self.plotly_calls = []
        self.button_calls = []
        self.selectbox_calls = []
        
        # Add session_state as a dictionary that can be used in tests
        self.session_state = {
            "current_time_period": "1 Year",
            "all_optimization_results": {},
            "most_efficient_strategy": "dca"
        }
        
    def metric(self, label, value):
        self.metrics[label] = value
        
    def markdown(self, text):
        self.markdown_calls.append(text)
        
    def info(self, text):
        self.info_calls.append(text)
        
    def success(self, text):
        self.success_calls.append(text)
        
    def error(self, text):
        self.error_calls.append(text)
        
    def warning(self, text):
        self.warning_calls.append(text)
        
    def title(self, text):
        self.title_calls.append(text)
        
    def header(self, text):
        self.header_calls.append(text)
        
    def subheader(self, text):
        self.subheader_calls.append(text)
        
    def columns(self, n):
        return [self] * n  # Return list of self as columns
        
    def dataframe(self, df, use_container_width=False):
        self.dataframe_calls.append(df)
        
    def write(self, text):
        # Can be used to examine write calls if needed
        pass
        
    def plotly_chart(self, fig, use_container_width=False):
        self.plotly_calls.append(fig)
        
    def button(self, label, type=None, key=None, help=None):
        self.button_calls.append(label)
        return False  # Default to not pressed
        
    def selectbox(self, label, options, index=0, key=None):
        self.selectbox_calls.append((label, options))
        return options[index]
        
    def expander(self, text):
        self.expander_calls.append(text)
        return self  # Return self for context manager
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
        
    def spinner(self, text):
        # Mock the spinner context manager
        class MockSpinner:
            def __enter__(self):
                return None
            def __exit__(self, *args):
                pass
        return MockSpinner()
        
    class sidebar:
        @staticmethod
        def header(text):
            pass
            
        @staticmethod
        def radio(label, options, index=0, help=None):
            return options[index]
            
        @staticmethod
        def checkbox(label, value=False):
            return value
            
        @staticmethod
        def button(label, type=None, key=None, help=None):
            return False  # Default to not pressed

# Create a fixture for the mock streamlit
@pytest.fixture
def mock_st(monkeypatch):
    """Create a mock Streamlit instance and patch the module"""
    mock = MockSt()
    monkeypatch.setattr("app.st", mock)
    return mock

@pytest.fixture
def sample_price_data():
    """Create sample Bitcoin price data for testing"""
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    # Create price data with some volatility
    prices = [20000 + i * 100 + (i % 5) * 200 for i in range(30)]
    
    # Create day of week and Sunday flags
    day_of_week = [(start_date + timedelta(days=i)).weekday() for i in range(30)]
    is_sunday = [dow == 6 for dow in day_of_week]
    
    # Calculate returns
    returns = [0.0]
    for i in range(1, 30):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    
    # Convert to Polars DataFrame
    df = pl.DataFrame({
        "date": dates,
        "price": prices,
        "day_of_week": day_of_week,
        "is_sunday": is_sunday,
        "returns": returns
    })
    
    return df

@pytest.fixture
def mock_optimization_dir():
    """Create a temporary directory with mock optimization files"""
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a sample optimization file
        today = datetime.now()
        end_date = today
        start_date = end_date.replace(year=end_date.year - 1)
        start_date_str = start_date.strftime("%d%m%Y")
        end_date_str = end_date.strftime("%d%m%Y")
        
        # Create optimization data for DCA
        data = {
            "strategy": "dca",
            "param_exchange_id": "binance",
            "param_weekly_investment": 100.0,
            "param_use_discount": True,
            "performance_final_btc": 0.45678912,
            "performance_max_drawdown": 0.21,
            "performance_sortino_ratio": 1.35,
        }
        df = pl.DataFrame([data])
        
        # Create the directory structure
        os.makedirs(os.path.join(temp_dir, "optimizations"), exist_ok=True)
        
        # Save the file
        file_path = os.path.join(temp_dir, "optimizations", f"dca_{start_date_str}_{end_date_str}_AUD.arrow")
        df.write_ipc(file_path)
        
        yield temp_dir
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

def test_ensure_optimization_data_exists(monkeypatch, mock_optimization_dir):
    """Test the function to ensure optimization data exists"""
    import streamlit as st
    
    # Mock the optimization directory path
    monkeypatch.setattr("scripts.generate_optimizations_for_periods.OPTIMIZATION_DIR", 
                       os.path.join(mock_optimization_dir, "optimizations"))
    
    # Mock the streamlit functions
    monkeypatch.setattr(st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "error", lambda *args, **kwargs: None)
    
    # Mock the generate_optimizations function import
    # This is a better approach than mocking import_module which isn't used
    monkeypatch.setattr("scripts.generate_optimizations_for_periods.main", lambda: None)
    
    # Create a file so ensure_optimization_data_exists doesn't try to generate files
    os.makedirs(os.path.join(mock_optimization_dir, "optimizations"), exist_ok=True)
    with open(os.path.join(mock_optimization_dir, "optimizations", "dca_09052024_09052025_AUD.arrow"), "w") as f:
        f.write("test")
    
    # Call the function
    result = ensure_optimization_data_exists()
    
    # Verify the function returns True when it succeeds
    assert result is True

def test_run_selected_strategies_empty_data():
    """Test that run_selected_strategies handles empty dataframes gracefully"""
    from app import run_selected_strategies
    import polars as pl
    
    # Create an empty dataframe
    empty_df = pl.DataFrame({
        "date": [],
        "price": [],
        "day_of_week": [],
        "is_sunday": [],
        "returns": []
    })
    
    # Test with empty data
    results, metrics = run_selected_strategies(
        empty_df, {"dca": True}, {"dca": {}}, 100, None, False
    )
    
    # Should return empty dictionaries
    assert results == {}
    assert metrics == {}

def test_run_selected_strategies(sample_price_data):
    """Test the run_selected_strategies function with different strategy selections"""
    from app import run_selected_strategies
    
    # Test with all strategies
    strategy_selections = {
        "dca": True,
        "value_avg": True,
        "maco": True,
        "rsi": True,
        "volatility": True
    }
    
    # Create parameter dictionary
    strategy_params = {
        "dca": {},
        "value_avg": {"target_growth_rate": 0.02},
        "maco": {"short_window": 10, "long_window": 30},
        "rsi": {"rsi_period": 7, "oversold_threshold": 20, "overbought_threshold": 80},
        "volatility": {"vol_window": 10, "vol_threshold": 2.0}
    }
    
    weekly_investment = 100
    
    # Run with all strategies
    results, metrics = run_selected_strategies(
        sample_price_data, strategy_selections, strategy_params, weekly_investment, None, False
    )
    
    # Verify all strategies were included
    assert "DCA (Baseline)" in results
    assert "Value Averaging" in results
    assert "MACO" in results
    assert "RSI" in results
    assert "Volatility" in results
    
    # Test with exchange fees
    results_with_fees, metrics_with_fees = run_selected_strategies(
        sample_price_data, {"dca": True}, {"dca": {}}, weekly_investment, "binance", False
    )
    
    # Test with exchange discount
    results_with_discount, metrics_with_discount = run_selected_strategies(
        sample_price_data, {"dca": True}, {"dca": {}}, weekly_investment, "binance", True
    )
    
    # Verify exchange fees are applied correctly (less BTC with fees than without)
    dca_key = next((k for k in results.keys() if "DCA" in k), None)
    dca_key_fees = next((k for k in results_with_fees.keys() if "DCA" in k), None)
    dca_key_discount = next((k for k in results_with_discount.keys() if "DCA" in k), None)
    
    assert metrics_with_fees[dca_key_fees]["final_btc"] < metrics[dca_key]["final_btc"]
    assert metrics_with_discount[dca_key_discount]["final_btc"] > metrics_with_fees[dca_key_fees]["final_btc"]

def test_get_optimization_files(monkeypatch, mock_optimization_dir):
    """Test the function to get optimization files for a specific time period"""
    from app import get_optimization_files
    
    # Mock the optimization directory path
    monkeypatch.setattr("scripts.generate_optimizations_for_periods.OPTIMIZATION_DIR", 
                       os.path.join(mock_optimization_dir, "optimizations"))
    
    # Create a few test optimization files
    today = datetime.now()
    end_date = today
    
    # 1 year file
    start_date_1y = end_date.replace(year=end_date.year - 1)
    start_date_1y_str = start_date_1y.strftime("%d%m%Y")
    end_date_str = end_date.strftime("%d%m%Y")
    
    # 5 year file
    start_date_5y = end_date.replace(year=end_date.year - 5)
    start_date_5y_str = start_date_5y.strftime("%d%m%Y")
    
    # Create the directory structure
    os.makedirs(os.path.join(mock_optimization_dir, "optimizations"), exist_ok=True)
    
    # Create empty files for testing
    files_to_create = [
        f"dca_{start_date_1y_str}_{end_date_str}_AUD.arrow",
        f"maco_{start_date_1y_str}_{end_date_str}_AUD.arrow",
        f"dca_{start_date_5y_str}_{end_date_str}_AUD.arrow",
        f"rsi_{start_date_5y_str}_{end_date_str}_AUD.arrow",
        f"volatility_{start_date_5y_str}_{end_date_str}_AUD.arrow"
    ]
    
    for file_name in files_to_create:
        with open(os.path.join(mock_optimization_dir, "optimizations", file_name), "w") as f:
            f.write("test")
    
    # Test getting files for 1 Year period
    results = get_optimization_files(period="1 Year")
    assert "dca" in results
    assert "maco" in results
    assert len(results.get("dca", [])) == 1
    assert len(results.get("maco", [])) == 1
    
    # Test getting files for 5 Years period
    results = get_optimization_files(period="5 Years")
    assert "dca" in results
    assert "rsi" in results
    assert "volatility" in results
    assert len(results.get("dca", [])) == 1
    
    # Test filtering by strategies
    results = get_optimization_files(period="5 Years", strategies=["dca", "rsi"])
    assert "dca" in results
    assert "rsi" in results
    assert "volatility" not in results
    
    # Test default behavior (all files)
    results = get_optimization_files()
    assert "dca" in results
    assert "maco" in results
    assert "rsi" in results
    assert "volatility" in results
    assert len(results.get("dca", [])) == 2  # Both 1 Year and 5 Years files

def test_get_strategy_parameters():
    """Test getting strategy parameters with selected strategy"""
    # Test with DCA
    params = get_strategy_parameters("dca")
    assert "exchange_id" in params
    assert "weekly_investment" in params
    assert "use_discount" in params
    # Verify default values
    assert params["weekly_investment"] == 100.0
    
    # Test with MACO
    params = get_strategy_parameters("maco")
    assert "exchange_id" in params
    assert "weekly_investment" in params
    assert "short_window" in params
    assert "long_window" in params
    # Verify MACO specific parameter defaults
    assert params["short_window"] == 20
    assert params["long_window"] == 100
    
    # Test with RSI
    params = get_strategy_parameters("rsi")
    assert "exchange_id" in params
    assert "weekly_investment" in params
    assert "rsi_period" in params
    assert "oversold_threshold" in params
    assert "overbought_threshold" in params
    # Verify RSI specific parameter defaults
    assert params["rsi_period"] == 14
    assert params["oversold_threshold"] == 30
    assert params["overbought_threshold"] == 70
    
    # Test with volatility
    params = get_strategy_parameters("volatility")
    assert "exchange_id" in params
    assert "weekly_investment" in params
    assert "vol_window" in params
    assert "vol_threshold" in params
    # Verify volatility specific parameter defaults
    assert params["vol_window"] == 14
    assert params["vol_threshold"] == 1.5
    
    # Test with invalid strategy - should return empty dict
    params = get_strategy_parameters("invalid_strategy")
    assert params == {}

def test_run_strategies_with_parameters(sample_price_data):
    """Test running strategies with parameters"""
    # Define test parameters for all strategies to validate
    strategies_with_params = {
        "DCA (Baseline)": {
            "strategy": "dca",
            "parameters": {
                "weekly_investment": 100.0,
                "exchange_id": "binance",
                "use_discount": False
            }
        },
        "MACO": {
            "strategy": "maco",
            "parameters": {
                "weekly_investment": 150.0,  # Different investment amount
                "exchange_id": "coinbase",   # Different exchange
                "short_window": 10,
                "long_window": 20
            }
        },
        "Value Averaging": {
            "strategy": "value_avg",
            "parameters": {
                "weekly_investment": 120.0,
                "target_growth_rate": 0.02  # 2% monthly
            }
        },
        "RSI Strategy": {
            "strategy": "rsi",
            "parameters": {
                "weekly_investment": 130.0,
                "rsi_period": 10,
                "oversold_threshold": 25,
                "overbought_threshold": 75
            }
        },
        "Volatility": {
            "strategy": "volatility",
            "parameters": {
                "weekly_investment": 110.0,
                "vol_window": 7,
                "vol_threshold": 2.0
            }
        }
    }
    
    # Run strategies
    results, metrics = run_strategies_with_parameters(sample_price_data, strategies_with_params)
    
    # Check results structure for all strategies
    expected_strategies = ["DCA (Baseline)", "MACO", "Value Averaging", "RSI Strategy", "Volatility"]
    for strategy_name in expected_strategies:
        assert strategy_name in results, f"Strategy {strategy_name} missing from results"
        assert strategy_name in metrics, f"Strategy {strategy_name} missing from metrics"
    
    # Verify metrics content for each strategy
    for strategy_name in expected_strategies:
        strategy_metrics = metrics[strategy_name]
        assert "final_btc" in strategy_metrics
        assert "max_drawdown" in strategy_metrics
        assert "sortino_ratio" in strategy_metrics
    
    # Verify that strategies have different results due to different parameters
    final_btc_values = [metrics[s]["final_btc"] for s in expected_strategies]
    # At least some should be different (strategies should not all produce identical results)
    assert len(set(final_btc_values)) > 1, "Strategies should produce different results"
    
    # Verify that strategies with different investment amounts show expected proportionality
    dca_df = results["DCA (Baseline)"]
    maco_df = results["MACO"]
    
    # DCA should invest exactly 100 per week (on Sundays)
    sunday_dca_investments = dca_df.filter(pl.col("is_sunday"))["investment"]
    if len(sunday_dca_investments) > 0:
        # All Sunday investments should be 100 (or very close due to numeric precision)
        assert all(abs(inv - 100.0) < 0.001 for inv in sunday_dca_investments if inv > 0)
    
    # MACO should have some investments that are 150 (or multiples)
    sunday_maco_investments = maco_df.filter(pl.col("is_sunday"))["investment"]
    if len(sunday_maco_investments) > 0:
        # No Sunday investment should match the DCA amount exactly
        assert not any(abs(inv - 100.0) < 0.001 for inv in sunday_maco_investments if inv > 0)
    
    # Verify that the total invested matches the parameters
    sunday_count = sample_price_data.filter(pl.col("is_sunday")).height
    if sunday_count > 0:
        # Check if total DCA investment is close to expected (100 × number of Sundays)
        expected_dca_investment = 100.0 * sunday_count
        actual_dca_investment = dca_df["investment"].sum()
        # Allow for some investments to be skipped based on strategy
        assert actual_dca_investment <= expected_dca_investment
        
        # Confirm that strategies with different parameters lead to different total investments
        assert dca_df["investment"].sum() != maco_df["investment"].sum()