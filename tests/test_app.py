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
    # Mock the optimization directory path
    monkeypatch.setattr("scripts.generate_optimizations_for_periods.OPTIMIZATION_DIR", 
                       os.path.join(mock_optimization_dir, "optimizations"))
    
    # Mock the generate_optimizations function
    monkeypatch.setattr("app.import_module", lambda _: None)
    monkeypatch.setattr("app.subprocess.run", lambda *args, **kwargs: None)
    
    # Call the function
    ensure_optimization_data_exists()
    
    # Since we can't easily verify the function's behavior directly,
    # we're mainly checking that it doesn't throw exceptions

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
    # Define some test parameters with distinct values to validate strategy behavior
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
        }
    }
    
    # Run strategies
    results, metrics = run_strategies_with_parameters(sample_price_data, strategies_with_params)
    
    # Check results structure
    assert "DCA (Baseline)" in results
    assert "MACO" in results
    
    # Check metrics structure
    assert "DCA (Baseline)" in metrics
    assert "MACO" in metrics
    
    # Verify metrics content and that they differ (which validates calculation)
    dca_metrics = metrics["DCA (Baseline)"]
    maco_metrics = metrics["MACO"]
    
    assert "final_btc" in dca_metrics
    assert "max_drawdown" in dca_metrics
    assert "sortino_ratio" in dca_metrics
    
    # Verify calculated values are different between strategies (with different parameters)
    # This validates that the strategies are actually using the parameters
    assert dca_metrics["final_btc"] != maco_metrics["final_btc"], "Strategy results should differ with different parameters"
    
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