"""
Tests for the optimization page of the Bitcoin Strategy Backtester.
"""

import os
import sys
import pytest
import polars as pl
from datetime import datetime
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from optimize_app import display_optimization_results, OPTIMIZATION_DIR

# Mock Streamlit since we can't test it directly
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
        self.sidebar_calls = []
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
        
    def dataframe(self, df, use_container_width=False, height=None):
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
    monkeypatch.setattr("optimize_app.st", mock)
    return mock

@pytest.fixture
def sample_optimization_dir():
    """Create a temporary directory with sample optimization files"""
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    # Save original dir
    original_dir = OPTIMIZATION_DIR
    
    try:
        # Override the optimization directory
        os.environ["OPTIMIZATION_DIR"] = temp_dir
        
        # Create sample optimization files
        today = datetime.now()
        end_date = today
        
        # Create for 1 year
        start_date = end_date.replace(year=end_date.year - 1)
        start_date_str = start_date.strftime("%d%m%Y")
        end_date_str = end_date.strftime("%d%m%Y")
        
        # Create DCA data
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
        file_path = os.path.join(temp_dir, f"dca_{start_date_str}_{end_date_str}_AUD.arrow")
        df.write_ipc(file_path)
        
        # Create XGBoost ML data
        xgboost_data = {
            "strategy": "xgboost_ml",
            "param_exchange_id": "kraken",
            "param_weekly_investment": 180.0,
            "param_use_discount": True,
            "param_training_window": 14,
            "param_prediction_threshold": 0.58,
            "param_feature_set": "price,returns,volume,volatility",
            "performance_final_btc": 0.65678912,
            "performance_max_drawdown": 0.24,
            "performance_sortino_ratio": 1.55,
        }
        xgboost_df = pl.DataFrame([xgboost_data])
        xgboost_file_path = os.path.join(temp_dir, f"xgboost_ml_{start_date_str}_{end_date_str}_AUD.arrow")
        xgboost_df.write_ipc(xgboost_file_path)
        
        yield temp_dir
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        # Restore original dir
        os.environ["OPTIMIZATION_DIR"] = original_dir

def test_display_optimization_results_single_strategy(mock_st):
    """Test displaying results for a single strategy"""
    # Test data
    results = {
        "strategy": "dca",
        "best_params": {
            "exchange_id": "binance",
            "weekly_investment": 100.0,
            "use_discount": True
        },
        "performance": {
            "final_btc": 0.45678912,
            "max_drawdown": 0.21,
            "sortino_ratio": 1.35
        }
    }
    
    # Call the function
    display_optimization_results(results, single_strategy=True, currency="AUD")
    
    # Assertions
    assert "DCA" in mock_st.subheader_calls[0]
    assert f"Efficiency (BTC per AUD)" in mock_st.metrics
    assert "Final BTC Holdings" in mock_st.metrics
    assert "Total Investment" in mock_st.metrics

def test_display_optimization_results_multiple_strategies(mock_st):
    """Test displaying results for multiple strategies"""
    # Test data
    results = {
        "dca": {
            "strategy": "dca",
            "best_params": {
                "exchange_id": "binance",
                "weekly_investment": 100.0,
                "use_discount": True
            },
            "performance": {
                "final_btc": 0.45678912,
                "max_drawdown": 0.21,
                "sortino_ratio": 1.35
            }
        },
        "maco": {
            "strategy": "maco",
            "best_params": {
                "exchange_id": "coinbase",
                "weekly_investment": 150.0,
                "use_discount": False,
                "short_window": 15,
                "long_window": 75
            },
            "performance": {
                "final_btc": 0.55678912,
                "max_drawdown": 0.28,
                "sortino_ratio": 1.12
            }
        },
        "xgboost_ml": {
            "strategy": "xgboost_ml",
            "best_params": {
                "exchange_id": "kraken",
                "weekly_investment": 180.0,
                "use_discount": True,
                "training_window": 14,
                "prediction_threshold": 0.58,
                "feature_set": "price,returns,volume,volatility"
            },
            "performance": {
                "final_btc": 0.65678912,
                "max_drawdown": 0.24,
                "sortino_ratio": 1.55
            }
        }
    }
    
    # Call the function
    display_optimization_results(results, best_strategy_name="maco", single_strategy=False, currency="AUD")
    
    # Assertions
    assert "Strategy Comparison" in mock_st.subheader_calls[0]
    assert len(mock_st.dataframe_calls) == 1
    assert "most efficient strategy" in mock_st.success_calls[0]

def test_xgboost_ml_optimization_display(mock_st):
    """Test displaying optimization results for XGBoost ML strategy"""
    # Test data for XGBoost ML strategy
    results = {
        "strategy": "xgboost_ml",
        "best_params": {
            "exchange_id": "kraken",
            "weekly_investment": 180.0,
            "use_discount": True,
            "training_window": 14,
            "prediction_threshold": 0.58,
            "feature_set": "price,returns,volume,volatility"
        },
        "performance": {
            "final_btc": 0.65678912,
            "max_drawdown": 0.24,
            "sortino_ratio": 1.55
        }
    }
    
    # Call the function
    display_optimization_results(results, single_strategy=True, currency="AUD")
    
    # Assertions
    assert "XGBOOST_ML" in mock_st.subheader_calls[0]
    assert f"Efficiency (BTC per AUD)" in mock_st.metrics
    assert "Final BTC Holdings" in mock_st.metrics
    assert "Total Investment" in mock_st.metrics
    assert "Training Window" in str(mock_st.info_calls)
    assert "Prediction Threshold" in str(mock_st.info_calls)
    assert "Feature Set" in str(mock_st.info_calls)
    
def test_calculate_efficiency(mock_st):
    """Test efficiency calculation"""
    # Test data
    results = {
        "strategy": "dca",
        "best_params": {
            "exchange_id": "binance",
            "weekly_investment": 100.0,
            "use_discount": True
        },
        "performance": {
            "final_btc": 0.5,  # 0.5 BTC
            "max_drawdown": 0.21,
            "sortino_ratio": 1.35
        }
    }
    
    # Call the function
    display_optimization_results(results, single_strategy=True, currency="AUD")
    
    # Assertions - 0.5 BTC with 100 AUD weekly for 52 weeks = 0.5 / 5200 = 0.0000961...
    expected_efficiency = 0.5 / (100.0 * 52)
    assert f"{expected_efficiency:.8f}" in mock_st.metrics[f"Efficiency (BTC per AUD)"]