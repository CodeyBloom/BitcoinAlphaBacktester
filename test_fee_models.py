"""
Tests for the fee_models module.

This file tests the exchange fee calculations and other functionality
in the fee_models module.
"""

import pytest
import os
import json
from fee_models import (
    TransactionType,
    load_exchange_profiles,
    save_exchange_profiles,
    get_exchange_fee,
    calculate_transaction_cost,
    get_optimal_exchange_for_strategy,
    estimate_annual_fees,
    adjust_btc_for_fees,
    EXCHANGE_DATA_PATH
)

@pytest.fixture
def temp_exchange_data():
    """Create a temporary exchange data file for testing."""
    # Backup existing file if it exists
    backup_path = None
    if os.path.exists(EXCHANGE_DATA_PATH):
        backup_path = f"{EXCHANGE_DATA_PATH}.bak"
        os.rename(EXCHANGE_DATA_PATH, backup_path)
    
    # Create test data
    test_data = {
        "test_exchange": {
            "name": "Test Exchange",
            "standard_fee": 0.002,
            "maker_fee": 0.001,
            "taker_fee": 0.002,
            "withdrawal_fee_btc": 0.0001,
            "supported_currencies": ["USD", "EUR"]
        }
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(EXCHANGE_DATA_PATH), exist_ok=True)
    
    # Write test data
    with open(EXCHANGE_DATA_PATH, 'w') as f:
        json.dump(test_data, f)
    
    yield test_data
    
    # Cleanup
    if os.path.exists(EXCHANGE_DATA_PATH):
        os.remove(EXCHANGE_DATA_PATH)
    
    # Restore backup if it existed
    if backup_path and os.path.exists(backup_path):
        os.rename(backup_path, EXCHANGE_DATA_PATH)

def test_load_exchange_profiles(temp_exchange_data):
    """Test loading exchange profiles from file."""
    profiles = load_exchange_profiles()
    assert "test_exchange" in profiles
    assert profiles["test_exchange"]["name"] == "Test Exchange"
    assert profiles["test_exchange"]["standard_fee"] == 0.002

def test_save_exchange_profiles(temp_exchange_data):
    """Test saving exchange profiles to file."""
    profiles = load_exchange_profiles()
    profiles["test_exchange"]["standard_fee"] = 0.003
    save_exchange_profiles(profiles)
    
    # Load again to verify changes were saved
    updated_profiles = load_exchange_profiles()
    assert updated_profiles["test_exchange"]["standard_fee"] == 0.003

def test_get_exchange_fee(temp_exchange_data):
    """Test getting exchange fees."""
    # Test standard fee
    fee = get_exchange_fee("test_exchange", TransactionType.BUY)
    assert fee == 0.002
    
    # Test maker fee (using default value since test exchange has both)
    fee = get_exchange_fee("test_exchange", TransactionType.BUY)
    assert fee == 0.002  # Should default to standard_fee

def test_calculate_transaction_cost(temp_exchange_data):
    """Test calculating transaction costs."""
    amount = 1000
    net_amount, fee_amount = calculate_transaction_cost(amount, "test_exchange", TransactionType.BUY)
    
    assert fee_amount == 1000 * 0.002
    assert net_amount == 1000 - (1000 * 0.002)

def test_get_optimal_exchange_for_strategy(temp_exchange_data):
    """Test finding optimal exchange for a strategy."""
    # Only one exchange in our test data, so it should be the optimal one
    exchange, fee = get_optimal_exchange_for_strategy("dca", 100, "USD")
    assert exchange == "test_exchange"
    
    # Test with unsupported currency
    exchange, fee = get_optimal_exchange_for_strategy("dca", 100, "GBP")
    assert exchange is None

def test_estimate_annual_fees(temp_exchange_data):
    """Test estimating annual fees for different strategies."""
    # DCA strategy with weekly $100 investment
    annual_fee = estimate_annual_fees("test_exchange", "dca", 100)
    # 52 weeks * $100 per week * 0.002 fee
    expected_fee = 52 * (100 * 0.002)
    assert annual_fee == pytest.approx(expected_fee)
    
    # RSI strategy should have more transactions
    rsi_fee = estimate_annual_fees("test_exchange", "rsi", 100)
    assert rsi_fee > annual_fee  # RSI should have higher fees due to more transactions

def test_adjust_btc_for_fees(temp_exchange_data):
    """Test adjusting BTC for withdrawal fees."""
    # Without withdrawal
    btc_amount = 1.0
    adjusted = adjust_btc_for_fees(btc_amount, "test_exchange", withdrawal=False)
    assert adjusted == btc_amount
    
    # With withdrawal
    adjusted = adjust_btc_for_fees(btc_amount, "test_exchange", withdrawal=True)
    assert adjusted == btc_amount - 0.0001  # Withdrawal fee from test data

def test_unknown_exchange():
    """Test behavior with unknown exchange."""
    with pytest.raises(ValueError):
        get_exchange_fee("nonexistent_exchange", TransactionType.BUY)

def test_default_fees():
    """Test default fees for exchanges without specific fee types."""
    # Load default profiles and add a minimal exchange
    profiles = load_exchange_profiles()
    profiles["minimal"] = {"name": "Minimal Exchange", "supported_currencies": ["USD"]}
    save_exchange_profiles(profiles)
    
    # Should use default fee of 0.005 (0.5%)
    fee = get_exchange_fee("minimal", TransactionType.BUY)
    assert fee == 0.005