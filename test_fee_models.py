"""
Unit tests for the fee models module.

This module contains tests for the fee calculation functions that
support different cryptocurrency exchanges.
"""

import json
import os
import pytest
from fee_models import (
    TransactionType,
    load_exchange_profiles,
    save_exchange_profiles,
    get_exchange_fee,
    calculate_transaction_cost,
    get_optimal_exchange_for_strategy,
    estimate_annual_fees,
    adjust_btc_for_fees
)

def test_load_exchange_profiles():
    """Test loading exchange profiles from JSON file."""
    profiles = load_exchange_profiles()
    assert isinstance(profiles, dict)
    assert len(profiles) > 0
    
    # Check some expected exchanges
    assert "binance" in profiles
    assert "river" in profiles
    assert "kraken" in profiles
    
    # Check structure of a profile
    river = profiles.get("river")
    assert river is not None
    assert "name" in river
    assert "url" in river
    assert "dca_fee" in river
    assert "standard_fee" in river

def test_get_exchange_fee():
    """Test fee calculation for different exchanges and transaction types."""
    # River offers 0% for DCA purchases
    assert get_exchange_fee("river", TransactionType.BUY) == 0.0
    
    # Binance has 0.1% maker fee, reduced to 0.075% with BNB discount
    assert get_exchange_fee("binance", TransactionType.BUY) == 0.001
    assert get_exchange_fee("binance", TransactionType.BUY, use_discount=True) == 0.00075
    
    # Kraken has tiered fees
    assert get_exchange_fee("kraken", TransactionType.BUY, volume_tier="tier1") == 0.0026
    assert get_exchange_fee("kraken", TransactionType.BUY, volume_tier="tier2") == 0.0024
    assert get_exchange_fee("kraken", TransactionType.BUY, volume_tier="tier3") == 0.0020
    
    # Coinbase has 0.6% fee, 0% with Coinbase One
    assert get_exchange_fee("coinbase", TransactionType.BUY) == 0.006
    assert get_exchange_fee("coinbase", TransactionType.BUY, use_discount=True) == 0.0
    
    # Swan has 0.99% fee for DCA
    assert get_exchange_fee("swan", TransactionType.BUY) == 0.0099

def test_calculate_transaction_cost():
    """Test transaction cost calculation including fees."""
    # River with $100 purchase (no fees)
    net_amount, fee_amount = calculate_transaction_cost(100, "river", TransactionType.BUY)
    assert fee_amount == 0.0
    assert net_amount == 100.0
    
    # Binance with $100 purchase (0.1% fee)
    net_amount, fee_amount = calculate_transaction_cost(100, "binance", TransactionType.BUY)
    assert fee_amount == 0.1
    assert net_amount == 99.9
    
    # Binance with BNB discount (0.075% fee)
    net_amount, fee_amount = calculate_transaction_cost(
        100, "binance", TransactionType.BUY, use_discount=True
    )
    assert fee_amount == 0.075
    assert net_amount == 99.925
    
    # Swan with $100 purchase (0.99% fee)
    net_amount, fee_amount = calculate_transaction_cost(100, "swan", TransactionType.BUY)
    assert fee_amount == 0.99
    assert net_amount == 99.01

def test_get_optimal_exchange_for_strategy():
    """Test finding the optimal exchange for a strategy."""
    # For DCA, River should be best (0% fee)
    exchange, fee = get_optimal_exchange_for_strategy("dca", 100, "USD")
    assert exchange == "river"
    assert fee == 0.0
    
    # Test with a currency not supported by River
    exchange, fee = get_optimal_exchange_for_strategy("dca", 100, "EUR")
    assert exchange != "river"  # River doesn't support EUR
    
    # Test with a small weekly investment (below minimum for some exchanges)
    exchange, fee = get_optimal_exchange_for_strategy("dca", 1, "USD")
    # Should find the exchange with lowest minimum purchase

def test_estimate_annual_fees():
    """Test annual fee estimation for different strategies."""
    # DCA on River (0% fee)
    annual_fee = estimate_annual_fees("river", "dca", 100)
    assert annual_fee == 0.0
    
    # DCA on Swan (0.99% fee)
    annual_fee = estimate_annual_fees("swan", "dca", 100)
    assert annual_fee > 0.0
    
    # MACO should have fewer transactions than DCA
    dca_fee = estimate_annual_fees("coinbase", "dca", 100)
    maco_fee = estimate_annual_fees("coinbase", "maco", 100)
    assert maco_fee < dca_fee
    
    # RSI should have more transactions than DCA
    rsi_fee = estimate_annual_fees("coinbase", "rsi", 100)
    assert rsi_fee > dca_fee

def test_adjust_btc_for_fees():
    """Test BTC adjustment for withdrawal fees."""
    # No withdrawal, no adjustment
    assert adjust_btc_for_fees(1.0, "binance", withdrawal=False) == 1.0
    
    # Withdrawal from Binance (should subtract withdrawal fee)
    binance_withdrawal = adjust_btc_for_fees(1.0, "binance", withdrawal=True)
    assert binance_withdrawal < 1.0
    
    # Withdrawal from River (no withdrawal fee)
    river_withdrawal = adjust_btc_for_fees(1.0, "river", withdrawal=True)
    assert river_withdrawal == 1.0
    
    # Edge case: withdrawal amount less than fee
    small_amount = adjust_btc_for_fees(0.00001, "binance", withdrawal=True)
    assert small_amount == 0.00001  # Should not subtract if amount is too small