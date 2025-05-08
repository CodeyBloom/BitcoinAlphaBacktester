"""
Fee models for different cryptocurrency exchanges.

This module manages the fee structures for various exchanges and provides functions
to calculate transaction costs for different strategies. The fee data is structured 
to be easily updated via GitHub Actions.

GitHub Actions can be configured to:
1. Periodically scrape exchange websites for updated fee information
2. Update the EXCHANGE_PROFILES dictionary in an external JSON file
3. Run tests to verify the fee calculations
4. Commit and push the updated data to the repository
"""

import json
import os
from datetime import datetime
from enum import Enum

# Define transaction types
class TransactionType(Enum):
    BUY = "buy"
    SELL = "sell"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"

# Path to exchange fee data JSON
EXCHANGE_DATA_PATH = "data/exchange_fees.json"

# Default exchange profiles if the file doesn't exist
DEFAULT_EXCHANGE_PROFILES = {
    "river": {
        "name": "River Financial",
        "url": "https://river.com",
        "dca_fee": 0.0,  # River offers fee-free recurring buys
        "standard_fee": 0.0025,  # 0.25% for regular transactions
        "withdrawal_fee": 0.0,  # Free withdrawals
        "min_purchase": 25,  # Minimum purchase amount in USD
        "supported_currencies": ["USD"],
        "last_updated": datetime.now().isoformat()
    },
    "binance": {
        "name": "Binance",
        "url": "https://binance.com",
        "maker_fee": 0.001,  # 0.1% maker fee
        "taker_fee": 0.001,  # 0.1% taker fee
        "bnb_discount": 0.25,  # 25% discount when paying fees with BNB
        "withdrawal_fee_btc": 0.0005,  # BTC withdrawal fee
        "min_purchase": 10,  # Minimum purchase amount in USD
        "supported_currencies": ["USD", "USDT", "BUSD"],
        "last_updated": datetime.now().isoformat()
    },
    "kraken": {
        "name": "Kraken",
        "url": "https://kraken.com",
        "tier1_fee": 0.0026,  # Tier 1 fee (lower volume)
        "tier2_fee": 0.0024,  # Tier 2 fee (medium volume)
        "tier3_fee": 0.0020,  # Tier 3 fee (higher volume)
        "withdrawal_fee_btc": 0.00015,  # BTC withdrawal fee
        "min_purchase": 10,  # Minimum purchase amount in USD
        "supported_currencies": ["USD", "EUR", "GBP", "AUD"],
        "last_updated": datetime.now().isoformat()
    },
    "coinbase": {
        "name": "Coinbase",
        "url": "https://coinbase.com",
        "standard_fee": 0.006,  # 0.6% standard fee
        "coinbase_one_fee": 0.0,  # Coinbase One subscription offers zero fees
        "withdrawal_fee": 0.0,  # Network fees apply
        "min_purchase": 2,  # Minimum purchase amount in USD
        "supported_currencies": ["USD", "EUR", "GBP", "AUD"],
        "last_updated": datetime.now().isoformat()
    },
    "swan": {
        "name": "Swan Bitcoin",
        "url": "https://swanbitcoin.com",
        "dca_fee": 0.0099,  # 0.99% for DCA purchases
        "withdrawal_fee": 0.0,  # Free withdrawals
        "min_purchase": 10,  # Minimum purchase amount in USD
        "supported_currencies": ["USD"],
        "last_updated": datetime.now().isoformat()
    }
}

def load_exchange_profiles():
    """
    Load exchange profiles from the JSON file.
    If the file doesn't exist, create it with default values.
    
    Returns:
        dict: Dictionary of exchange profiles
    """
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
            
        if not os.path.exists(EXCHANGE_DATA_PATH):
            # Create the file with default profiles
            with open(EXCHANGE_DATA_PATH, 'w') as f:
                json.dump(DEFAULT_EXCHANGE_PROFILES, f, indent=4)
            return DEFAULT_EXCHANGE_PROFILES
        
        # Load the existing file
        with open(EXCHANGE_DATA_PATH, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        print(f"Error loading exchange profiles: {e}")
        return DEFAULT_EXCHANGE_PROFILES

def save_exchange_profiles(profiles):
    """
    Save exchange profiles to the JSON file.
    
    Args:
        profiles (dict): Dictionary of exchange profiles to save
    """
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
            
        with open(EXCHANGE_DATA_PATH, 'w') as f:
            json.dump(profiles, f, indent=4)
            
    except Exception as e:
        print(f"Error saving exchange profiles: {e}")

def get_exchange_fee(exchange_id, transaction_type, amount=None, volume_tier="tier1", use_discount=False):
    """
    Calculate the fee for a transaction based on exchange and type.
    
    Args:
        exchange_id (str): Exchange identifier (e.g., "binance")
        transaction_type (TransactionType): Type of transaction
        amount (float, optional): Transaction amount
        volume_tier (str, optional): Volume tier for exchanges with tiered fees
        use_discount (bool, optional): Whether to apply available discounts
        
    Returns:
        float: Fee amount as a decimal percentage
    """
    profiles = load_exchange_profiles()
    
    if exchange_id not in profiles:
        raise ValueError(f"Unknown exchange: {exchange_id}")
    
    exchange = profiles[exchange_id]
    
    # DCA purchase on River has no fee
    if exchange_id == "river" and transaction_type == TransactionType.BUY:
        return exchange.get("dca_fee", 0.0)
    
    # Coinbase with Coinbase One subscription
    if exchange_id == "coinbase" and use_discount:
        return exchange.get("coinbase_one_fee", 0.0)
    
    # Binance with potential BNB discount
    if exchange_id == "binance":
        base_fee = exchange.get("maker_fee" if transaction_type == TransactionType.BUY else "taker_fee", 0.001)
        if use_discount:
            discount = exchange.get("bnb_discount", 0.0)
            return base_fee * (1 - discount)
        return base_fee
    
    # Kraken with tiered pricing
    if exchange_id == "kraken":
        if volume_tier == "tier2":
            return exchange.get("tier2_fee", 0.0024)
        elif volume_tier == "tier3":
            return exchange.get("tier3_fee", 0.0020)
        else:  # Default to tier1
            return exchange.get("tier1_fee", 0.0026)
    
    # Swan Bitcoin
    if exchange_id == "swan" and transaction_type == TransactionType.BUY:
        return exchange.get("dca_fee", 0.0099)
    
    # Default to standard fee if available, otherwise 0.5%
    return exchange.get("standard_fee", 0.005)

def calculate_transaction_cost(amount, exchange_id, transaction_type, volume_tier="tier1", use_discount=False):
    """
    Calculate the cost of a transaction including fees.
    
    Args:
        amount (float): Transaction amount
        exchange_id (str): Exchange identifier
        transaction_type (TransactionType): Type of transaction
        volume_tier (str, optional): Volume tier for exchanges with tiered fees
        use_discount (bool, optional): Whether to apply available discounts
        
    Returns:
        tuple: (net_amount, fee_amount)
    """
    fee_percentage = get_exchange_fee(exchange_id, transaction_type, amount, volume_tier, use_discount)
    fee_amount = amount * fee_percentage
    net_amount = amount - fee_amount
    
    return net_amount, fee_amount

def get_optimal_exchange_for_strategy(strategy_type, weekly_investment, currency="USD"):
    """
    Determine the most cost-effective exchange for a given strategy.
    
    Args:
        strategy_type (str): Type of strategy ("dca", "value_avg", "maco", "rsi", "volatility")
        weekly_investment (float): Weekly investment amount
        currency (str, optional): Currency to use
        
    Returns:
        tuple: (exchange_id, estimated_annual_fee)
    """
    profiles = load_exchange_profiles()
    optimal_exchange = None
    lowest_annual_fee = float('inf')
    
    for exchange_id, profile in profiles.items():
        # Skip if the exchange doesn't support the currency
        if currency not in profile.get("supported_currencies", []):
            continue
            
        # Skip if minimum purchase is higher than weekly investment
        if profile.get("min_purchase", 0) > weekly_investment:
            continue
        
        # Calculate estimated annual fees based on strategy type
        annual_fee = estimate_annual_fees(exchange_id, strategy_type, weekly_investment)
        
        if annual_fee < lowest_annual_fee:
            lowest_annual_fee = annual_fee
            optimal_exchange = exchange_id
    
    return optimal_exchange, lowest_annual_fee

def estimate_annual_fees(exchange_id, strategy_type, weekly_investment):
    """
    Estimate annual fees for a given exchange and strategy.
    
    Args:
        exchange_id (str): Exchange identifier
        strategy_type (str): Type of strategy
        weekly_investment (float): Weekly investment amount
        
    Returns:
        float: Estimated annual fee in currency units
    """
    annual_investment = weekly_investment * 52
    
    # Estimated number of transactions per year for each strategy
    if strategy_type == "dca":
        transactions = 52  # Once per week
    elif strategy_type == "value_avg":
        transactions = 52  # Weekly adjustments
    elif strategy_type == "maco":
        transactions = 26  # Estimated number of crossovers per year
    elif strategy_type == "rsi":
        transactions = 78  # More frequent due to oscillations
    elif strategy_type == "volatility":
        transactions = 104  # Highest frequency due to volatility triggers
    else:
        transactions = 52  # Default to weekly
    
    # Average transaction amount
    avg_transaction = annual_investment / transactions
    
    # Calculate fee for a typical transaction
    _, fee_per_transaction = calculate_transaction_cost(
        avg_transaction, 
        exchange_id, 
        TransactionType.BUY
    )
    
    return fee_per_transaction * transactions

# GitHub Actions can update this function to fetch current fees
def update_exchange_fees():
    """
    Update exchange fees from their respective APIs or websites.
    This function is designed to be called by GitHub Actions.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        profiles = load_exchange_profiles()
        
        # In a real implementation, this would scrape or use APIs
        # to get the latest fee information for each exchange
        
        # For now, just update the last_updated timestamp
        for exchange in profiles:
            profiles[exchange]["last_updated"] = datetime.now().isoformat()
        
        save_exchange_profiles(profiles)
        return True
    except Exception as e:
        print(f"Error updating exchange fees: {e}")
        return False

def adjust_btc_for_fees(btc_amount, exchange_id, withdrawal=False):
    """
    Adjust BTC amount for withdrawal fees if needed.
    
    Args:
        btc_amount (float): Original BTC amount
        exchange_id (str): Exchange identifier
        withdrawal (bool): Whether to include withdrawal fee
        
    Returns:
        float: Adjusted BTC amount
    """
    if not withdrawal:
        return btc_amount
        
    profiles = load_exchange_profiles()
    
    if exchange_id not in profiles:
        return btc_amount
        
    exchange = profiles[exchange_id]
    
    # Apply withdrawal fee if available
    withdrawal_fee = exchange.get("withdrawal_fee_btc", 0)
    
    # Only subtract if we have enough BTC
    if btc_amount > withdrawal_fee:
        return btc_amount - withdrawal_fee
    else:
        return btc_amount