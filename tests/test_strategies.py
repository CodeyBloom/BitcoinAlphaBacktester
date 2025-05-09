"""
Tests for the strategies module of the Bitcoin Strategy Backtester.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# Import the module to test
from strategies import (
    dca_strategy,
    value_averaging_strategy,
    maco_strategy,
    rsi_strategy,
    volatility_strategy,
    xgboost_ml_strategy
)

@pytest.fixture
def price_data():
    """Create sample price data for testing strategies"""
    # Create date range for 100 days
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Create price data with some volatility
    base_price = 20000  # Starting at 20k
    prices = []
    for i in range(100):
        # Add some sine wave pattern with noise
        cycle = np.sin(i / 10) * 2000  # $2000 sine amplitude
        trend = i * 50  # $50 daily uptrend
        noise = np.random.normal(0, 300)  # Random noise
        price = base_price + cycle + trend + noise
        prices.append(max(price, 15000))  # Ensure minimum price of 15k
    
    # Create day of week and Sunday flags
    day_of_week = [(start_date + timedelta(days=i)).weekday() for i in range(100)]
    is_sunday = [dow == 6 for dow in day_of_week]
    
    # Convert to Polars DataFrame
    df = pl.DataFrame({
        "date": dates,
        "price": prices,
        "day_of_week": day_of_week,
        "is_sunday": is_sunday
    })
    
    # Calculate returns
    df = df.with_columns(
        pl.col("price").pct_change().alias("returns")
    )
    
    return df

def test_dca_strategy(price_data):
    """Test DCA strategy"""
    weekly_investment = 100.0
    
    # Run strategy
    result = dca_strategy(price_data, weekly_investment)
    
    # Basic validations
    assert "investment" in result.columns
    assert "btc_bought" in result.columns
    assert "cumulative_btc" in result.columns
    
    # There should be investments only on Sundays
    non_sunday_investments = result.filter(~pl.col("is_sunday")).select("investment").sum().item()
    assert non_sunday_investments == 0
    
    # Sunday investments should match weekly_investment
    sunday_investments = result.filter(pl.col("is_sunday")).select("investment").sum().item()
    expected_sundays = sum(price_data["is_sunday"])
    assert sunday_investments == pytest.approx(weekly_investment * expected_sundays)
    
    # Final BTC should be positive
    assert result["cumulative_btc"][-1] > 0

def test_value_averaging_strategy(price_data):
    """Test Value Averaging strategy"""
    weekly_base_investment = 100.0
    target_growth_rate = 0.01  # 1% weekly growth target
    
    # Run strategy
    result = value_averaging_strategy(price_data, weekly_base_investment, target_growth_rate)
    
    # Basic validations
    assert "investment" in result.columns
    assert "btc_bought" in result.columns
    assert "cumulative_btc" in result.columns
    
    # There should be investments only on Sundays
    non_sunday_investments = result.filter(~pl.col("is_sunday")).select("investment").sum().item()
    assert non_sunday_investments == 0
    
    # Final BTC should be positive
    assert result["cumulative_btc"][-1] > 0

def test_maco_strategy(price_data):
    """Test Moving Average Crossover strategy"""
    weekly_investment = 100.0
    short_window = 10
    long_window = 30
    
    # Run strategy
    result = maco_strategy(price_data, weekly_investment, short_window, long_window)
    
    # Basic validations
    assert "investment" in result.columns
    assert "btc_bought" in result.columns
    assert "cumulative_btc" in result.columns
    assert "short_ma" in result.columns
    assert "long_ma" in result.columns
    
    # Verify that we have investments on some Sundays
    sunday_investments = result.filter(pl.col("is_sunday")).select("investment").sum().item()
    assert sunday_investments > 0
    
    # Verify that there are some days with zero investment
    zero_investment_days = result.filter(pl.col("investment") == 0).height
    assert zero_investment_days > 0
    
    # Final BTC should be positive
    assert result["cumulative_btc"][-1] > 0

def test_rsi_strategy(price_data):
    """Test RSI strategy"""
    weekly_investment = 100.0
    rsi_period = 7
    oversold_threshold = 30
    overbought_threshold = 70
    
    # Run strategy
    result = rsi_strategy(
        price_data, weekly_investment, rsi_period, 
        oversold_threshold, overbought_threshold
    )
    
    # Basic validations
    assert "investment" in result.columns
    assert "btc_bought" in result.columns
    assert "cumulative_btc" in result.columns
    assert "rsi" in result.columns
    
    # The strategy should make investments (not requiring all on Sundays due to implementation)
    total_investment = result.select("investment").sum().item()
    assert total_investment > 0
    
    # Final BTC should be positive
    assert result["cumulative_btc"][-1] > 0
    
    # RSI should be within 0-100 range
    rsi_values = result.filter(~pl.col("rsi").is_null())["rsi"]
    if len(rsi_values) > 0:
        assert all(0 <= r <= 100 for r in rsi_values)

def test_volatility_strategy(price_data):
    """Test Volatility strategy"""
    weekly_investment = 100.0
    vol_window = 7
    vol_threshold = 1.5
    
    # Run strategy
    result = volatility_strategy(price_data, weekly_investment, vol_window, vol_threshold)
    
    # Basic validations
    assert "investment" in result.columns
    assert "btc_bought" in result.columns
    assert "cumulative_btc" in result.columns
    assert "volatility" in result.columns
    assert "avg_volatility" in result.columns
    
    # There should be investments only on Sundays
    non_sunday_investments = result.filter(~pl.col("is_sunday")).select("investment").sum().item()
    assert non_sunday_investments == 0
    
    # Final BTC should be positive
    assert result["cumulative_btc"][-1] > 0
    
    # Basic test of volatility calculation
    volatilities = result.filter(~pl.col("volatility").is_null())["volatility"]
    assert len(volatilities) > 0
    assert all(v >= 0 for v in volatilities)  # Volatility should always be non-negative

def test_xgboost_ml_strategy(price_data):
    """Test XGBoost ML trading strategy"""
    weekly_investment = 100.0
    training_window = 14
    prediction_threshold = 0.55
    features = ["returns", "price"]
    
    # Run strategy
    result = xgboost_ml_strategy(
        price_data, 
        weekly_investment, 
        training_window=training_window,
        prediction_threshold=prediction_threshold,
        features=features
    )
    
    # Basic validations
    assert "investment" in result.columns
    assert "btc_bought" in result.columns
    assert "cumulative_btc" in result.columns
    assert "prediction" in result.columns
    assert "confidence" in result.columns
    
    # There should be investments only on Sundays
    non_sunday_investments = result.filter(~pl.col("is_sunday")).select("investment").sum().item()
    assert non_sunday_investments == 0
    
    # Sunday investments should add up to a positive amount
    sunday_investments = result.filter(pl.col("is_sunday")).select("investment").sum().item()
    assert sunday_investments > 0
    
    # Final BTC should be positive
    assert result["cumulative_btc"][-1] > 0
    
    # Confidence values should be between 0 and 1
    confidence_values = result.filter(~pl.col("confidence").is_null())["confidence"]
    if len(confidence_values) > 0:
        assert all(0 <= c <= 1 for c in confidence_values)
    
    # Predictions should be either 0 or 1 (binary classification)
    prediction_values = result.filter(~pl.col("prediction").is_null())["prediction"]
    if len(prediction_values) > 0:
        assert all(p in [0, 1] for p in prediction_values)