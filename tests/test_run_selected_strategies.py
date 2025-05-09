"""
Tests for the run_selected_strategies function that follows TDD principles.

These tests define the expected behavior of a function that doesn't exist yet.
The function will be responsible for running selected strategies on price data.
"""

import pytest
import polars as pl
from datetime import datetime, timedelta

# We'll define a function that doesn't exist yet, but should be implemented
# to make these tests pass

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

def test_run_selected_strategies_all_selected():
    """
    Test that the function runs all selected strategies correctly.
    This test defines the expected interface and behavior.
    """
    # Import the function that doesn't exist yet, but will be implemented
    from app import run_selected_strategies
    
    # Create sample data with more days to ensure strategies have enough data
    start_date = datetime(2023, 1, 1)  # This is a Sunday
    dates = [start_date + timedelta(days=i) for i in range(120)]  # Extended to 120 days
    
    # Create price data with some volatility to trigger strategy signals
    prices = []
    base_price = 20000.0  # Make sure we're using float
    for i in range(120):
        # Add some volatility
        if i > 0:
            # Add trend and volatility
            change = base_price * 0.01 * (0.5 - (i % 10) / 10.0)  # Oscillating price changes
            base_price += change
        prices.append(float(base_price))  # Ensure it's a float
    
    day_of_week = [(start_date + timedelta(days=i)).weekday() for i in range(120)]
    
    # Make Jan 1, 2023 and every 7th day a Sunday
    is_sunday = [
        True if i % 7 == 0 else False
        for i in range(120)
    ]
    
    # Calculate returns
    returns = [0.0]
    for i in range(1, 120):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    
    df = pl.DataFrame({
        "date": dates,
        "price": prices,
        "day_of_week": day_of_week,
        "is_sunday": is_sunday,
        "returns": returns
    })
    
    # Print some info about the test data
    sundays_count = sum(is_sunday)
    print(f"Test data: {df.height} days, {sundays_count} Sundays")
    print(f"First few prices: {prices[:10]}")
    print(f"Last few prices: {prices[-10:]}")
    
    # Define selected strategies
    strategy_selections = {
        "dca": True,         # Always run as baseline
        "value_avg": True,   # Selected
        "maco": True,        # Selected
        "rsi": True,         # Selected
        "volatility": True   # Selected
    }
    
    # Define strategy parameters
    strategy_params = {
        "value_avg": {"target_growth_rate": 0.01},
        "maco": {"short_window": 10, "long_window": 20},
        "rsi": {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70},
        "volatility": {"vol_window": 14, "vol_threshold": 1.5}
    }
    
    # Run the function
    weekly_investment = 100.0
    exchange_id = None
    use_exchange_discount = False
    
    results, metrics = run_selected_strategies(
        df, 
        strategy_selections, 
        strategy_params, 
        weekly_investment,
        exchange_id,
        use_exchange_discount
    )
    
    # Print debug information
    print("\nResults keys:", list(results.keys()))
    print("\nMetrics keys:", list(metrics.keys()))
    
    for strategy, metric in metrics.items():
        print(f"\n{strategy} final BTC: {metric['final_btc']}")
    
    for strategy, result_df in results.items():
        sunday_investments = result_df.filter(pl.col("is_sunday"))["investment"].sum()
        print(f"{strategy} total investment on Sundays: {sunday_investments}")
        btc_bought = result_df.filter(pl.col("is_sunday"))["btc_bought"].sum()
        print(f"{strategy} total BTC bought on Sundays: {btc_bought}")
    
    # Verify the results
    assert "DCA (Baseline)" in results
    assert "Value Averaging" in results
    assert "MACO" in results
    assert "RSI" in results
    assert "Volatility" in results
    
    # Verify each result has expected structure
    for result in results.values():
        assert "cumulative_btc" in result.columns
        assert "investment" in result.columns
        assert "btc_bought" in result.columns
    
    # Verify the metrics
    assert "DCA (Baseline)" in metrics
    assert "Value Averaging" in metrics
    assert "MACO" in metrics
    assert "RSI" in metrics
    assert "Volatility" in metrics
    
    # Verify each metric has expected fields
    for metric in metrics.values():
        assert "final_btc" in metric
        assert "max_drawdown" in metric
        assert "sortino_ratio" in metric
        assert metric["final_btc"] > 0  # Should have bought some BTC

def test_run_selected_strategies_subset_selected():
    """
    Test that the function runs only the selected strategies.
    """
    # Import the function that doesn't exist yet, but will be implemented
    from app import run_selected_strategies
    
    # Create sample data with more days to ensure strategies have enough data
    start_date = datetime(2023, 1, 1)  # This is a Sunday
    dates = [start_date + timedelta(days=i) for i in range(120)]  # Extended to 120 days
    
    # Create price data with some volatility to trigger strategy signals
    prices = []
    base_price = 20000.0  # Make sure we're using float
    for i in range(120):
        # Add some volatility
        if i > 0:
            # Add trend and volatility
            change = base_price * 0.01 * (0.5 - (i % 10) / 10.0)  # Oscillating price changes
            base_price += change
        prices.append(float(base_price))  # Ensure it's a float
    
    day_of_week = [(start_date + timedelta(days=i)).weekday() for i in range(120)]
    
    # Make Jan 1, 2023 and every 7th day a Sunday
    is_sunday = [
        True if i % 7 == 0 else False
        for i in range(120)
    ]
    
    # Calculate returns
    returns = [0.0]
    for i in range(1, 120):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    
    df = pl.DataFrame({
        "date": dates,
        "price": prices,
        "day_of_week": day_of_week,
        "is_sunday": is_sunday,
        "returns": returns
    })
    
    # Define selected strategies (only some selected)
    strategy_selections = {
        "dca": True,         # Always run as baseline
        "value_avg": False,  # Not selected
        "maco": True,        # Selected
        "rsi": False,        # Not selected
        "volatility": True   # Selected
    }
    
    # Define strategy parameters
    strategy_params = {
        "value_avg": {"target_growth_rate": 0.01},
        "maco": {"short_window": 10, "long_window": 20},
        "rsi": {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70},
        "volatility": {"vol_window": 14, "vol_threshold": 1.5}
    }
    
    # Run the function
    weekly_investment = 100.0
    exchange_id = None
    use_exchange_discount = False
    
    results, metrics = run_selected_strategies(
        df, 
        strategy_selections, 
        strategy_params, 
        weekly_investment,
        exchange_id,
        use_exchange_discount
    )
    
    # Verify the results (only selected strategies)
    assert "DCA (Baseline)" in results
    assert "Value Averaging" not in results
    assert "MACO" in results
    assert "RSI" not in results
    assert "Volatility" in results
    
    # Verify the metrics (only selected strategies)
    assert "DCA (Baseline)" in metrics
    assert "Value Averaging" not in metrics
    assert "MACO" in metrics
    assert "RSI" not in metrics
    assert "Volatility" in metrics

def test_run_selected_strategies_with_exchange():
    """
    Test that the function correctly applies exchange fees.
    """
    # Import the function that doesn't exist yet, but will be implemented
    from app import run_selected_strategies
    
    # Create sample data with more days to ensure strategies have enough data
    start_date = datetime(2023, 1, 1)  # This is a Sunday
    dates = [start_date + timedelta(days=i) for i in range(120)]  # Extended to 120 days
    
    # Create price data with some volatility to trigger strategy signals
    prices = []
    base_price = 20000.0  # Make sure we're using float
    for i in range(120):
        # Add some volatility
        if i > 0:
            # Add trend and volatility
            change = base_price * 0.01 * (0.5 - (i % 10) / 10.0)  # Oscillating price changes
            base_price += change
        prices.append(float(base_price))  # Ensure it's a float
    
    day_of_week = [(start_date + timedelta(days=i)).weekday() for i in range(120)]
    
    # Make Jan 1, 2023 and every 7th day a Sunday
    is_sunday = [
        True if i % 7 == 0 else False
        for i in range(120)
    ]
    
    # Calculate returns
    returns = [0.0]
    for i in range(1, 120):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    
    df = pl.DataFrame({
        "date": dates,
        "price": prices,
        "day_of_week": day_of_week,
        "is_sunday": is_sunday,
        "returns": returns
    })
    
    # Define selected strategies
    strategy_selections = {
        "dca": True,
        "value_avg": False,
        "maco": False,
        "rsi": False,
        "volatility": False
    }
    
    # Define strategy parameters
    strategy_params = {}
    
    # Run the function with exchange
    weekly_investment = 100.0
    exchange_id = "binance"  # Use an exchange
    use_exchange_discount = True
    
    results_with_exchange, metrics_with_exchange = run_selected_strategies(
        df, 
        strategy_selections, 
        strategy_params, 
        weekly_investment,
        exchange_id,
        use_exchange_discount
    )
    
    # Run again without exchange
    exchange_id = None  # No exchange
    results_no_exchange, metrics_no_exchange = run_selected_strategies(
        df, 
        strategy_selections, 
        strategy_params, 
        weekly_investment,
        exchange_id,
        use_exchange_discount
    )
    
    # Verify the exchange fees affect the results
    dca_with_exchange = metrics_with_exchange["DCA (Baseline)"]["final_btc"]
    dca_no_exchange = metrics_no_exchange["DCA (Baseline)"]["final_btc"]
    
    # With exchange fees, we should get less BTC
    assert dca_with_exchange < dca_no_exchange
    
def test_run_selected_strategies_no_data():
    """
    Test that the function handles empty data gracefully.
    """
    # Import the function that doesn't exist yet, but will be implemented
    from app import run_selected_strategies
    
    # Create empty DataFrame
    df = pl.DataFrame({
        "date": [],
        "price": [],
        "day_of_week": [],
        "is_sunday": [],
        "returns": []
    })
    
    # Define selected strategies
    strategy_selections = {
        "dca": True,
        "value_avg": True,
        "maco": True,
        "rsi": True,
        "volatility": True
    }
    
    # Define strategy parameters
    strategy_params = {
        "value_avg": {"target_growth_rate": 0.01},
        "maco": {"short_window": 10, "long_window": 20},
        "rsi": {"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70},
        "volatility": {"vol_window": 14, "vol_threshold": 1.5}
    }
    
    # Run the function
    weekly_investment = 100.0
    exchange_id = None
    use_exchange_discount = False
    
    results, metrics = run_selected_strategies(
        df, 
        strategy_selections, 
        strategy_params, 
        weekly_investment,
        exchange_id,
        use_exchange_discount
    )
    
    # Verify we got empty but properly structured results
    assert isinstance(results, dict)
    assert isinstance(metrics, dict)
    
    # No results should be present for empty data
    assert len(results) == 0
    assert len(metrics) == 0