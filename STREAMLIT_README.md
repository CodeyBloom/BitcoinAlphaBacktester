# Bitcoin Strategy Backtester

A sophisticated cryptocurrency strategy backtesting platform that transforms complex investment analysis into an intuitive, user-friendly experience through advanced data visualization and optimization techniques.

## Features

- **Backtest Multiple Strategies**: Compare various Bitcoin investment strategies including DCA, Moving Average Crossover, RSI-based, Volatility-based, and XGBoost ML.
- **Performance Metrics**: Analyze key metrics such as BTC accumulation, efficiency (BTC per currency unit), maximum drawdown, and Sortino ratio.
- **Optimization Results**: View pre-calculated optimization results for different time periods.
- **Exchange Fee Comparison**: Account for different exchange fees to see their impact on performance.
- **Interactive Visualizations**: Explore strategy performance through interactive charts.

## Running the App

### Local Development

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

### Deployment

This application is optimized for deployment on Streamlit Cloud.

## Data Updates

Bitcoin price data and strategy optimizations are automatically updated through GitHub Actions on a weekly basis.

## Technical Architecture

The application uses:
- Python with Streamlit for the interactive UI
- Pandas and Polars for data processing
- Plotly for interactive visualizations
- XGBoost for machine learning strategies
- GitHub Actions for automatic data updates

## License

MIT License