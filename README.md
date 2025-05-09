# Bitcoin Strategy Backtester

A sophisticated cryptocurrency strategy backtesting tool that enables comprehensive performance analysis and comparison of multiple investment strategies using advanced metrics and fee modeling.

## Key Features

- Multiple bitcoin investment strategies (DCA, Moving Average, RSI, Volatility)
- Detailed performance metrics with focus on efficiency (BTC per currency unit spent)
- Robust testing framework with extensive test coverage
- Support for multiple exchange fee models
- Streamlit-based interactive web interface
- Automated data updates and optimization via GitHub Actions

## Strategies

The backtester includes several investment strategies:

1. **Dollar Cost Averaging (DCA)**: Regular purchases of a fixed currency amount.
2. **Moving Average Crossover (MACO)**: Buy/sell based on short-term vs. long-term moving average crossovers.
3. **RSI-Based Strategy**: Adjust investment based on Relative Strength Index.
4. **Volatility-Based Strategy**: Adjust investment based on price volatility.

## Viewing Strategy Performance

The application provides two main ways to analyze Bitcoin investment strategies:

1. **Optimized Strategies**: View pre-calculated optimal parameters for each strategy over 1, 5, or 10-year periods.
2. **Custom Backtest** (coming soon): Set your own parameters for each strategy and compare against optimized ones.

## Maintaining Up-to-Date Data with GitHub Actions

This project uses GitHub Actions to automate data updates and strategy optimizations. The automated workflow:

1. Fetches the latest Bitcoin price data from CoinGecko
2. Runs optimizations for all strategies over predefined time periods
3. Saves results to the repository
4. Runs weekly on Mondays at 2:00 UTC

### Setting Up GitHub Actions

The workflow is already configured in `.github/workflows/update_bitcoin_data.yml`. When you push this project to GitHub, you'll need to:

1. Make sure the repository has proper permissions for GitHub Actions:
   - Go to your repository on GitHub
   - Navigate to Settings > Actions > General
   - Under "Workflow permissions," select "Read and write permissions"

2. (Optional) Configure CoinGecko API Key:
   - If you have a CoinGecko API key, add it as a repository secret
   - Go to Settings > Secrets and variables > Actions
   - Click "New repository secret"
   - Name: `COINGECKO_API_KEY`
   - Value: Your CoinGecko API key

3. (Optional) Adjust the schedule:
   - Edit `.github/workflows/update_bitcoin_data.yml`
   - Modify the `cron` expression under `schedule`

### Manual Triggering

You can also trigger the data update and optimization process manually:

1. Go to the repository on GitHub
2. Navigate to the "Actions" tab
3. Select the "Update Bitcoin Data and Optimizations" workflow
4. Click "Run workflow"

## Development Setup

To set up the project for local development:

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```
   
   Note: For Streamlit Cloud deployment, the port is automatically configured.

## Streamlit Cloud Deployment

To deploy this application to Streamlit Cloud:

1. Push this repository to GitHub (make sure it's public for the free tier)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch (main), and main file (app.py)
6. Click "Deploy"

Your app will be available at `https://[your-name]-[repo-name].streamlit.app`

## Project Structure

- `app.py`: Main Streamlit application entry point
- `data_fetcher.py`: Functions for fetching Bitcoin price data
- `domain.py`: Core domain logic for strategies
- `fee_models.py`: Exchange fee calculations
- `metrics.py`: Performance metrics calculations
- `optimize_app.py`: Optimization results view
- `scripts/`: Maintenance scripts for GitHub Actions
- `data/`: Contains Bitcoin price data and optimization results
- `.github/workflows/`: GitHub Actions workflow definitions