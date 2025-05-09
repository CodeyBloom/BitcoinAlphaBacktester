# Bitcoin Strategy Backtester Data Maintenance Guide

This document provides instructions on how the Bitcoin Strategy Backtester maintains up-to-date data through automated GitHub Actions workflows.

## Overview

The system automatically maintains:
1. Bitcoin historical price data (10 years of daily prices)
2. Optimization results for all strategies and time periods (1, 5, and 10 years)
3. Data integrity verification

## Automated Data Update Process

The data update process is fully automated via GitHub Actions and runs:
- Weekly (every Monday at 2:00 UTC)
- On-demand via manual workflow dispatch

### What the Process Does

1. **Fetch Bitcoin Price Data**:
   - Fetches 10 years of historical BTC/AUD price data from CryptoCompare API
   - Saves to `data/bitcoin_prices.arrow`
   - Contains robust error handling and retry mechanisms

2. **Generate Optimization Results**:
   - Creates baseline sample optimization files as a fallback
   - Runs real optimizations using actual historical data
   - Runs separate optimizations for each time period (1, 5, 10 years) to manage complexity
   - Saves results to `data/optimizations/` directory

3. **Verify Data Integrity**:
   - Checks completeness of Bitcoin price data
   - Verifies all optimization files exist and are valid
   - Saves verification results to `data/last_verification.json`

4. **Commit Changes**:
   - Automatically commits and pushes updates to the repository

## Key Files

- **Data Files**:
  - `data/bitcoin_prices.arrow`: Bitcoin historical price data
  - `data/optimizations/*.arrow`: Strategy optimization results
  - `data/last_verification.json`: Latest data integrity verification results

- **Scripts**:
  - `scripts/fetch_cryptocompare_data.py`: Fetches Bitcoin price data
  - `scripts/generate_optimizations_for_periods.py`: Generates sample optimization files
  - `scripts/run_real_optimizations.py`: Runs real optimizations using historical data
  - `scripts/verify_data_integrity.py`: Verifies data integrity

- **Workflow Definition**:
  - `.github/workflows/update_bitcoin_data.yml`: GitHub Actions workflow definition

## Manual Data Update

To manually trigger the data update workflow:
1. Go to the GitHub repository
2. Navigate to the "Actions" tab
3. Select the "Update Bitcoin Data and Optimizations" workflow
4. Click "Run workflow" button
5. Select the branch to run on (usually "main")
6. Click "Run workflow" to start the process

## Troubleshooting

If the data update fails:

1. **Check GitHub Actions logs** for specific error details
2. Common issues and solutions:
   - API rate limits: Wait and retry later
   - Network issues: Check connectivity to CryptoCompare API
   - Missing data: Verify CryptoCompare API is returning expected data
   - Failed optimizations: Check logs for specific strategy failure reasons

3. **Verification Results**: Check `data/last_verification.json` for detailed information about data integrity issues

## Implementation Guide for GitHub Actions

To implement the GitHub Actions workflow:

1. Ensure your GitHub repository has Actions enabled
2. The `.github/workflows/update_bitcoin_data.yml` file defines the workflow
3. Required secrets: None (uses public CryptoCompare API)
4. Required permissions: Write access to repository for commits

The workflow is designed to be fully autonomous and will:
- Run on schedule without manual intervention
- Maintain data currency over time (days, weeks, years)
- Provide clear logs and verification results
- Handle errors gracefully with reasonable fallbacks

## Future Improvements

Consider these potential enhancements:
- Email notifications on workflow failures
- Dashboard for data freshness monitoring
- Multiple API fallbacks if primary data source fails
- Automated testing of generated optimizations