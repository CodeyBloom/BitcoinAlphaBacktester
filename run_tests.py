"""
Test runner for the Bitcoin Strategy Backtester.

This script runs all tests and generates a coverage report.
"""

import sys
import pytest

if __name__ == "__main__":
    # Run tests with coverage
    args = [
        "--cov=domain",
        "--cov=data_fetcher_new",
        "--cov=strategies_new",
        "-v",
        "test_domain.py",
        "test_data_fetcher.py"
    ]
    
    # Add any command line arguments
    args.extend(sys.argv[1:])
    
    # Run pytest
    sys.exit(pytest.main(args))