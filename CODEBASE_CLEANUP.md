# Codebase Cleanup Recommendations

After an extensive review of the codebase, here are recommendations for cleaning up redundant or outdated files to keep the repository lean and efficient.

## Files That Can Be Safely Removed

### 1. Deprecated Data Fetching Scripts

These scripts have been replaced by `fetch_cryptocompare_data.py` and are no longer used in the GitHub Actions workflow:

- `scripts/fetch_10yr_bitcoin_data.py`: Old Bitcoin data fetching script
- `scripts/update_bitcoin_data.py`: Outdated data update script

### 2. Redundant Optimization Scripts

Since we now have a comprehensive `run_real_optimizations.py` script:

- `scripts/run_all_optimizations.py`: Functionality is now covered by `run_real_optimizations.py`

### 3. Sample Data No Longer Needed

If you have sample or test data files that aren't used in production:

- Any sample CSV files in `data/` directory
- Backup or duplicate optimization files

## Safe Removal Process

1. Before deleting files, ensure they're not referenced elsewhere:
   ```bash
   grep -r "filename.py" .
   ```

2. Add files to be removed to your `.gitignore` if you want to keep them locally but not in the repository

3. Commit the removal:
   ```bash
   git rm scripts/fetch_10yr_bitcoin_data.py scripts/update_bitcoin_data.py scripts/run_all_optimizations.py
   git commit -m "Remove deprecated scripts to clean up codebase"
   ```

## Code Refactoring Opportunities

1. **Consolidate Optimization Scripts**: Consider merging `generate_optimizations_for_periods.py` and `generate_sample_optimizations.py` since they have overlapping functionality.

2. **Streamline Script Parameters**: Update all scripts to use consistent command-line argument patterns (like we did with `run_real_optimizations.py`).

## Data Organization Improvements

1. **Version Data Files**: Consider adding a version identifier to data files to track changes over time.

2. **Data Archiving Strategy**: Implement a policy for archiving old optimization results rather than deleting them (perhaps keep the last 3 months).

## Maintenance Best Practices

1. **Code Review Check**: Add a code review item to look for redundancies during each pull request.

2. **Documentation Updates**: Ensure documentation is up-to-date with the current codebase organization.

3. **Dependency Cleanup**: Regularly update requirements.txt to remove unused dependencies.

4. **Test Coverage**: Ensure tests are updated when refactoring code to maintain coverage.

## Implementation Plan

1. First, back up any files to be removed (just in case)
2. Remove the deprecated scripts
3. Update documentation to reflect the changes
4. Run the application to ensure everything still works properly
5. Commit the changes to the repository