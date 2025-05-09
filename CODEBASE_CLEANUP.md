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

### 3. Redundant Data Files

These data files are duplicates or no longer needed:

- `data/bitcoin_prices_usd.arrow`: We're only using AUD data as per requirements
- Duplicate optimization files in `data/optimizations/` with different date formats (08052025 vs 09052025)

### 4. Duplicate Data in scripts/data/

- The `scripts/data/` directory contains duplicate optimization files that are already in `data/optimizations/`

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

### Phase 1: Script Cleanup

```bash
# Create backup directory
mkdir -p backup/scripts

# Back up scripts before removal
cp scripts/fetch_10yr_bitcoin_data.py backup/scripts/
cp scripts/update_bitcoin_data.py backup/scripts/
cp scripts/run_all_optimizations.py backup/scripts/

# Remove redundant scripts
git rm scripts/fetch_10yr_bitcoin_data.py
git rm scripts/update_bitcoin_data.py
git rm scripts/run_all_optimizations.py

# Commit changes
git commit -m "Remove deprecated scripts to clean up codebase"
```

### Phase 2: Data File Cleanup

```bash
# Create backup directory
mkdir -p backup/data

# Back up data files before removal
cp data/bitcoin_prices_usd.arrow backup/data/

# Remove redundant data files
git rm data/bitcoin_prices_usd.arrow

# Remove scripts/data directory which has duplicates
git rm -r scripts/data

# Commit changes
git commit -m "Remove redundant data files to clean up repository"
```

### Phase 3: Cleanup Verification

1. Restart the application
2. Verify all functionality works as expected
3. Run tests to ensure nothing is broken
4. Update documentation to reflect the changes

### Phase 4: Optimization Consolidation (Future Work)

1. Merge `generate_optimizations_for_periods.py` and `generate_sample_optimizations.py`
2. Update GitHub Actions workflow to use the consolidated script
3. Test the workflow
4. Commit the changes