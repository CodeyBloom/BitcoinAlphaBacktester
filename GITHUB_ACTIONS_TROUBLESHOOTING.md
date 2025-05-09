# GitHub Actions Troubleshooting Guide

Having issues with your Bitcoin data update workflow? Here are simple solutions to common problems.

## Common Issues and Solutions

### 1. The Workflow Failed to Run

**What it looks like:** 
❌ Red X on the workflow run.

**Possible causes and solutions:**

- **API Rate Limit Exceeded**
  - **Signs:** Error message about "rate limit" or "too many requests"
  - **Solution:** Wait an hour and manually run the workflow again
  
- **Network Connection Issue**
  - **Signs:** Error message about "connection timeout" or "network error"
  - **Solution:** Check your internet connection and try again later

- **Missing Dependencies**
  - **Signs:** Error about "module not found" or "import error"
  - **Solution:** Make sure your requirements.txt file includes all needed packages

### 2. Workflow Runs But Data Doesn't Update

**What it looks like:**
✅ Green checkmarks on the workflow run, but data is outdated.

**Possible causes and solutions:**

- **No New Data Available**
  - **Signs:** Log messages indicate "No changes to commit"
  - **Solution:** This is normal if prices haven't changed since last run

- **Permission Issues**
  - **Signs:** Error about "permission denied" when trying to commit
  - **Solution:** Make sure the repository has Actions write permissions:
    1. Go to repository Settings
    2. Click on "Actions" on the left
    3. Under "Workflow permissions" select "Read and write permissions"
    4. Click "Save"

### 3. Workflow Doesn't Run Automatically

**What it looks like:** 
No automatic runs showing up in the Actions tab.

**Possible causes and solutions:**

- **Incorrect Schedule Format**
  - **Signs:** No scheduled runs occurring
  - **Solution:** Check the cron syntax in the workflow file:
    ```yaml
    on:
      schedule:
        - cron: '0 2 * * 1'  # This runs at 2:00 UTC on Mondays
    ```

- **Repository Not Active**
  - **Signs:** No automatic actions for a long time
  - **Solution:** GitHub may disable scheduled actions for inactive repositories
    1. Make a small commit to the repository
    2. Manually run the workflow once

### 4. Optimization Files Missing or Invalid

**What it looks like:**
Workflow says successful but data verification shows missing files.

**Possible causes and solutions:**

- **Timeout During Optimization**
  - **Signs:** Some optimization files missing, logs show process ended early
  - **Solution:** The workflow might be timing out
    1. Edit `.github/workflows/update_bitcoin_data.yml`
    2. Increase the timeout value: `timeout-minutes: 120`
    3. Commit the change

### 5. Changes Not Being Committed

**What it looks like:**
Workflow runs successfully but changes aren't visible in repository.

**Possible causes and solutions:**

- **Git Configuration Issue**
  - **Signs:** Error related to git configuration or identity
  - **Solution:** Check the git configuration steps in the workflow:
    ```yaml
    - name: Commit and push if there are changes
      run: |
        git config --local user.email "actions@github.com"
        git config --local user.name "GitHub Actions"
        git add data/
        git diff --staged --quiet || git commit -m "Update Bitcoin data and optimization results"
        git push
    ```

## How to Get More Help

If you're still having issues:

1. Look at the detailed logs:
   - Click on the failed workflow run
   - Click on the specific step that failed
   - Read the complete error message

2. Check GitHub Actions Status:
   - Visit [GitHub Status](https://www.githubstatus.com/) to make sure GitHub Actions is operational

3. Modify the Workflow for More Information:
   - Add more `echo` statements to see what's happening
   - For example: `echo "Current directory: $(pwd)"`

4. Get Community Help:
   - Post your question on [Stack Overflow](https://stackoverflow.com/questions/tagged/github-actions) with the tag `github-actions`

Remember that most issues can be solved by checking the workflow logs carefully to identify where exactly the process is failing.