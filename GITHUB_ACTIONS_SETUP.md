# GitHub Actions Setup Tutorial

This step-by-step guide will help you set up GitHub Actions to keep your Bitcoin Strategy Backtester data always up to date. No prior experience with GitHub Actions is required!

## What is GitHub Actions?

GitHub Actions is like having a robot assistant that can automatically run tasks for your code repository. We're going to use it to fetch fresh Bitcoin data and update optimizations every week.

## Step 1: Make Sure You Have a GitHub Repository

First, make sure your project is in a GitHub repository. If not:

1. Create a GitHub account if you don't have one at [github.com](https://github.com/)
2. Create a new repository
3. Upload your project files to this repository

## Step 2: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click on the "Actions" tab at the top of your repository
3. You might see a message saying "Get started with GitHub Actions" - if so, just click "Skip this and set up a workflow yourself"

## Step 3: Add the Workflow File

Our data update workflow file is already prepared at `.github/workflows/update_bitcoin_data.yml`. You just need to make sure it's in your repository.

If you need to add it:

1. In your repository, click on "Add file" then "Create new file"
2. For the file path, enter `.github/workflows/update_bitcoin_data.yml`
3. Copy and paste the content of the update_bitcoin_data.yml file
4. Click "Commit new file" at the bottom

## Step 4: Verify the Workflow is Set Up

1. Go to the "Actions" tab again
2. You should now see "Update Bitcoin Data and Optimizations" listed as a workflow
3. This workflow is set to run automatically every Monday at 2:00 UTC

## Step 5: Run the Workflow Manually (Optional)

To test that everything works:

1. In the "Actions" tab, click on "Update Bitcoin Data and Optimizations"
2. Click the "Run workflow" button (dropdown on the right)
3. Keep the default branch selected
4. Click the green "Run workflow" button

## Step 6: Check the Results

After running (this might take several minutes):

1. Click on the workflow run that appears
2. You'll see steps like "Fetch Bitcoin historical data", "Run real optimizations", etc.
3. Each step will have a ✅ if it succeeded or ❌ if it failed
4. Click on any step to see more details

## What the Workflow Does

Every week, this workflow will:

1. Download 10 years of Bitcoin price data
2. Generate optimizations for different strategies (DCA, MACO, RSI, etc.)
3. Verify everything is working properly
4. Save all the updated files to your repository

## Common Issues and Solutions

### "Workflow Failed" Error

If the workflow fails:

1. Click on the failed run in the Actions tab
2. Look for steps with a red ❌
3. Click on those steps to see the error details
4. Common issues include:
   - API rate limits: Just run the workflow again later
   - Connection problems: Check if your repository has internet access

### Making Changes to the Workflow

If you need to modify how the workflow runs:

1. Edit the `.github/workflows/update_bitcoin_data.yml` file
2. The schedule is set with `cron: '0 2 * * 1'` (Monday 2:00 UTC)
3. To change the schedule, modify these numbers ([cron syntax guide](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onschedule))

## Monitoring Your Data

Every time the workflow runs, it creates a `data/last_verification.json` file with information about:

- How many days of Bitcoin data you have
- Whether all optimization files are valid
- Any issues that need attention

## That's It!

Your Bitcoin Strategy Backtester will now automatically stay up to date with the latest data! The workflow will run every week in the background without you having to do anything.