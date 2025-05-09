# GitHub Actions Setup with Screenshots

This guide shows you exactly how to set up GitHub Actions with helpful screenshots. Follow along to keep your Bitcoin Strategy Backtester data automatically updated!

## Step 1: Access Actions Tab

Go to your GitHub repository and click on the "Actions" tab:

```
[Repository Name]
┌───────┐ ┌─────┐ ┌───────┐ ┌────────┐ ┌─────────┐
│ Code  │ │ ... │ │ Pull  │ │ Actions│ │ Projects│
└───────┘ └─────┘ └───────┘ └────────┘ └─────────┘
            Your repository files will be here
```

## Step 2: Set Up a New Workflow

You'll see one of these screens:

### If your repository has no workflows yet:

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│     Get started with GitHub Actions                  │
│                                                      │
│     ┌──────────────────┐ ┌─────────────────┐        │
│     │ Skip this and    │ │ Set up workflow │        │
│     │ set up workflow  │ │ yourself        │        │
│     └──────────────────┘ └─────────────────┘        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

Click "Skip this and set up workflow yourself".

### If you already have workflows:

```
┌──────────────────────────────────────────────────────┐
│     All workflows                                    │
│                                                      │
│     ┌─────────────────────────────────────────┐      │
│     │ New workflow      ▼                      │      │
│     └─────────────────────────────────────────┘      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

Click "New workflow".

## Step 3: Create the Workflow File

If you're creating a new file, you'll see something like this:

```
┌──────────────────────────────────────────────────────┐
│ Name: .github/workflows/update_bitcoin_data.yml      │
│                                                      │
│ ┌────────────────────────────────────────────────┐   │
│ │                                                │   │
│ │ # Enter your workflow content here             │   │
│ │                                                │   │
│ └────────────────────────────────────────────────┘   │
│                                                      │
│           [ Start commit ] [ Cancel ]                │
│                                                      │
└──────────────────────────────────────────────────────┘
```

Now copy and paste the content from our `update_bitcoin_data.yml` file into this editor.

## Step 4: Commit the Workflow File

Click the "Start commit" button, add a commit message like "Add Bitcoin data update workflow", and click "Commit new file".

```
┌──────────────────────────────────────────────────────┐
│ Commit new file                                      │
│                                                      │
│ ┌────────────────────────────────────────────────┐   │
│ │ Add Bitcoin data update workflow               │   │
│ └────────────────────────────────────────────────┘   │
│                                                      │
│ [ Commit directly to main branch ] ○ Create branch   │
│                                                      │
│           [ Commit new file ]                        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Step 5: View Your Workflow

You should now see your workflow in the Actions tab:

```
┌──────────────────────────────────────────────────────┐
│ Actions                                              │
│                                                      │
│ Workflows                                            │
│                                                      │
│ ┌────────────────────────────────────────────────┐   │
│ │ Update Bitcoin Data and Optimizations          │   │
│ │                                                │   │
│ │ This workflow runs weekly to update Bitcoin    │   │
│ │ data and strategy optimizations                │   │
│ │                                                │   │
│ │ Last run: Never                Run workflow >  │   │
│ └────────────────────────────────────────────────┘   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Step 6: Run the Workflow Manually

Click "Run workflow" to manually trigger the workflow:

```
┌──────────────────────────────────────────────────────┐
│ ┌────────────────────────────────────────────────┐   │
│ │ Branch: main                                   │   │
│ │                                                │   │
│ │              [ Run workflow ]                  │   │
│ └────────────────────────────────────────────────┘   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Step 7: Monitor the Workflow Run

You'll see the workflow running:

```
┌──────────────────────────────────────────────────────┐
│ Update Bitcoin Data and Optimizations #1             │
│ Workflow: Update Bitcoin Data and Optimizations      │
│ Status: In progress                                  │
│                                                      │
│ ● Set up job           ✓ Completed                   │
│ ● Check out repository ✓ Completed                   │
│ ● Set up Python        ✓ Completed                   │
│ ● Install dependencies ✓ Completed                   │
│ ● Fetch Bitcoin data   ⟳ In progress                │
│ ○ Generate optimizations [not started]               │
│ ○ Run real optimizations [not started]               │
│ ○ Verify data integrity [not started]                │
│ ○ Commit changes [not started]                       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Step 8: Check the Results

After the workflow completes (which may take several minutes), you'll see all steps marked as complete:

```
┌──────────────────────────────────────────────────────┐
│ Update Bitcoin Data and Optimizations #1             │
│ Workflow: Update Bitcoin Data and Optimizations      │
│ Status: Completed                                    │
│                                                      │
│ ● Set up job           ✓ Completed                   │
│ ● Check out repository ✓ Completed                   │
│ ● Set up Python        ✓ Completed                   │
│ ● Install dependencies ✓ Completed                   │
│ ● Fetch Bitcoin data   ✓ Completed                   │
│ ● Generate optimizations ✓ Completed                 │
│ ● Run real optimizations ✓ Completed                 │
│ ● Verify data integrity ✓ Completed                  │
│ ● Commit changes       ✓ Completed                   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Step 9: Verify Automatic Schedule

Your workflow is now set to run automatically every Monday at 2:00 UTC. You can see this in the workflow file:

```yaml
on:
  schedule:
    # Run weekly on Monday at 2:00 UTC
    - cron: '0 2 * * 1'
  workflow_dispatch:  # Allow manual triggering
```

## That's It!

Your GitHub Actions workflow is now set up! It will:

1. Run automatically every week
2. Fetch the latest Bitcoin price data
3. Update all strategy optimizations
4. Verify data integrity
5. Commit the changes back to your repository

All without you having to do anything manually!