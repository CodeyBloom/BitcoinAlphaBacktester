#!/bin/bash
# GitHub Setup Script for Bitcoin Strategy Backtester

echo "Bitcoin Strategy Backtester - GitHub Setup"
echo "=========================================="
echo

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git first."
    exit 1
fi

# Ask for GitHub details
echo "Please enter your GitHub username:"
read github_username

echo "Please enter your repository name (e.g., bitcoin-strategy-backtester):"
read repo_name

echo "Which branch would you like to use? (default: main)"
read branch_name
branch_name=${branch_name:-main}

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
    echo "Git repository initialized."
else
    echo "Git repository already initialized."
fi

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore file..."
    cp SUGGESTED_GITIGNORE .gitignore
    echo ".gitignore created."
fi

# Show status before adding files
echo
echo "Current Git Status:"
git status

# Add files
echo
echo "Adding files to Git..."
git add .
echo "Files added."

# Initial commit
echo
echo "Creating initial commit..."
git commit -m "Initial commit of Bitcoin Strategy Backtester"
echo "Commit created."

# Add remote
echo
echo "Adding remote repository..."
git remote add origin "https://github.com/$github_username/$repo_name.git"
echo "Remote repository added."

# Push to GitHub
echo
echo "Pushing to GitHub ($branch_name branch)..."
git push -u origin $branch_name

echo
echo "Setup completed!"
echo
echo "Next steps:"
echo "1. Go to https://streamlit.io/cloud"
echo "2. Sign in with your GitHub account"
echo "3. Click 'New app'"
echo "4. Select your repository ($repo_name), branch ($branch_name), and main file (app.py)"
echo "5. Click 'Deploy'"
echo
echo "Your app should be live in a few minutes!"