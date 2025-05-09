# Streamlit Cloud Deployment Guide

## Prerequisites
1. A GitHub account
2. Your repository pushed to GitHub
3. All requirements listed in requirements.txt

## Deployment Steps

1. **Sign in to Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account

2. **Deploy Your App**
   - Click "New app"
   - Select your repository, branch (usually main)
   - For Main file path, enter: `app.py`
   - Leave Python version as the default (3.10 or newer)
   - Click "Deploy"
   
3. **Important Config File Settings**
   - Make sure your `.streamlit/config.toml` file has these settings:
     ```toml
     [server]
     headless = true
     enableXsrfProtection = false
     enableCORS = false
     
     [browser]
     gatherUsageStats = false
     ```
   - Do NOT specify a port in the config file - Streamlit Cloud handles this

3. **Advanced Settings (optional)**
   - You can configure additional settings:
     - Python version (recommend 3.9+)
     - Package dependencies
     - Secrets management
     - Custom subdomains (available on paid plans)

## Managing Your App

- **Updates**: Every commit to your selected branch will automatically update your app
- **Monitoring**: You can view app logs in the Streamlit Cloud dashboard
- **Sharing**: Your app will have a public URL you can share

## Troubleshooting

### Common Issues

1. **App crashing on startup**
   - Check if all dependencies are in requirements.txt
   - Verify your app can run locally
   - Check Streamlit Cloud logs for error messages
   - Make sure your .streamlit/config.toml doesn't have a hardcoded port
   - Streamlit Cloud expects to reach your app on port 8501

2. **Data not loading**
   - Ensure data files are in the repository or downloaded at runtime
   - Check file paths (they should be relative)
   - Verify that GitHub Actions data updates are working

3. **Memory limits**
   - Free tier is limited to 1GB RAM
   - Optimize memory usage or consider upgrading to a paid plan

## Streamlit Cloud Free Tier Limitations

- 1 app per account
- 1GB RAM per app
- Public repositories only
- Weekly app inactivity timeout (apps sleep if not used)
- No custom domains (only streamlit.app URLs)

## Useful Streamlit Cloud Documentation

- [App deployment](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [Managing apps](https://docs.streamlit.io/streamlit-cloud/get-started/manage-your-app)
- [Advanced features](https://docs.streamlit.io/streamlit-cloud/get-started/advanced-features)