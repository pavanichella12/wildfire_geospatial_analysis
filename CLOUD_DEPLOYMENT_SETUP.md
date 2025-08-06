# 🌐 Cloud Deployment Setup Guide

## 📋 Overview
This guide will help you deploy your wildfire analysis dashboard with the full dataset using cloud storage.

## 🚀 Step 1: Upload Data to Cloud Storage

### Option A: Google Drive (Recommended)

1. **Upload your data file:**
   - Go to [Google Drive](https://drive.google.com)
   - Upload `data/National_USFS_Fire_Occurrence_Point_(Feature_Layer).geojson`
   - Right-click the file → "Share" → "Copy link"

2. **Get the File ID:**
   - The link looks like: `https://drive.google.com/file/d/YOUR_FILE_ID/view`
   - Copy the `YOUR_FILE_ID` part

3. **Make it publicly accessible:**
   - Right-click the file → "Share" → "Anyone with the link" → "Viewer"

### Option B: Dropbox

1. **Upload your data file:**
   - Go to [Dropbox](https://dropbox.com)
   - Upload `data/National_USFS_Fire_Occurrence_Point_(Feature_Layer).geojson`
   - Right-click the file → "Share" → "Copy link"

2. **Make it publicly accessible:**
   - Click the link settings → "Allow downloads"

## 🚀 Step 2: Deploy to Streamlit Cloud

### 1. Push to GitHub

```bash
# Add all files (excluding the large data file)
git add .

# Commit changes
git commit -m "Add deployment-ready dashboard with cloud storage support"

# Push to GitHub
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure your app:**
   - **Repository:** `pavanichella12/wildfire_geospatial_analysis`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app_deploy.py`
   - **Python version:** `3.9`

### 3. Configure Secrets

In Streamlit Cloud, go to your app settings and add these secrets:

#### For Google Drive:
```toml
GOOGLE_DRIVE_FILE_ID = "your_file_id_here"
```

#### For Dropbox:
```toml
DROPBOX_SHARE_LINK = "your_dropbox_share_link_here"
```

## 🚀 Step 3: Test Your Deployment

1. **Wait for deployment** (usually 2-5 minutes)
2. **Visit your app URL** (provided by Streamlit Cloud)
3. **Test the data loading** by selecting different data sources in the sidebar

## 🔧 Troubleshooting

### If data doesn't load:

1. **Check file permissions** - Make sure your cloud storage file is publicly accessible
2. **Verify file ID/link** - Double-check the secrets configuration
3. **Check logs** - In Streamlit Cloud, go to your app → "Manage app" → "Logs"

### If the app is slow:

1. **Reduce sample size** - The app samples data for performance
2. **Use local data** - For development, use the local file option
3. **Optimize data** - Consider compressing the data file

## 📊 Performance Tips

- **Sample Size:** The app uses intelligent sampling (1,000 fires by default) for interactive maps
- **Caching:** Data is cached to improve performance
- **Lazy Loading:** Only loads data when needed

## 🎯 Next Steps

1. **Share your app** - Send the Streamlit Cloud URL to potential employers
2. **Add to portfolio** - Include this in your GitHub portfolio
3. **Document features** - Update your README with the live demo link

## 📞 Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify your cloud storage configuration
3. Test with a smaller dataset first

---

**🎉 Congratulations!** Your wildfire analysis dashboard is now deployed and accessible worldwide! 