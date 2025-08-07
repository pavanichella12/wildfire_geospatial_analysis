import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import boto3
from io import BytesIO
import requests

# Page configuration
st.set_page_config(
    page_title="🔥 Wildfire Analysis Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_from_s3():
    """Load wildfire data from S3."""
    try:
        # Access Streamlit secrets
        aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
        bucket_name = st.secrets["S3_BUCKET_NAME"]
        file_path = st.secrets["S3_OBJECT_KEY"]
        
        st.info("☁️ Loading data from S3...")
        
        # Connect to S3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        
        # Download the file into memory
        obj = s3.get_object(Bucket=bucket_name, Key=file_path)
        geojson_data = obj["Body"].read()
        
        # Load it using GeoPandas
        gdf = gpd.read_file(BytesIO(geojson_data))
        
        st.success(f"✅ Successfully loaded {len(gdf)} records from S3")
        return gdf
        
    except Exception as e:
        st.error(f"❌ Failed to load data from S3: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🔥 Wildfire Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    gdf = load_data_from_s3()
    
    if gdf is not None:
        st.write(f"📊 Dataset loaded: {len(gdf)} wildfire records")
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Fires", f"{len(gdf):,}")
        with col2:
            st.metric("Years Covered", f"{gdf['FIREYEAR'].nunique()}")
        with col3:
            st.metric("States", f"{gdf['STATE'].nunique()}")
        
        # Show sample data
        st.subheader("📋 Sample Data")
        st.dataframe(gdf.head())
        
    else:
        st.error("❌ No data loaded. Please check your S3 configuration.")

if __name__ == "__main__":
    main() 