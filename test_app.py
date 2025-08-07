import streamlit as st

st.title("🔥 Wildfire Analysis Dashboard")
st.write("Testing basic functionality...")

# Test basic imports
try:
    import pandas as pd
    st.success("✅ pandas imported successfully")
except Exception as e:
    st.error(f"❌ pandas import failed: {e}")

try:
    import geopandas as gpd
    st.success("✅ geopandas imported successfully")
except Exception as e:
    st.error(f"❌ geopandas import failed: {e}")

try:
    import boto3
    st.success("✅ boto3 imported successfully")
except Exception as e:
    st.error(f"❌ boto3 import failed: {e}")

try:
    import plotly.express as px
    st.success("✅ plotly imported successfully")
except Exception as e:
    st.error(f"❌ plotly import failed: {e}")

st.write("If you see all green checkmarks, the basic setup is working!") 