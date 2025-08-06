"""
Deployment-ready Streamlit dashboard for wildfire analysis.
Supports loading data from cloud storage (Google Drive, Dropbox) or local files.
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import folium
# from streamlit_folium import folium_static
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import load_wildfire_data, get_data_loader
# from data_processing import WildfireDataProcessor
# from spatial_analysis import WildfireSpatialAnalyzer
# from visualization import WildfireVisualizer

# Page configuration
st.set_page_config(
    page_title="🔥 Wildfire Analysis Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b35;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_from_cloud():
    """
    Load wildfire data from cloud storage with fallback to local.
    """
    try:
        # Try to load from cloud storage first
        # You can configure these in Streamlit Cloud secrets
        google_drive_id = st.secrets.get("GOOGLE_DRIVE_FILE_ID", "1KWRrW5i6Tccj7egW3oXuna6vqeo28g9z")
        dropbox_link = st.secrets.get("DROPBOX_SHARE_LINK", "")
        
        if google_drive_id:
            st.info("🌐 Loading data from Google Drive...")
            return load_wildfire_data("google_drive", file_id=google_drive_id)
        elif dropbox_link:
            st.info("🌐 Loading data from Dropbox...")
            return load_wildfire_data("dropbox", share_link=dropbox_link)
        else:
            # Fallback to local file
            st.info("📁 Loading data from local storage...")
            return load_wildfire_data("local")
            
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        st.info("💡 To use cloud storage, configure GOOGLE_DRIVE_FILE_ID or DROPBOX_SHARE_LINK in Streamlit Cloud secrets.")
        return None

def show_overview_page(gdf):
    """Display overview page with data insights."""
    st.markdown('<h1 class="main-header">🔥 Wildfire Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Fires", f"{len(gdf):,}")
    
    with col2:
        year_range = f"{gdf['FIREYEAR'].min()}-{gdf['FIREYEAR'].max()}"
        st.metric("Year Range", year_range)
    
    with col3:
        total_acres = gdf['TOTALACRES'].sum()
        st.metric("Total Acres Burned", f"{total_acres:,.0f}")
    
    with col4:
        avg_size = gdf['TOTALACRES'].mean()
        st.metric("Average Fire Size", f"{avg_size:.1f} acres")
    
    # Data quality report
    st.subheader("📊 Data Quality Overview")
    
    # Missing values
    missing_data = gdf.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Percentage': (missing_data.values / len(gdf) * 100).round(2)
    }).sort_values('Missing Values', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values by Column:**")
        st.dataframe(missing_df[missing_df['Missing Values'] > 0], use_container_width=True)
    
    with col2:
        # Data completeness
        complete_records = len(gdf.dropna(subset=['FIREYEAR', 'TOTALACRES', 'STATCAUSE']))
        completeness = (complete_records / len(gdf) * 100).round(2)
        
        st.write("**Data Completeness:**")
        st.metric("Complete Records", f"{complete_records:,} ({completeness}%)")
        
        # Coordinate system info
        st.write("**Coordinate System:**")
        st.write(f"EPSG: {gdf.crs.to_string()}")
    
    # Data distribution charts
    st.subheader("📈 Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fire years distribution
        year_counts = gdf['FIREYEAR'].value_counts().sort_index()
        fig_years = px.line(
            x=year_counts.index, 
            y=year_counts.values,
            title="Wildfires by Year",
            labels={'x': 'Year', 'y': 'Number of Fires'}
        )
        fig_years.update_layout(height=400)
        st.plotly_chart(fig_years, use_container_width=True)
    
    with col2:
        # Fire size distribution
        size_bins = [0, 10, 100, 1000, 10000, float('inf')]
        size_labels = ['<10', '10-100', '100-1K', '1K-10K', '>10K']
        gdf['size_category'] = pd.cut(gdf['TOTALACRES'], bins=size_bins, labels=size_labels)
        
        size_counts = gdf['size_category'].value_counts()
        fig_size = px.bar(
            x=size_counts.index,
            y=size_counts.values,
            title="Fire Size Distribution (Acres)",
            labels={'x': 'Size Category', 'y': 'Number of Fires'}
        )
        fig_size.update_layout(height=400)
        st.plotly_chart(fig_size, use_container_width=True)

def show_data_exploration_page(gdf):
    """Display data exploration page with interactive charts."""
    st.header("📊 Data Exploration")
    
    # Filter controls
    st.sidebar.subheader("🔍 Data Filters")
    
    # Year range filter
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(gdf['FIREYEAR'].min()),
        max_value=int(gdf['FIREYEAR'].max()),
        value=(int(gdf['FIREYEAR'].min()), int(gdf['FIREYEAR'].max())))
    
    # Size range filter
    size_range = st.sidebar.slider(
        "Fire Size Range (Acres)",
        min_value=float(gdf['TOTALACRES'].min()),
        max_value=float(gdf['TOTALACRES'].max()),
        value=(float(gdf['TOTALACRES'].min()), float(gdf['TOTALACRES'].max())))
    
    # Apply filters
    filtered_gdf = gdf[
        (gdf['FIREYEAR'] >= year_range[0]) & 
        (gdf['FIREYEAR'] <= year_range[1]) &
        (gdf['TOTALACRES'] >= size_range[0]) & 
        (gdf['TOTALACRES'] <= size_range[1])
    ]
    
    st.write(f"**Showing {len(filtered_gdf):,} fires** (filtered from {len(gdf):,} total)")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top fire causes
        cause_counts = filtered_gdf['STATCAUSE'].value_counts().head(10)
        fig_causes = px.bar(
            x=cause_counts.values,
            y=cause_counts.index,
            orientation='h',
            title="Top 10 Fire Causes",
            labels={'x': 'Number of Fires', 'y': 'Cause'}
        )
        fig_causes.update_layout(height=400)
        st.plotly_chart(fig_causes, use_container_width=True)
    
    with col2:
        # Fire size vs year
        fig_scatter = px.scatter(
            filtered_gdf,
            x='FIREYEAR',
            y='TOTALACRES',
            title="Fire Size vs Year",
            labels={'FIREYEAR': 'Year', 'TOTALACRES': 'Acres Burned'}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Statistics
    st.subheader("📈 Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Fire Size", f"{filtered_gdf['TOTALACRES'].mean():.1f} acres")
        st.metric("Median Fire Size", f"{filtered_gdf['TOTALACRES'].median():.1f} acres")
        st.metric("Largest Fire", f"{filtered_gdf['TOTALACRES'].max():,.0f} acres")
    
    with col2:
        st.metric("Total Acres Burned", f"{filtered_gdf['TOTALACRES'].sum():,.0f}")
        st.metric("Number of Years", f"{filtered_gdf['FIREYEAR'].nunique()}")
        st.metric("Average Fires/Year", f"{len(filtered_gdf) / filtered_gdf['FIREYEAR'].nunique():.0f}")
    
    with col3:
        st.metric("Most Common Cause", filtered_gdf['STATCAUSE'].mode().iloc[0] if not filtered_gdf['STATCAUSE'].mode().empty else "N/A")
        st.metric("Unique Causes", f"{filtered_gdf['STATCAUSE'].nunique()}")
        st.metric("Data Completeness", f"{(1 - filtered_gdf.isnull().sum().sum() / (len(filtered_gdf) * len(filtered_gdf.columns)) * 100):.1f}%")

def show_spatial_analysis_page(gdf):
    """Display spatial analysis page with interactive maps."""
    st.header("🗺️ Spatial Analysis")
    
    # Map controls
    st.sidebar.subheader("🗺️ Map Controls")
    
    sample_size = st.sidebar.slider("Sample Size", 100, 5000, 1000)
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(gdf['FIREYEAR'].min()),
        max_value=int(gdf['FIREYEAR'].max()),
        value=(int(gdf['FIREYEAR'].min()), int(gdf['FIREYEAR'].max())))
    
    size_range = st.sidebar.slider(
        "Fire Size Range (Acres)",
        min_value=float(gdf['TOTALACRES'].min()),
        max_value=float(gdf['TOTALACRES'].max()),
        value=(float(gdf['TOTALACRES'].min()), float(gdf['TOTALACRES'].max())))
    
    color_by = st.sidebar.selectbox(
        "Color By",
        ["Fire Size (Acres)", "Year", "Cause", "Size Category"]
    )
    
    size_by = st.sidebar.selectbox(
        "Size By",
        ["Fire Size", "Fixed Size", "Size Category"]
    )
    
    map_style = st.sidebar.selectbox(
        "Map Style",
        ["Light", "Dark", "Satellite", "Terrain"]
    )
    
    # Filter and sample data
    filtered_gdf = gdf[
        (gdf['FIREYEAR'] >= year_range[0]) & 
        (gdf['FIREYEAR'] <= year_range[1]) &
        (gdf['TOTALACRES'] >= size_range[0]) & 
        (gdf['TOTALACRES'] <= size_range[1])
    ]
    
    if len(filtered_gdf) > sample_size:
        viz_data = filtered_gdf.sample(n=sample_size, random_state=42)
    else:
        viz_data = filtered_gdf.copy()
    
    # Clean data for visualization
    viz_data['TOTALACRES_clean'] = viz_data['TOTALACRES'].fillna(0)
    viz_data = viz_data.dropna(subset=['geometry'])
    
    # Create size categories
    size_bins = [0, 10, 100, 1000, 10000, float('inf')]
    size_labels = ['<10', '10-100', '100-1K', '1K-10K', '>10K']
    viz_data['size_category'] = pd.cut(viz_data['TOTALACRES_clean'], bins=size_bins, labels=size_labels)
    
    # Set color and size columns
    color_col = {
        "Fire Size (Acres)": "TOTALACRES_clean",
        "Year": "FIREYEAR",
        "Cause": "STATCAUSE",
        "Size Category": "size_category"
    }[color_by]
    
    size_col = {
        "Fire Size": "TOTALACRES_clean",
        "Fixed Size": None,
        "Size Category": "size_category"
    }[size_by]
    
    # Map styles
    map_styles = {
        "Light": "carto-positron",
        "Dark": "carto-darkmatter",
        "Satellite": "satellite-streets",
        "Terrain": "stamen-terrain"
    }
    
    # Create interactive map
    st.subheader("🌍 Interactive Wildfire Map")
    
    try:
        fig_spatial = px.scatter_map(
            viz_data,
            lat=viz_data.geometry.y,
            lon=viz_data.geometry.x,
            color=color_col,
            size=size_col if size_col else None,
            hover_data=['FIRENAME', 'FIREYEAR', 'STATCAUSE', 'TOTALACRES'] if all(col in viz_data.columns for col in ['FIRENAME', 'FIREYEAR', 'STATCAUSE', 'TOTALACRES']) else None,
            title=f"Interactive Wildfire Map - {len(viz_data):,} fires displayed",
            map_style=map_styles[map_style],
            zoom=3,
            color_continuous_scale="viridis" if color_by in ["Fire Size (Acres)", "Year"] else None,
        )
        
        fig_spatial.update_layout(height=600)
        st.plotly_chart(fig_spatial, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating spatial visualization: {str(e)}")
    
    # Color legend
    st.subheader("🎨 Color Legend")
    
    if color_by == "Fire Size (Acres)":
        st.write("**Fire Size Color Scale:**")
        st.write("- 🔴 Red: Large fires (>10,000 acres)")
        st.write("- 🟡 Yellow: Medium fires (1,000-10,000 acres)")
        st.write("- 🟢 Green: Small fires (<1,000 acres)")
    elif color_by == "Year":
        st.write("**Year Color Scale:**")
        st.write("- 🔴 Red: Recent years")
        st.write("- 🟡 Yellow: Middle years")
        st.write("- 🟢 Green: Early years")
    elif color_by == "Cause":
        st.write("**Fire Cause Categories:**")
        cause_counts = viz_data['STATCAUSE'].value_counts().head(5)
        for cause, count in cause_counts.items():
            st.write(f"- {cause}: {count} fires")
    
    # Map statistics
    st.subheader("📊 Map Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fires Displayed", len(viz_data))
        st.metric("Total Fires in Range", len(filtered_gdf))
        st.metric("Sampling Rate", f"{(len(viz_data) / len(filtered_gdf) * 100):.1f}%")
    
    with col2:
        st.metric("Average Fire Size", f"{viz_data['TOTALACRES_clean'].mean():.1f} acres")
        st.metric("Largest Fire", f"{viz_data['TOTALACRES_clean'].max():,.0f} acres")
        st.metric("Smallest Fire", f"{viz_data['TOTALACRES_clean'].min():.1f} acres")
    
    with col3:
        st.metric("Year Range", f"{viz_data['FIREYEAR'].min()}-{viz_data['FIREYEAR'].max()}")
        st.metric("Unique Causes", viz_data['STATCAUSE'].nunique())
        st.metric("Geographic Coverage", f"{viz_data.geometry.bounds.iloc[0]['minx']:.2f}° to {viz_data.geometry.bounds.iloc[0]['maxx']:.2f}°")

def main():
    """Main application function."""
    
    # Sidebar navigation
    st.sidebar.title("🔥 Wildfire Analysis")
    
    # Data source selection
    st.sidebar.subheader("📁 Data Source")
    data_source = st.sidebar.selectbox(
        "Choose Data Source",
        ["Cloud Storage (Auto)", "Local File", "Google Drive", "Dropbox"]
    )
    
    # Load data based on selection
    if data_source == "Cloud Storage (Auto)":
        gdf = load_data_from_cloud()
    elif data_source == "Local File":
        gdf = load_wildfire_data("local")
    elif data_source == "Google Drive":
        file_id = st.sidebar.text_input("Google Drive File ID")
        if file_id:
            gdf = load_wildfire_data("google_drive", file_id=file_id)
        else:
            gdf = None
    elif data_source == "Dropbox":
        share_link = st.sidebar.text_input("Dropbox Share Link")
        if share_link:
            gdf = load_wildfire_data("dropbox", share_link=share_link)
        else:
            gdf = None
    
    if gdf is None:
        st.error("❌ No data loaded. Please check your data source configuration.")
        st.info("💡 For deployment, configure cloud storage credentials in Streamlit Cloud secrets.")
        return
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Data Exploration", "Spatial Analysis"]
    )
    
    # Display selected page
    if page == "Overview":
        show_overview_page(gdf)
    elif page == "Data Exploration":
        show_data_exploration_page(gdf)
    elif page == "Spatial Analysis":
        show_spatial_analysis_page(gdf)

if __name__ == "__main__":
    main() 