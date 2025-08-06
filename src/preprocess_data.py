#!/usr/bin/env python3
"""
Pre-process wildfire data for faster dashboard loading.
This script creates cached versions of processed data.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data():
    """Pre-process and cache wildfire data."""
    
    data_path = Path("data/National_USFS_Fire_Occurrence_Point_(Feature_Layer).geojson")
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    logger.info("Loading wildfire data...")
    gdf = gpd.read_file(data_path)
    
    logger.info("Cleaning data...")
    # Basic cleaning
    gdf = gdf.dropna(subset=['TOTALACRES', 'FIREYEAR'])
    gdf = gdf[gdf['FIREYEAR'].between(1900, 2024)]
    
    # Remove invalid geometries
    gdf = gdf[gdf.geometry.is_valid]
    
    logger.info("Engineering features...")
    # Create features
    gdf['is_large_fire'] = (gdf['TOTALACRES'] > 100).astype(int)
    gdf['decade'] = (gdf['FIREYEAR'] // 10) * 10
    gdf['year_category'] = pd.cut(gdf['FIREYEAR'], 
                                  bins=[1900, 1950, 2000, 2025], 
                                  labels=['Historical', 'Mid-Century', 'Modern'])
    
    # Size categories
    gdf['size_category'] = pd.cut(gdf['TOTALACRES'],
                                  bins=[0, 10, 100, 1000, float('inf')],
                                  labels=['A', 'B', 'C', 'D'])
    
    # Spatial features
    gdf['lat_bin'] = (gdf.geometry.y * 10).astype(int)
    gdf['lon_bin'] = (gdf.geometry.x * 10).astype(int)
    
    logger.info("Saving processed data...")
    
    # Save processed GeoDataFrame
    gdf.to_pickle(cache_dir / "processed_wildfires.pkl")
    
    # Save sample for fast demo
    sample_gdf = gdf.sample(n=1000, random_state=42)
    sample_gdf.to_pickle(cache_dir / "sample_wildfires.pkl")
    
    # Save summary statistics
    summary = {
        'total_records': len(gdf),
        'date_range': (gdf['FIREYEAR'].min(), gdf['FIREYEAR'].max()),
        'total_acres': gdf['TOTALACRES'].sum(),
        'avg_fire_size': gdf['TOTALACRES'].mean(),
        'large_fires': len(gdf[gdf['TOTALACRES'] > 100]),
        'spatial_bounds': gdf.total_bounds.tolist()
    }
    
    with open(cache_dir / "summary_stats.pkl", 'wb') as f:
        pickle.dump(summary, f)
    
    logger.info(f"Pre-processing complete! Processed {len(gdf):,} records.")
    logger.info(f"Cache files saved to: {cache_dir}")
    
    return gdf, summary

if __name__ == "__main__":
    preprocess_data() 