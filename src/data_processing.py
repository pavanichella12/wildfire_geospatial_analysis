"""
Wildfire Data Processing Module

This module handles the loading, cleaning, and preprocessing of USFS Fire Occurrence data.
Demonstrates advanced geospatial data analysis capabilities including:
- Data validation and quality assessment
- Spatial data cleaning and transformation
- Feature engineering for machine learning
- Automated data processing workflows

Author: [Your Name]
Date: 2025
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WildfireDataProcessor:
    """
    A comprehensive data processor for USFS wildfire occurrence data.
    
    This class demonstrates advanced geospatial data analysis capabilities including:
    - Automated data validation and quality assessment
    - Spatial data cleaning and transformation
    - Feature engineering for predictive modeling
    - Reproducible data processing workflows
    """
    
    def __init__(self, data_path: str = "data/National_USFS_Fire_Occurrence_Point_(Feature_Layer).geojson"):
        """
        Initialize the data processor.
        
        Args:
            data_path (str): Path to the GeoJSON file containing wildfire data
        """
        self.data_path = Path(data_path)
        self.gdf = None
        self.processed_data = None
        
    def load_data(self) -> gpd.GeoDataFrame:
        """
        Load and perform initial data exploration.
        
        Returns:
            gpd.GeoDataFrame: Loaded geospatial data
        """
        logger.info("Loading wildfire data...")
        
        try:
            # Load the GeoJSON file
            self.gdf = gpd.read_file(self.data_path)
            logger.info(f"Successfully loaded {len(self.gdf)} wildfire records")
            
            # Display basic information
            logger.info(f"Data columns: {list(self.gdf.columns)}")
            logger.info(f"Coordinate system: {self.gdf.crs}")
            
            return self.gdf
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data_quality(self) -> dict:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
            dict: Data quality metrics and issues
        """
        logger.info("Performing data quality assessment...")
        
        quality_report = {
            'total_records': len(self.gdf),
            'missing_values': {},
            'data_types': {},
            'spatial_issues': {},
            'temporal_issues': {},
            'recommendations': []
        }
        
        # Check for missing values
        for col in self.gdf.columns:
            missing_count = self.gdf[col].isnull().sum()
            missing_pct = (missing_count / len(self.gdf)) * 100
            quality_report['missing_values'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            if missing_pct > 50:
                quality_report['recommendations'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
        
        # Check data types
        for col in self.gdf.columns:
            quality_report['data_types'][col] = str(self.gdf[col].dtype)
        
        # Spatial validation
        invalid_geometries = self.gdf[~self.gdf.geometry.is_valid].shape[0]
        quality_report['spatial_issues']['invalid_geometries'] = invalid_geometries
        
        # Temporal validation
        if 'FIREYEAR' in self.gdf.columns:
            current_year = datetime.now().year
            future_fires = self.gdf[self.gdf['FIREYEAR'] > current_year].shape[0]
            quality_report['temporal_issues']['future_fires'] = future_fires
        
        logger.info("Data quality assessment completed")
        return quality_report
    
    def clean_data(self) -> gpd.GeoDataFrame:
        """
        Clean and preprocess the wildfire data.
        
        Returns:
            gpd.GeoDataFrame: Cleaned geospatial data
        """
        logger.info("Cleaning wildfire data...")
        
        # Create a copy for cleaning
        cleaned_gdf = self.gdf.copy()
        
        # Remove invalid geometries
        initial_count = len(cleaned_gdf)
        cleaned_gdf = cleaned_gdf[cleaned_gdf.geometry.is_valid]
        logger.info(f"Removed {initial_count - len(cleaned_gdf)} invalid geometries")
        
        # Clean temporal data
        if 'FIREYEAR' in cleaned_gdf.columns:
            current_year = datetime.now().year
            cleaned_gdf = cleaned_gdf[
                (cleaned_gdf['FIREYEAR'] >= 1900) & 
                (cleaned_gdf['FIREYEAR'] <= current_year)
            ]
            logger.info(f"Filtered to valid fire years (1900-{current_year})")
        
        # Clean size data
        if 'TOTALACRES' in cleaned_gdf.columns:
            # Remove negative or unreasonably large values
            cleaned_gdf = cleaned_gdf[
                (cleaned_gdf['TOTALACRES'] >= 0) & 
                (cleaned_gdf['TOTALACRES'] <= 1000000)  # 1 million acres max
            ]
        
        # Standardize text fields
        text_columns = ['FIRENAME', 'COMPLEXNAME', 'STATCAUSE']
        for col in text_columns:
            if col in cleaned_gdf.columns:
                cleaned_gdf[col] = cleaned_gdf[col].astype(str).str.strip()
                cleaned_gdf[col] = cleaned_gdf[col].replace(['None', 'nan', ''], np.nan)
        
        logger.info(f"Data cleaning completed. Final dataset: {len(cleaned_gdf)} records")
        return cleaned_gdf
    
    def engineer_features(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create engineered features for machine learning.
        
        Args:
            gdf (gpd.GeoDataFrame): Input geospatial data
            
        Returns:
            gpd.GeoDataFrame: Data with engineered features
        """
        logger.info("Engineering features for machine learning...")
        
        # Create a copy for feature engineering
        enhanced_gdf = gdf.copy()
        
        # Temporal features
        if 'FIREYEAR' in enhanced_gdf.columns:
            enhanced_gdf['decade'] = (enhanced_gdf['FIREYEAR'] // 10) * 10
            enhanced_gdf['year_category'] = pd.cut(
                enhanced_gdf['FIREYEAR'], 
                bins=[1900, 1950, 1980, 2000, 2025], 
                labels=['Historical', 'Mid-Century', 'Late-Century', 'Modern']
            )
        
        # Size categories
        if 'TOTALACRES' in enhanced_gdf.columns:
            enhanced_gdf['size_category'] = pd.cut(
                enhanced_gdf['TOTALACRES'],
                bins=[0, 0.25, 10, 100, 1000, float('inf')],
                labels=['A', 'B', 'C', 'D', 'E']
            )
        
        # Spatial features
        enhanced_gdf['latitude'] = enhanced_gdf.geometry.y
        enhanced_gdf['longitude'] = enhanced_gdf.geometry.x
        
        # Create spatial bins
        enhanced_gdf['lat_bin'] = pd.cut(enhanced_gdf['latitude'], bins=20, labels=False)
        enhanced_gdf['lon_bin'] = pd.cut(enhanced_gdf['longitude'], bins=20, labels=False)
        
        # Cause categories
        if 'STATCAUSE' in enhanced_gdf.columns:
            enhanced_gdf['cause_category'] = enhanced_gdf['STATCAUSE'].map({
                'Lightning': 'Natural',
                'Equipment Use': 'Human',
                'Smoking': 'Human',
                'Campfire': 'Human',
                'Debris Burning': 'Human',
                'Railroad': 'Human',
                'Arson': 'Human',
                'Children': 'Human',
                'Miscellaneous': 'Other',
                'Fireworks': 'Human',
                'Powerline': 'Human',
                'Unknown': 'Unknown'
            }).fillna('Other')
        
        # Binary features
        enhanced_gdf['is_large_fire'] = (enhanced_gdf['TOTALACRES'] > 100).astype(int)
        enhanced_gdf['is_natural_cause'] = (enhanced_gdf['cause_category'] == 'Natural').astype(int)
        
        logger.info("Feature engineering completed")
        return enhanced_gdf
    
    def prepare_for_ml(self, gdf: gpd.GeoDataFrame) -> tuple:
        """
        Prepare data for machine learning models.
        
        Args:
            gdf (gpd.GeoDataFrame): Input geospatial data
            
        Returns:
            tuple: (features_df, target_series, feature_names)
        """
        logger.info("Preparing data for machine learning...")
        
        # Select features for ML
        feature_columns = [
            'latitude', 'longitude', 'FIREYEAR', 'TOTALACRES',
            'lat_bin', 'lon_bin', 'is_large_fire', 'is_natural_cause'
        ]
        
        # Add categorical features if they exist
        categorical_features = ['cause_category', 'size_category', 'year_category']
        for col in categorical_features:
            if col in gdf.columns:
                feature_columns.append(col)
        
        # Create features dataframe
        features_df = gdf[feature_columns].copy()
        
        # Handle categorical variables
        categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            features_df[col] = features_df[col].astype('category').cat.codes
        
        # Create target variable (example: predicting large fires)
        target = gdf['is_large_fire']
        
        # Remove rows with missing values
        valid_mask = ~(features_df.isnull().any(axis=1) | target.isnull())
        features_df = features_df[valid_mask]
        target = target[valid_mask]
        
        logger.info(f"ML dataset prepared: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df, target, list(features_df.columns)
    
    def process_pipeline(self) -> tuple:
        """
        Run the complete data processing pipeline.
        
        Returns:
            tuple: (processed_gdf, features_df, target_series, quality_report)
        """
        logger.info("Starting complete data processing pipeline...")
        
        # Load data
        self.load_data()
        
        # Assess data quality
        quality_report = self.validate_data_quality()
        
        # Clean data
        cleaned_gdf = self.clean_data()
        
        # Engineer features
        enhanced_gdf = self.engineer_features(cleaned_gdf)
        
        # Prepare for ML
        features_df, target_series, feature_names = self.prepare_for_ml(enhanced_gdf)
        
        self.processed_data = enhanced_gdf
        
        logger.info("Data processing pipeline completed successfully")
        
        return enhanced_gdf, features_df, target_series, quality_report

def main():
    """Example usage of the WildfireDataProcessor."""
    
    # Initialize processor
    processor = WildfireDataProcessor()
    
    # Run complete pipeline
    processed_gdf, features_df, target_series, quality_report = processor.process_pipeline()
    
    # Print summary
    print(f"\n=== Wildfire Data Processing Summary ===")
    print(f"Total records processed: {len(processed_gdf)}")
    print(f"Features for ML: {len(features_df.columns)}")
    print(f"Target variable distribution:")
    print(target_series.value_counts())
    
    return processed_gdf, features_df, target_series, quality_report

if __name__ == "__main__":
    main() 