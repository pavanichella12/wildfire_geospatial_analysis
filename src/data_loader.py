"""
Data loader module for downloading wildfire data from cloud storage.
Supports multiple cloud storage providers for deployment flexibility.
"""

import os
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from typing import Optional, Union
import gzip
import tempfile
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudDataLoader:
    """
    Loads wildfire data from various cloud storage providers.
    Supports Google Drive, Dropbox, AWS S3, and local files.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_from_google_drive(self, file_id: str, filename: str) -> str:
        """
        Download file from Google Drive using file ID.
        
        Args:
            file_id: Google Drive file ID
            filename: Local filename to save as
            
        Returns:
            Path to downloaded file
        """
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return str(filepath)
            
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        
        logger.info(f"Downloading from Google Drive: {filename}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Successfully downloaded: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download from Google Drive: {e}")
            raise
    
    def download_from_dropbox(self, share_link: str, filename: str) -> str:
        """
        Download file from Dropbox using share link.
        
        Args:
            share_link: Dropbox share link
            filename: Local filename to save as
            
        Returns:
            Path to downloaded file
        """
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return str(filepath)
        
        # Convert Dropbox share link to direct download link
        direct_link = share_link.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
        direct_link = direct_link.replace('?dl=0', '').replace('?dl=1', '')
        
        logger.info(f"Downloading from Dropbox: {filename}")
        
        try:
            response = requests.get(direct_link, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Successfully downloaded: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to download from Dropbox: {e}")
            raise
    
    def download_from_s3(self, bucket_name: str, object_key: str, filename: str = None) -> str:
        """
        Download file from AWS S3.
        
        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key (file path in bucket)
            filename: Local filename to save as (optional)
            
        Returns:
            Path to downloaded file
        """
        if filename is None:
            filename = os.path.basename(object_key)
            
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return str(filepath)
        
        logger.info(f"Downloading from S3: s3://{bucket_name}/{object_key}")
        
        try:
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # Download file
            s3_client.download_file(bucket_name, object_key, str(filepath))
            
            logger.info(f"Successfully downloaded: {filepath}")
            return str(filepath)
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except ClientError as e:
            logger.error(f"S3 error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            raise
    
    def load_wildfire_data(self, source: str = "local", **kwargs) -> gpd.GeoDataFrame:
        """
        Load wildfire data from various sources.
        
        Args:
            source: Data source ("local", "google_drive", "dropbox")
            **kwargs: Source-specific parameters
            
        Returns:
            GeoDataFrame with wildfire data
        """
        filename = "National_USFS_Fire_Occurrence_Point_(Feature_Layer).geojson"
        
        if source == "local":
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
                
        elif source == "google_drive":
            file_id = kwargs.get("file_id")
            if not file_id:
                raise ValueError("Google Drive file_id is required")
            filepath = self.download_from_google_drive(file_id, filename)
            
        elif source == "dropbox":
            share_link = kwargs.get("share_link")
            if not share_link:
                raise ValueError("Dropbox share_link is required")
            filepath = self.download_from_dropbox(share_link, filename)
            
        elif source == "s3":
            bucket_name = kwargs.get("bucket_name")
            object_key = kwargs.get("object_key")
            if not bucket_name or not object_key:
                raise ValueError("S3 bucket_name and object_key are required")
            filepath = self.download_from_s3(bucket_name, object_key, filename)
            
        else:
            raise ValueError(f"Unsupported source: {source}")
        
        logger.info(f"Loading wildfire data from: {filepath}")
        
        try:
            # Try to read as GeoJSON with explicit driver
            gdf = gpd.read_file(filepath, driver='GeoJSON')
            logger.info(f"Successfully loaded {len(gdf)} wildfire records")
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to load wildfire data: {e}")
            # Try alternative methods
            try:
                logger.info("Trying alternative file reading method...")
                gdf = gpd.read_file(filepath, driver='GeoJSON')
                logger.info(f"Successfully loaded {len(gdf)} wildfire records with alternative method")
                return gdf
            except Exception as e2:
                logger.error(f"Alternative method also failed: {e2}")
                raise e

def get_data_loader() -> CloudDataLoader:
    """
    Factory function to create a data loader instance.
    """
    return CloudDataLoader()

# Convenience function for quick data loading
def load_wildfire_data(source: str = "local", **kwargs) -> gpd.GeoDataFrame:
    """
    Convenience function to load wildfire data.
    
    Args:
        source: Data source ("local", "google_drive", "dropbox")
        **kwargs: Source-specific parameters
        
    Returns:
        GeoDataFrame with wildfire data
    """
    loader = get_data_loader()
    return loader.load_wildfire_data(source, **kwargs) 