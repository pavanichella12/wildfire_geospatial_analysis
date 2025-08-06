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
            
        else:
            raise ValueError(f"Unsupported source: {source}")
        
        logger.info(f"Loading wildfire data from: {filepath}")
        
        try:
            gdf = gpd.read_file(filepath)
            logger.info(f"Successfully loaded {len(gdf)} wildfire records")
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to load wildfire data: {e}")
            raise

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