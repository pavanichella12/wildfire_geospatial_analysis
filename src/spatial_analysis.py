"""
Spatial Analysis Module for Wildfire Data

This module demonstrates advanced geospatial analysis techniques including:
- Spatial clustering and hot spot analysis
- Spatial autocorrelation analysis
- Point pattern analysis
- Spatial interpolation and density estimation
- Advanced spatial statistics

Author: [Your Name]
Date: 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import logging
import warnings
warnings.filterwarnings('ignore')

# Spatial analysis libraries
try:
    import pysal
    from pysal.explore.esda import Moran, Geary, Local_Moran
    from pysal.explore.esda import G_Local, Join_Counts
    from pysal.weights import Queen, KNN, DistanceBand
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    logging.warning("PySAL not available. Some spatial analysis functions will be limited.")

logger = logging.getLogger(__name__)

class WildfireSpatialAnalyzer:
    """
    Advanced spatial analysis for wildfire data.
    
    This class demonstrates sophisticated geospatial analysis techniques including:
    - Spatial clustering and hot spot detection
    - Spatial autocorrelation analysis
    - Point pattern analysis
    - Density estimation and interpolation
    - Advanced spatial statistics
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        """
        Initialize the spatial analyzer.
        
        Args:
            gdf (gpd.GeoDataFrame): Geospatial wildfire data
        """
        self.gdf = gdf.copy()
        self.results = {}
        
        # Ensure we have a projected coordinate system for accurate analysis
        if self.gdf.crs.is_geographic:
            logger.info("Converting to projected coordinate system for spatial analysis")
            # Convert to a suitable projected CRS (Web Mercator for continental US)
            self.gdf = self.gdf.to_crs('EPSG:3857')
    
    def perform_spatial_clustering(self, method='dbscan', **kwargs) -> dict:
        """
        Perform spatial clustering analysis on wildfire points.
        
        Args:
            method (str): Clustering method ('dbscan', 'kmeans', 'hierarchical')
            **kwargs: Additional parameters for clustering algorithms
            
        Returns:
            dict: Clustering results and statistics
        """
        logger.info(f"Performing {method} spatial clustering...")
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in self.gdf.geometry])
        
        if method == 'dbscan':
            # DBSCAN for density-based clustering
            eps = kwargs.get('eps', 50000)  # 50km default
            min_samples = kwargs.get('min_samples', 5)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            labels = clustering.labels_
            
        elif method == 'kmeans':
            # K-means clustering
            n_clusters = kwargs.get('n_clusters', 10)
            
            clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
            labels = clustering.labels_
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Add cluster labels to the dataframe
        self.gdf[f'{method}_cluster'] = labels
        
        # Calculate cluster statistics
        cluster_stats = self.gdf.groupby(f'{method}_cluster').agg({
            'TOTALACRES': ['count', 'mean', 'sum'],
            'FIREYEAR': ['min', 'max', 'mean']
        }).round(2)
        
        # Calculate spatial extent of clusters
        cluster_extents = {}
        for cluster_id in self.gdf[f'{method}_cluster'].unique():
            if cluster_id != -1:  # Skip noise points for DBSCAN
                cluster_points = self.gdf[self.gdf[f'{method}_cluster'] == cluster_id]
                cluster_extents[cluster_id] = {
                    'count': len(cluster_points),
                    'bbox': cluster_points.total_bounds,
                    'area': cluster_points.geometry.union_all().convex_hull.area / 1e6  # km²
                }
        
        results = {
            'method': method,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'noise_points': np.sum(labels == -1) if -1 in labels else 0,
            'cluster_stats': cluster_stats,
            'cluster_extents': cluster_extents,
            'parameters': kwargs
        }
        
        self.results['clustering'] = results
        logger.info(f"Clustering completed: {results['n_clusters']} clusters found")
        
        return results
    
    def analyze_spatial_autocorrelation(self, variable='TOTALACRES', method='moran') -> dict:
        """
        Perform spatial autocorrelation analysis.
        
        Args:
            variable (str): Variable to analyze
            method (str): Autocorrelation method ('moran', 'geary', 'local_moran')
            
        Returns:
            dict: Spatial autocorrelation results
        """
        if not SPATIAL_AVAILABLE:
            logger.warning("PySAL not available. Skipping spatial autocorrelation analysis.")
            return {}
        
        logger.info(f"Analyzing spatial autocorrelation for {variable} using {method}...")
        
        # Prepare data
        data = self.gdf[variable].dropna()
        coords = np.array([[point.x, point.y] for point in self.gdf.loc[data.index].geometry])
        
        # Create spatial weights matrix
        weights = DistanceBand.from_array(coords, threshold=100000)  # 100km threshold
        
        results = {}
        
        if method == 'moran':
            # Global Moran's I
            moran = Moran(data, weights)
            results = {
                'moran_i': moran.I,
                'p_value': moran.p_norm,
                'z_score': moran.z_norm,
                'interpretation': self._interpret_moran(moran.I, moran.p_norm)
            }
            
        elif method == 'geary':
            # Geary's C
            geary = Geary(data, weights)
            results = {
                'geary_c': geary.C,
                'p_value': geary.p_norm,
                'z_score': geary.z_norm
            }
            
        elif method == 'local_moran':
            # Local Moran's I
            local_moran = Local_Moran(data, weights)
            results = {
                'local_moran_i': local_moran.Is,
                'p_values': local_moran.p_sim,
                'z_scores': local_moran.z_scores
            }
        
        self.results['spatial_autocorrelation'] = results
        return results
    
    def detect_hot_spots(self, variable='TOTALACRES', method='getis_ord') -> dict:
        """
        Detect spatial hot spots using various methods.
        
        Args:
            variable (str): Variable to analyze
            method (str): Hot spot detection method
            
        Returns:
            dict: Hot spot analysis results
        """
        logger.info(f"Detecting hot spots for {variable} using {method}...")
        
        # Prepare data
        data = self.gdf[variable].dropna()
        coords = np.array([[point.x, point.y] for point in self.gdf.loc[data.index].geometry])
        
        # Create spatial weights
        weights = DistanceBand.from_array(coords, threshold=100000)
        
        if method == 'getis_ord':
            # Getis-Ord Gi* statistic
            g_local = G_Local(data, weights, transform='R')
            
            # Identify hot and cold spots
            hot_spots = g_local.z_scores > 1.96  # 95% confidence
            cold_spots = g_local.z_scores < -1.96
            
            results = {
                'z_scores': g_local.z_scores,
                'p_values': g_local.p_sim,
                'hot_spots': hot_spots,
                'cold_spots': cold_spots,
                'n_hot_spots': np.sum(hot_spots),
                'n_cold_spots': np.sum(cold_spots)
            }
        
        self.results['hot_spots'] = results
        return results
    
    def analyze_point_patterns(self) -> dict:
        """
        Analyze spatial point patterns and distributions.
        
        Returns:
            dict: Point pattern analysis results
        """
        logger.info("Analyzing spatial point patterns...")
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in self.gdf.geometry])
        
        # Calculate nearest neighbor distances
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=2)  # k=2 to get nearest neighbor
        nearest_neighbor_distances = distances[:, 1]  # Skip self
        
        # Calculate spatial statistics
        mean_nn_distance = np.mean(nearest_neighbor_distances)
        std_nn_distance = np.std(nearest_neighbor_distances)
        
        # Calculate point density
        bbox = self.gdf.total_bounds
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        point_density = len(self.gdf) / area
        
        # Calculate spatial distribution metrics
        centroid = self.gdf.geometry.union_all().centroid
        distances_to_centroid = [point.distance(centroid) for point in self.gdf.geometry]
        
        results = {
            'total_points': len(self.gdf),
            'area': area,
            'point_density': point_density,
            'mean_nn_distance': mean_nn_distance,
            'std_nn_distance': std_nn_distance,
            'mean_distance_to_centroid': np.mean(distances_to_centroid),
            'std_distance_to_centroid': np.std(distances_to_centroid),
            'spatial_extent': {
                'x_min': bbox[0], 'x_max': bbox[2],
                'y_min': bbox[1], 'y_max': bbox[3]
            }
        }
        
        self.results['point_patterns'] = results
        return results
    
    def calculate_spatial_density(self, method='kernel', **kwargs) -> dict:
        """
        Calculate spatial density of wildfire points.
        
        Args:
            method (str): Density estimation method
            **kwargs: Additional parameters
            
        Returns:
            dict: Density analysis results
        """
        logger.info(f"Calculating spatial density using {method}...")
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in self.gdf.geometry])
        
        if method == 'kernel':
            # Simple kernel density estimation
            from scipy.stats import gaussian_kde
            
            # Calculate kernel density
            kde = gaussian_kde(coords.T)
            
            # Create grid for density estimation
            x_min, y_min, x_max, y_max = self.gdf.total_bounds
            grid_size = kwargs.get('grid_size', 100)
            
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Calculate density at grid points
            grid_coords = np.vstack([X.ravel(), Y.ravel()])
            density = kde(grid_coords).reshape(X.shape)
            
            results = {
                'density_grid': density,
                'x_grid': x_grid,
                'y_grid': y_grid,
                'max_density': np.max(density),
                'mean_density': np.mean(density),
                'density_percentiles': np.percentile(density, [25, 50, 75, 90, 95, 99])
            }
        
        self.results['spatial_density'] = results
        return results
    
    def _interpret_moran(self, moran_i: float, p_value: float) -> str:
        """Interpret Moran's I results."""
        if p_value > 0.05:
            return "No significant spatial autocorrelation"
        elif moran_i > 0:
            return "Positive spatial autocorrelation (clustering)"
        else:
            return "Negative spatial autocorrelation (dispersion)"
    
    def generate_spatial_report(self) -> dict:
        """
        Generate a comprehensive spatial analysis report.
        
        Returns:
            dict: Complete spatial analysis report
        """
        logger.info("Generating comprehensive spatial analysis report...")
        
        report = {
            'summary': {
                'total_points': len(self.gdf),
                'spatial_extent': self.gdf.total_bounds,
                'coordinate_system': str(self.gdf.crs)
            },
            'point_patterns': self.analyze_point_patterns(),
            'clustering': self.perform_spatial_clustering(),
            'spatial_autocorrelation': self.analyze_spatial_autocorrelation(),
            'hot_spots': self.detect_hot_spots(),
            'density': self.calculate_spatial_density()
        }
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: dict) -> list:
        """Generate recommendations based on spatial analysis results."""
        recommendations = []
        
        # Point pattern recommendations
        if report['point_patterns']['point_density'] > 1e-6:
            recommendations.append("High point density detected - consider clustering analysis")
        
        # Clustering recommendations
        if 'clustering' in report and report['clustering']['n_clusters'] > 5:
            recommendations.append("Multiple spatial clusters detected - investigate regional patterns")
        
        # Hot spot recommendations
        if 'hot_spots' in report and report['hot_spots']['n_hot_spots'] > 0:
            recommendations.append("Hot spots detected - prioritize these areas for monitoring")
        
        return recommendations

def main():
    """Example usage of the WildfireSpatialAnalyzer."""
    
    # Load processed data
    from data_processing import WildfireDataProcessor
    
    processor = WildfireDataProcessor()
    processed_gdf, _, _, _ = processor.process_pipeline()
    
    # Initialize spatial analyzer
    analyzer = WildfireSpatialAnalyzer(processed_gdf)
    
    # Generate comprehensive report
    report = analyzer.generate_spatial_report()
    
    # Print summary
    print("\n=== Spatial Analysis Summary ===")
    print(f"Total points analyzed: {report['summary']['total_points']}")
    print(f"Point density: {report['point_patterns']['point_density']:.2e} points/km²")
    print(f"Clusters found: {report['clustering']['n_clusters']}")
    print(f"Hot spots detected: {report['hot_spots']['n_hot_spots']}")
    
    return report

if __name__ == "__main__":
    main() 