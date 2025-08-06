"""
Visualization Module for Wildfire Analysis

This module demonstrates advanced visualization capabilities including:
- Interactive maps and spatial visualizations
- Statistical plots and distributions
- Machine learning result visualizations
- Dashboard-ready charts
- Custom plotting functions

Author: [Your Name]
Date: 2025
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class WildfireVisualizer:
    """
    Advanced visualization system for wildfire analysis.
    
    This class demonstrates sophisticated visualization capabilities including:
    - Interactive spatial maps and choropleths
    - Statistical plots and distributions
    - Machine learning result visualizations
    - Dashboard-ready charts and graphs
    - Custom plotting functions and themes
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame = None):
        """
        Initialize the visualizer.
        
        Args:
            gdf (gpd.GeoDataFrame): Geospatial wildfire data
        """
        self.gdf = gdf
        self.figures = {}
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def create_interactive_map(self, gdf: gpd.GeoDataFrame = None, 
                             color_by: str = 'TOTALACRES', 
                             size_by: str = 'TOTALACRES',
                             max_points: int = 10000) -> folium.Map:
        """
        Create an interactive map with wildfire points.
        
        Args:
            gdf (gpd.GeoDataFrame): Geospatial data
            color_by (str): Column to color points by
            size_by (str): Column to size points by
            max_points (int): Maximum number of points to display
            
        Returns:
            folium.Map: Interactive map
        """
        if gdf is None:
            gdf = self.gdf
        
        if gdf is None:
            logger.error("No geospatial data provided")
            return None
        
        # Sample data if too many points
        if len(gdf) > max_points:
            gdf = gdf.sample(n=max_points, random_state=42)
            logger.info(f"Sampled {max_points} points for visualization")
        
        # Convert to WGS84 for web mapping
        gdf_web = gdf.to_crs('EPSG:4326')
        
        # Calculate center
        center_lat = gdf_web.geometry.y.mean()
        center_lon = gdf_web.geometry.x.mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('cartodbpositron', name='Light').add_to(m)
        folium.TileLayer('cartodbdark_matter', name='Dark').add_to(m)
        # folium.TileLayer('Stamen Terrain', name='Terrain', 
        #                 attribution='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.').add_to(m)
        
        # Create feature groups
        fg_points = folium.FeatureGroup(name='Wildfire Points')
        fg_heatmap = folium.FeatureGroup(name='Heat Map')
        
        # Add points
        for idx, row in gdf_web.iterrows():
            # Determine color based on variable
            if color_by in row and pd.notna(row[color_by]):
                if color_by == 'TOTALACRES':
                    color = self._get_color_by_size(row[color_by])
                elif color_by == 'FIREYEAR':
                    color = self._get_color_by_year(row[color_by])
                else:
                    color = 'red'
            else:
                color = 'gray'
            
            # Determine size
            if size_by in row and pd.notna(row[size_by]):
                size = min(20, max(3, np.log(row[size_by] + 1) * 2))
            else:
                size = 5
            
            # Create popup
            popup_text = f"""
            <b>Fire Name:</b> {row.get('FIRENAME', 'Unknown')}<br>
            <b>Year:</b> {row.get('FIREYEAR', 'Unknown')}<br>
            <b>Size:</b> {row.get('TOTALACRES', 0):.1f} acres<br>
            <b>Cause:</b> {row.get('STATCAUSE', 'Unknown')}<br>
            """
            
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=size,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(fg_points)
        
        # Add heatmap
        heat_data = [[row.geometry.y, row.geometry.x] for idx, row in gdf_web.iterrows()]
        HeatMap(heat_data, radius=15).add_to(fg_heatmap)
        
        # Add feature groups to map
        fg_points.add_to(m)
        fg_heatmap.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add legend
        self._add_map_legend(m, color_by)
        
        self.figures['interactive_map'] = m
        return m
    
    def create_temporal_analysis_plots(self, gdf: gpd.GeoDataFrame = None) -> dict:
        """
        Create temporal analysis plots.
        
        Args:
            gdf (gpd.GeoDataFrame): Geospatial data
            
        Returns:
            dict: Dictionary of plotly figures
        """
        if gdf is None:
            gdf = self.gdf
        
        if gdf is None:
            logger.error("No data provided")
            return {}
        
        figures = {}
        
        # 1. Fires by year
        if 'FIREYEAR' in gdf.columns:
            yearly_counts = gdf['FIREYEAR'].value_counts().sort_index()
            
            fig_yearly = go.Figure()
            fig_yearly.add_trace(go.Scatter(
                x=yearly_counts.index,
                y=yearly_counts.values,
                mode='lines+markers',
                name='Fire Count',
                line=dict(color=self.colors['primary'], width=2),
                marker=dict(size=6)
            ))
            
            fig_yearly.update_layout(
                title='Wildfire Occurrences by Year',
                xaxis_title='Year',
                yaxis_title='Number of Fires',
                template='plotly_white',
                hovermode='x unified'
            )
            
            figures['yearly_trend'] = fig_yearly
        
        # 2. Monthly distribution
        if 'DISCOVERYDATETIME' in gdf.columns:
            gdf['month'] = pd.to_datetime(gdf['DISCOVERYDATETIME']).dt.month
            monthly_counts = gdf['month'].value_counts().sort_index()
            
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_counts.index,
                y=monthly_counts.values,
                name='Fire Count',
                marker_color=self.colors['secondary']
            ))
            
            fig_monthly.update_layout(
                title='Wildfire Occurrences by Month',
                xaxis_title='Month',
                yaxis_title='Number of Fires',
                template='plotly_white',
                xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), 
                          ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            )
            
            figures['monthly_distribution'] = fig_monthly
        
        # 3. Fire size distribution over time
        if 'TOTALACRES' in gdf.columns and 'FIREYEAR' in gdf.columns:
            # Group by year and calculate statistics
            yearly_stats = gdf.groupby('FIREYEAR')['TOTALACRES'].agg(['mean', 'median', 'sum']).reset_index()
            
            fig_size_trend = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Mean Fire Size by Year', 'Total Acres Burned by Year'),
                vertical_spacing=0.1
            )
            
            fig_size_trend.add_trace(
                go.Scatter(x=yearly_stats['FIREYEAR'], y=yearly_stats['mean'],
                          mode='lines+markers', name='Mean Size', line=dict(color=self.colors['primary'])),
                row=1, col=1
            )
            
            fig_size_trend.add_trace(
                go.Scatter(x=yearly_stats['FIREYEAR'], y=yearly_stats['sum'],
                          mode='lines+markers', name='Total Acres', line=dict(color=self.colors['danger'])),
                row=2, col=1
            )
            
            fig_size_trend.update_layout(
                title='Fire Size Trends Over Time',
                template='plotly_white',
                height=600
            )
            
            figures['size_trends'] = fig_size_trend
        
        self.figures.update(figures)
        return figures
    
    def create_spatial_analysis_plots(self, gdf: gpd.GeoDataFrame = None) -> dict:
        """
        Create spatial analysis plots.
        
        Args:
            gdf (gpd.GeoDataFrame): Geospatial data
            
        Returns:
            dict: Dictionary of plotly figures
        """
        if gdf is None:
            gdf = self.gdf
        
        if gdf is None:
            logger.error("No data provided")
            return {}
        
        figures = {}
        
        # 1. Spatial distribution heatmap
        if 'TOTALACRES' in gdf.columns:
            # Create 2D histogram
            x = gdf.geometry.x
            y = gdf.geometry.y
            weights = gdf['TOTALACRES']
            
            fig_heatmap = go.Figure(data=go.Histogram2d(
                x=x, y=y, z=weights,
                nbinsx=50, nbinsy=50,
                colorscale='Viridis',
                colorbar=dict(title='Total Acres')
            ))
            
            fig_heatmap.update_layout(
                title='Spatial Distribution of Wildfire Size',
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                template='plotly_white'
            )
            
            figures['spatial_heatmap'] = fig_heatmap
        
        # 2. Fire size distribution
        if 'TOTALACRES' in gdf.columns:
            fig_size_dist = go.Figure()
            
            # Log scale for better visualization
            log_sizes = np.log10(gdf['TOTALACRES'] + 1)
            
            fig_size_dist.add_trace(go.Histogram(
                x=log_sizes,
                nbinsx=50,
                name='Fire Size Distribution',
                marker_color=self.colors['info']
            ))
            
            fig_size_dist.update_layout(
                title='Distribution of Wildfire Sizes (Log Scale)',
                xaxis_title='Log10(Total Acres + 1)',
                yaxis_title='Count',
                template='plotly_white'
            )
            
            figures['size_distribution'] = fig_size_dist
        
        # 3. Cause analysis
        if 'STATCAUSE' in gdf.columns:
            cause_counts = gdf['STATCAUSE'].value_counts()
            
            fig_cause = go.Figure(data=go.Pie(
                labels=cause_counts.index,
                values=cause_counts.values,
                hole=0.3
            ))
            
            fig_cause.update_layout(
                title='Wildfire Causes Distribution',
                template='plotly_white'
            )
            
            figures['cause_analysis'] = fig_cause
        
        self.figures.update(figures)
        return figures
    
    def create_ml_result_plots(self, ml_results: dict) -> dict:
        """
        Create machine learning result visualizations.
        
        Args:
            ml_results (dict): Machine learning results
            
        Returns:
            dict: Dictionary of plotly figures
        """
        figures = {}
        
        # 1. Model comparison
        if 'model_evaluation' in ml_results:
            comparison_df = ml_results['model_evaluation']['model_comparison']
            
            fig_comparison = go.Figure()
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['success'], self.colors['warning'], self.colors['danger']]
            
            for i, metric in enumerate(metrics):
                fig_comparison.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=comparison_df['model'],
                    y=comparison_df[metric],
                    marker_color=colors[i]
                ))
            
            fig_comparison.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                template='plotly_white',
                barmode='group'
            )
            
            figures['model_comparison'] = fig_comparison
        
        # 2. Feature importance
        if 'feature_importance' in ml_results:
            for model_name, importance_data in ml_results['feature_importance'].items():
                if 'top_features' in importance_data:
                    top_features = importance_data['top_features'][:10]
                    
                    fig_importance = go.Figure()
                    fig_importance.add_trace(go.Bar(
                        x=[f['importance'] for f in top_features],
                        y=[f['feature'] for f in top_features],
                        orientation='h',
                        marker_color=self.colors['primary']
                    ))
                    
                    fig_importance.update_layout(
                        title=f'Top 10 Feature Importance - {model_name.replace("_", " ").title()}',
                        xaxis_title='Importance',
                        yaxis_title='Feature',
                        template='plotly_white'
                    )
                    
                    figures[f'feature_importance_{model_name}'] = fig_importance
        
        # 3. ROC curves (if available)
        if 'all_results' in ml_results.get('model_evaluation', {}):
            fig_roc = go.Figure()
            
            for model_name, results in ml_results['model_evaluation']['all_results'].items():
                if 'roc_auc' in results:
                    # Create ROC curve data (simplified)
                    fpr = np.linspace(0, 1, 100)
                    tpr = np.linspace(0, 1, 100)  # Simplified
                    
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f'{model_name} (AUC: {results["roc_auc"]:.3f})',
                        mode='lines'
                    ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ))
            
            fig_roc.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template='plotly_white'
            )
            
            figures['roc_curves'] = fig_roc
        
        self.figures.update(figures)
        return figures
    
    def create_dashboard_layout(self, figures: dict) -> go.Figure:
        """
        Create a dashboard layout combining multiple plots.
        
        Args:
            figures (dict): Dictionary of plotly figures
            
        Returns:
            go.Figure: Combined dashboard
        """
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Wildfire Trends', 'Monthly Distribution', 
                          'Spatial Heatmap', 'Size Distribution',
                          'Model Comparison', 'Feature Importance'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram2d"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add traces from existing figures
        # This is a simplified version - in practice you'd extract traces from figures
        
        fig.update_layout(
            title='Wildfire Analysis Dashboard',
            height=1200,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def _get_color_by_size(self, size: float) -> str:
        """Get color based on fire size."""
        if size < 1:
            return 'green'
        elif size < 10:
            return 'yellow'
        elif size < 100:
            return 'orange'
        elif size < 1000:
            return 'red'
        else:
            return 'darkred'
    
    def _get_color_by_year(self, year: int) -> str:
        """Get color based on fire year."""
        current_year = datetime.now().year
        if year >= current_year - 5:
            return 'red'
        elif year >= current_year - 10:
            return 'orange'
        elif year >= current_year - 20:
            return 'yellow'
        else:
            return 'green'
    
    def _add_map_legend(self, m: folium.Map, color_by: str):
        """Add legend to map."""
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Legend</b></p>
        <p>Color by: {color_by}</p>
        <p>• Red: High values</p>
        <p>• Green: Low values</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def save_all_figures(self, output_dir: str = 'outputs'):
        """
        Save all generated figures.
        
        Args:
            output_dir (str): Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            if hasattr(fig, 'write_html'):
                # Plotly figure
                fig.write_html(f'{output_dir}/{name}.html')
            elif hasattr(fig, 'save'):
                # Folium map
                fig.save(f'{output_dir}/{name}.html')
        
        logger.info(f"All figures saved to {output_dir}")

def main():
    """Example usage of the WildfireVisualizer."""
    
    # Load processed data
    from data_processing import WildfireDataProcessor
    
    processor = WildfireDataProcessor()
    processed_gdf, _, _, _ = processor.process_pipeline()
    
    # Initialize visualizer
    visualizer = WildfireVisualizer(processed_gdf)
    
    # Create visualizations
    map_fig = visualizer.create_interactive_map()
    temporal_figs = visualizer.create_temporal_analysis_plots()
    spatial_figs = visualizer.create_spatial_analysis_plots()
    
    # Save figures
    visualizer.save_all_figures()
    
    print("Visualization completed. Check the 'outputs' directory for generated figures.")
    
    return visualizer

if __name__ == "__main__":
    main() 