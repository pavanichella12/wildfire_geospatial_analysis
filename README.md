# 🔥 Wildfire Risk Analysis & Predictive Modeling System

## 📋 Project Overview
This project demonstrates advanced geospatial analysis capabilities using USFS Fire Occurrence data to build a comprehensive wildfire risk assessment and predictive modeling system. Perfect for showcasing geospatial analysis skills to potential employers.

## 🎯 Key Features
- **📊 Data Analysis**: Comprehensive exploration of 580,000+ wildfire occurrence points
- **🗺️ Spatial Analysis**: Advanced geospatial techniques including clustering and hot spot analysis
- **📈 Interactive Dashboard**: Web-based visualization system with real-time filtering
- **🔧 Automated Workflows**: Reproducible data processing and modeling pipelines
- **📱 Professional UI**: Responsive design with intuitive controls

## 🛠️ Technical Stack
- **Python**: Core analysis and modeling
- **GeoPandas**: Geospatial data manipulation
- **Scikit-learn**: Machine learning models
- **Plotly/Folium**: Interactive visualizations
- **Streamlit**: Web dashboard interface
- **Git**: Version control and documentation

## 📁 Project Structure
```
wildfire/
├── data/
│   └── National_USFS_Fire_Occurrence_Point_(Feature_Layer).geojson
├── src/
│   ├── data_processing.py      # Data cleaning and preprocessing
│   ├── spatial_analysis.py     # Geospatial analysis functions
│   ├── ml_models.py           # Machine learning models
│   └── visualization.py       # Plotting and visualization functions
├── dashboard/
│   └── app_simple.py          # Streamlit dashboard (main)
├── requirements.txt
├── README.md
└── DEPLOYMENT_GUIDE.md
```

## 🚀 Quick Start

### Option 1: Run Locally
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd wildfire

# 2. Create virtual environment
python -m venv wildfire_env
source wildfire_env/bin/activate  # On Windows: wildfire_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run dashboard/app_simple.py
```

### Option 2: Deploy to Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path to: `dashboard/app_simple.py`
5. Deploy!

## 📊 Dashboard Features

### 🎛️ Interactive Controls
- **Sample Size**: Adjust number of fires displayed (100-5,000)
- **Color Options**: Color by Fire Size, Year, Cause, or Size Category
- **Map Styles**: Light, Dark, Satellite, Terrain backgrounds
- **Filters**: Year range and fire size range filtering

### 📈 Visualizations
- **🗺️ Interactive Maps**: Both Plotly and Folium maps
- **📊 Temporal Analysis**: Wildfires by year with proper data cleaning
- **🔥 Cause Analysis**: Fire cause distribution with statistics
- **📋 Data Quality**: Comprehensive data quality metrics

### 🎨 Color Legends
- **Fire Size**: Red (large), Yellow (medium), Green (small), Blue (very small)
- **Year**: Red (recent), Yellow (mid-period), Green (historical)
- **Cause**: Distribution with percentages
- **Statistics**: Real-time metrics and sample rates

## 📈 Skills Demonstrated
- **Advanced Data Analysis**: Complex geospatial data manipulation
- **Machine Learning**: Predictive modeling with spatial data
- **Automation**: Reproducible workflows and pipelines
- **Visualization**: Interactive maps and dashboards
- **Documentation**: Comprehensive technical documentation
- **Deployment**: Cloud deployment and version control

## 🔧 Technical Highlights
- **Data Processing**: Full pipeline processing 580K+ records
- **Performance Optimization**: Intelligent sampling for large datasets
- **Error Handling**: Robust error handling and user feedback
- **Professional UI**: Clean, responsive interface
- **Real-time Filtering**: Dynamic data filtering and visualization

## 📚 Documentation
- `DEPLOYMENT_GUIDE.md`: Complete deployment instructions
- `PROJECT_SUMMARY.md`: Detailed project overview
- `STREAMLIT_CLOUD_DEPLOYMENT.md`: Cloud deployment guide

## 🌟 Perfect for Job Applications
This project showcases:
- **Geospatial Analysis**: Advanced spatial data processing
- **Data Science**: Machine learning and statistical analysis
- **Web Development**: Interactive dashboard creation
- **DevOps**: Cloud deployment and version control
- **Documentation**: Professional technical writing

## 📞 Support
For questions or issues, please refer to the documentation files or create an issue in the repository.

---
**Built with ❤️ for geospatial analysis and data science** 