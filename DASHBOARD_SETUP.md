# Energy Recommendation Engine Dashboard Setup Guide

## Overview

This document provides a comprehensive guide to the Plotly Dash dashboard setup for the Energy Recommendation Engine project. The dashboard provides interactive visualizations and analysis tools for energy efficiency recommendations.

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation Steps

1. **Install Dependencies**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   ```bash
   python3 run_dashboard.py
   ```

3. **Access Dashboard**
   Open your browser and navigate to: http://localhost:8050

## 📁 Project Structure

```
src/dashboard/
├── __init__.py          # Module initialization
├── app.py              # Main Dash application
├── components.py       # Reusable UI components
├── data_manager.py     # Data processing and management
└── config.py           # Configuration settings

run_dashboard.py        # Dashboard launcher script
test_dashboard.py       # Test suite
env.example            # Environment configuration template
```

## 🎯 Features

### Core Functionality
- **Interactive Visualizations**: Energy consumption trends, building type distribution, efficiency scores
- **Real-time Metrics**: Key performance indicators and cost analysis
- **Smart Recommendations**: AI-powered energy efficiency suggestions
- **Data Filtering**: Date range, building type, and efficiency score filters
- **Export Capabilities**: Download data in multiple formats

### Dashboard Components

#### 1. Key Metrics Cards
- Total Energy Saved
- Cost Reduction
- Efficiency Score
- Environmental Impact

#### 2. Interactive Charts
- **Energy Consumption Over Time**: Line chart showing daily energy usage patterns
- **Building Type Distribution**: Pie chart of building classifications
- **Efficiency Score Distribution**: Histogram of efficiency ratings
- **Temperature Correlation**: Scatter plot showing energy vs temperature relationship
- **Energy Heatmap**: Monthly/daily consumption patterns

#### 3. Recommendations Panel
- Prioritized energy efficiency recommendations
- Cost estimates and payback periods
- Potential savings calculations

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```bash
# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=True

# Data Configuration
DATA_PATH=data
CACHE_TIMEOUT=300

# Logging
LOG_LEVEL=INFO
LOG_FILE=dashboard.log
```

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_PORT` | 8050 | Server port number |
| `DASHBOARD_DEBUG` | True | Enable debug mode |
| `DATA_PATH` | data | Directory for data files |
| `LOG_LEVEL` | INFO | Logging verbosity |

## 📊 Data Management

### Sample Data Generation
The dashboard includes a comprehensive data generator that creates realistic energy consumption patterns:

- **Temporal Patterns**: Seasonal and weekly variations
- **Building Types**: Office, Residential, Industrial, Commercial
- **Environmental Factors**: Temperature, humidity, occupancy
- **Efficiency Metrics**: Calculated efficiency scores with correlations

### Data Processing
- Real-time metric calculations
- Trend analysis and forecasting
- Recommendation generation based on patterns
- Data quality validation

## 🧪 Testing

Run the test suite to verify installation:

```bash
python3 test_dashboard.py
```

The test suite validates:
- ✅ Module imports
- ✅ Configuration settings
- ✅ Data manager functionality
- ✅ Visualization components
- ✅ Dash application setup

## 🚀 Deployment

### Development Mode
```bash
python3 run_dashboard.py
```

### Production Mode
```bash
# Set environment variables
export DASHBOARD_DEBUG=False
export DASHBOARD_HOST=0.0.0.0
export DASHBOARD_PORT=8050

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8050 src.dashboard.app:app
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 8050
   lsof -i :8050
   
   # Kill the process or change port in .env
   ```

2. **Import Errors**
   ```bash
   # Reinstall dependencies
   python3 -m pip install -r requirements.txt --force-reinstall
   ```

3. **Dashboard Not Loading**
   - Check if the server is running: `curl http://localhost:8050`
   - Verify browser console for JavaScript errors
   - Check dashboard logs for Python errors

### Log Files
- Dashboard logs: `dashboard.log`
- Application logs: Console output

## 📈 Customization

### Adding New Visualizations
1. Add visualization function to `components.py`
2. Create callback in `app.py`
3. Add UI component to layout

### Integrating Real Data
1. Modify `data_manager.py` to load your data source
2. Update data schema if needed
3. Adjust visualization parameters

### Styling
- Bootstrap theme can be customized in `config.py`
- Chart colors and templates are configurable
- CSS customizations can be added to the layout

## 🔗 Integration

### API Integration
The dashboard is designed to integrate with:
- Energy monitoring systems
- Building management systems
- Weather data APIs
- Cost calculation services

### Database Integration
- PostgreSQL support via SQLAlchemy
- Redis caching for performance
- Data export capabilities

## 📚 Dependencies

### Core Dependencies
- **Dash**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning

### UI Dependencies
- **Bootstrap**: Responsive design
- **Dash Bootstrap Components**: UI components
- **Dash Core Components**: Interactive elements

### Development Dependencies
- **Pytest**: Testing framework
- **Black**: Code formatting
- **Flake8**: Linting
- **Pre-commit**: Git hooks

## 🎉 Success Indicators

Your dashboard setup is successful when:
- ✅ All tests pass: `python3 test_dashboard.py`
- ✅ Dashboard loads: http://localhost:8050
- ✅ Charts render properly
- ✅ Data updates in real-time
- ✅ Recommendations are generated

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test output
3. Check log files for errors
4. Verify configuration settings

---

**Dashboard Version**: 1.0.0  
**Last Updated**: January 2025  
**Compatibility**: Python 3.9+, Dash 3.0+ 