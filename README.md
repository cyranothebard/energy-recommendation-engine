# Energy Recommendation Engine

Production-quality ML system for building energy efficiency recommendations using distributed computing.

## Project Status
- ✅ Pipeline validation complete
- ✅ Dashboard environment setup complete
- 🔄 Team collaboration setup in progress

## Quick Start

### Dashboard Setup

The project includes a comprehensive Plotly Dash dashboard for energy efficiency analysis and recommendations.

#### Prerequisites
- Python 3.9+
- pip or conda package manager

#### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd energy-recommendation-engine
   ```

2. **Install dependencies**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. **Configure environment (optional)**
   ```bash
   cp env.example .env
   # Edit .env file with your configuration
   ```

4. **Run the dashboard**
   ```bash
   python3 run_dashboard.py
   ```

5. **Access the dashboard**
   Open your browser and navigate to: http://localhost:8050

#### Dashboard Features

- **Interactive Visualizations**: Energy consumption trends, building type distribution, efficiency scores
- **Real-time Metrics**: Key performance indicators and cost analysis
- **Smart Recommendations**: AI-powered energy efficiency suggestions
- **Data Filtering**: Date range, building type, and efficiency score filters
- **Export Capabilities**: Download data in multiple formats

#### Dashboard Structure

```
src/dashboard/
├── __init__.py          # Dashboard module initialization
├── app.py              # Main Dash application
├── components.py       # Reusable UI components and visualizations
├── data_manager.py     # Data processing and management
└── config.py           # Configuration settings
```

### Notebooks
See `notebooks/exploratory/` for demonstration notebooks.

## Team Structure
- Technical Lead: ML pipeline and distributed processing
- Dashboard Developer: Visualization and user interface  
- Documentation Lead: Evaluation framework and presentation

## Development

### Project Structure
```
energy-recommendation-engine/
├── src/
│   ├── dashboard/          # Dashboard application
│   ├── energy_recommender/ # Core ML components
│   └── pipeline.py         # Data processing pipeline
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
├── run_dashboard.py        # Dashboard launcher
└── env.example            # Environment configuration template
```

### Key Dependencies
- **Dash**: Interactive web application framework
- **Plotly**: Advanced data visualization
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Bootstrap**: Responsive UI components

### Configuration
The dashboard can be configured through environment variables or the `.env` file:
- `DASHBOARD_PORT`: Server port (default: 8050)
- `DASHBOARD_DEBUG`: Debug mode (default: True)
- `DATA_PATH`: Data directory path
- `LOG_LEVEL`: Logging level
