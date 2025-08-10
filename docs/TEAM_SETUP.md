# Energy Recommendation Engine - Team Setup Guide

## 🚀 Quick Start (5 minutes)

### 1. Repository Access
```bash
git clone https://github.com/cyranothebard/energy-recommendation-engine.git
cd energy-recommendation-engine
```

### 2. AWS Configuration
You should have received AWS credentials. Configure them:
```bash
aws configure
# Use the credentials provided in team communication
```

### 3. Python Environment Setup
```bash
# Create conda environment
conda create -n energy-rec python=3.9
conda activate energy-rec

# Install requirements
pip install pandas numpy boto3 matplotlib seaborn jupyter
```

### 4. Verify Setup
```bash
# Test AWS access
aws s3 ls s3://energy-recommendation-project-246773437083/

# Should show: raw-data/ folder
```

## 📁 Project Structure

```
energy-recommendation-engine/
├── src/energy_recommender/           # 🆕 PRODUCTION CODE
│   ├── features/engineering.py       # Feature engineering pipeline
│   ├── models/compliance.py          # Compliance prediction
│   ├── models/optimization.py        # Portfolio optimization
│   └── pipeline.py                   # End-to-end orchestration
├── notebooks/
│   ├── exploratory/                  # Research notebooks (Brandon's work)
│   └── clean/                        # 🆕 TEAM DEMOS (coming soon)
├── data/sample/                      # Sample data for development
├── config/                           # Configuration files
└── tests/                           # Unit tests (future)
```

## 🎯 Team Roles & Next Steps

### 👨‍💻 Brandon (Technical Lead)
- **Status**: Pipeline validation complete ✅
- **Next**: ML model training + distributed processing
- **Focus**: Spark implementation, model optimization

### 👩‍💻 Dashboard Developer
- **Goal**: Interactive recommendation visualization
- **Data Source**: Use `pipeline.py` outputs for sample data
- **Tech Stack**: Plotly Dash, Bootstrap
- **Start Here**: Create `notebooks/clean/dashboard_demo.ipynb`

### 📊 Documentation Lead  
- **Goal**: Evaluation framework + presentation
- **Focus**: A/B testing metrics, performance analysis
- **Start Here**: Create `notebooks/clean/evaluation_analysis.ipynb`

## 🔧 Development Workflow

### Working with the Pipeline
```python
# Import the production modules
from src.energy_recommender.pipeline import run_end_to_end_pipeline
from src.energy_recommender.features.engineering import engineer_building_features_comprehensive

# Load sample data (provided)
metadata_df = pd.read_csv('data/sample/sample_buildings.csv')

# Run complete pipeline
results = run_end_to_end_pipeline(metadata_df)

# Access outputs
building_features = results['building_features']
portfolio = results['portfolio_results']
```

### Sample Data Available
- **Location**: `s3://energy-recommendation-project-246773437083/raw-data/`
- **Sample Buildings**: 50 MA buildings with full timeseries
- **Metadata**: 8,111 buildings with characteristics
- **Format**: CSV files, ready to use

## 📋 Key Accomplishments (Context)

### ✅ Validated Pipeline Results
- **Feature Engineering**: 13 building types, 34 HVAC systems
- **Compliance Modeling**: 36.3% average compliance (realistic)
- **Portfolio Optimization**: 5.4% grid reduction (industry benchmark)
- **Performance**: <30 seconds, <50MB memory

### ✅ AWS Infrastructure  
- **Budget Controls**: $100/month with alerts
- **Data Access**: All team members have S3 access
- **Scalability**: Ready for SageMaker when needed

## 🚨 Important Notes

### Data Handling
- **DO NOT** download full dataset locally (14.5 GiB)
- **USE** sample data in `data/sample/` for development
- **AWS costs** are monitored - avoid unnecessary compute

### Code Organization
- **Research code**: `notebooks/exploratory/` (Brandon's domain)
- **Production code**: `src/energy_recommender/` (shared team code)
- **Clean demos**: `notebooks/clean/` (for team collaboration)

### Git Workflow
- **Main branch**: Always deployable
- **Feature branches**: For individual development
- **Pull requests**: For code review

## 📞 Getting Help

### Immediate Issues
1. **AWS Access Problems**: Check credentials with `aws sts get-caller-identity`
2. **Python Environment**: Ensure you're in the `energy-rec` conda environment
3. **Data Access**: Verify S3 permissions with the test command above

### Team Communication
- **Technical Questions**: Tag Brandon in team chat
- **AWS Issues**: Check AWS console budget dashboard
- **Git Problems**: Use GitHub issues or team chat

## 🎯 Success Criteria

### Week 1 Goals
- **Dashboard Developer**: Basic visualization of portfolio results
- **Documentation Lead**: Evaluation metrics framework
- **Technical Lead**: ML model training pipeline

### Integration Points
- **Data Interface**: All modules use standard DataFrame formats
- **API Design**: Clean function signatures for easy integration
- **Performance**: All components process sample data in <10 seconds

---

## 🚀 Ready to Start?

1. **Complete setup steps above** (should take 5-10 minutes)
2. **Run the pipeline test** to verify everything works
3. **Check team chat** for role-specific guidance
4. **Begin development** in your assigned area

The pipeline is validated and ready - let's build something great! 🔥