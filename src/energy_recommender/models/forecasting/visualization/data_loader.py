"""
Data Loading Utilities for Visualization Module

Handles loading of trained models, scenario data, and preprocessing
for visualization generation.
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional

def load_trained_model():
    """Load the trained LSTM model for visualization"""
    
    # Get the models directory relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "..", "trained_models")
    
    model_path = os.path.join(models_dir, "multi_cohort_lstm_model.pth")
    config_path = os.path.join(models_dir, "multi_cohort_lstm_config.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"❌ Model files not found:")
        print(f"   Model: {model_path}")
        print(f"   Config: {config_path}")
        print("   Please run train_lstm.py first to generate trained models.")
        return None
    
    try:
        # Import the LSTM architecture
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        sys.path.append(project_root)
        
        from src.energy_recommender.models.forecasting.pytorch_lstm_architecture import create_lstm_trainer
        
        # Create model instance
        lstm_model = create_lstm_trainer()
        
        # Load the trained model
        lstm_model.load_model(os.path.join(models_dir, "multi_cohort_lstm"))
        
        print(f"✅ Loaded trained LSTM model from {models_dir}")
        return lstm_model
        
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return None

def load_scenario_data():
    """Load or regenerate scenario data for visualization"""
    
    try:
        # Import scenario generation
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        sys.path.append(project_root)
        
        from src.energy_recommender.models.forecasting.demand_simulation import generate_all_demand_scenarios
        
        print("📊 Generating scenario data for visualization...")
        all_scenarios = generate_all_demand_scenarios()
        
        print(f"✅ Loaded {len(all_scenarios)} scenarios for visualization")
        return all_scenarios
        
    except Exception as e:
        print(f"❌ Failed to load scenario data: {str(e)}")
        return None

def load_existing_results():
    """Load existing analysis results if available"""
    
    # Get the results directory relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "..", "results")
    
    results = {}
    
    # Load RMSE analysis if available
    rmse_path = os.path.join(results_dir, "rmse_analysis_detailed.csv")
    if os.path.exists(rmse_path):
        results['rmse_analysis'] = pd.read_csv(rmse_path)
        print(f"✅ Loaded RMSE analysis from {rmse_path}")
    
    # Load sample predictions if available
    predictions_path = os.path.join(results_dir, "sample_predictions.csv")
    if os.path.exists(predictions_path):
        results['sample_predictions'] = pd.read_csv(predictions_path)
        print(f"✅ Loaded sample predictions from {predictions_path}")
    
    return results if results else None

def prepare_cohort_data(all_scenarios):
    """Prepare cohort-level data for visualization"""
    
    cohort_summary = {}
    
    for scenario_name, scenario_data in all_scenarios.items():
        grid_data = scenario_data['grid_data']
        
        # Extract cohort demand columns
        cohort_columns = [col for col in grid_data.columns if col.startswith('demand_mw_')]
        cohort_names = [col.replace('demand_mw_', '') for col in cohort_columns]
        
        # Calculate cohort statistics
        cohort_stats = {}
        for i, cohort_name in enumerate(cohort_names):
            cohort_col = cohort_columns[i]
            cohort_demand = grid_data[cohort_col]
            
            cohort_stats[cohort_name] = {
                'mean_demand': cohort_demand.mean(),
                'peak_demand': cohort_demand.max(),
                'min_demand': cohort_demand.min(),
                'std_demand': cohort_demand.std(),
                'total_energy': cohort_demand.sum()
            }
        
        cohort_summary[scenario_name] = cohort_stats
    
    return cohort_summary

def calculate_grid_metrics(all_scenarios):
    """Calculate grid-level performance metrics"""
    
    grid_metrics = {}
    
    for scenario_name, scenario_data in all_scenarios.items():
        grid_data = scenario_data['grid_data']
        summary = scenario_data['summary']
        
        # Temperature analysis
        temp_c = grid_data['dry_bulb_temperature_c']
        temp_f = temp_c * 9/5 + 32
        
        # Total demand analysis
        cohort_columns = [col for col in grid_data.columns if col.startswith('demand_mw_')]
        total_demand = grid_data[cohort_columns].sum(axis=1)
        
        # Time-based analysis
        grid_data_with_time = grid_data.copy()
        grid_data_with_time['hour'] = range(len(grid_data))
        grid_data_with_time['hour_of_day'] = grid_data_with_time['hour'] % 24
        
        # Calculate hourly patterns
        hourly_demand = grid_data_with_time.groupby('hour_of_day')[cohort_columns].sum().sum(axis=1)
        
        grid_metrics[scenario_name] = {
            'duration_hours': len(grid_data),
            'peak_demand_mw': total_demand.max(),
            'avg_demand_mw': total_demand.mean(),
            'min_demand_mw': total_demand.min(),
            'demand_volatility': total_demand.std(),
            'temp_range_f': (temp_f.min(), temp_f.max()),
            'temp_avg_f': temp_f.mean(),
            'grid_capacity_pct': summary.get('peak_capacity_pct', 0),
            'strain_hours': summary.get('strain_hours', 0),
            'hourly_demand_pattern': hourly_demand.to_dict()
        }
    
    return grid_metrics
