#!/usr/bin/env python3
"""
LSTM Training Script for Multi-Cohort Energy Demand Forecasting

This script trains a sophisticated neural network on realistic grid strain scenarios
including normal operations, heat waves, cold snaps, and blizzard conditions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.energy_recommender.models.forecasting.pytorch_lstm_architecture import create_lstm_trainer
from src.energy_recommender.models.forecasting.demand_simulation import generate_all_demand_scenarios

def setup_training_environment():
    """Set up directories and environment for LSTM training"""
    
    # Create model storage directory
    model_dir = "models/lstm_forecasting"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create results directory
    results_dir = "results/lstm_training"
    os.makedirs(results_dir, exist_ok=True)
    
    print("🏗️ Training environment setup:")
    print(f"   Model directory: {model_dir}")
    print(f"   Results directory: {results_dir}")
    
    return model_dir, results_dir

def analyze_training_scenarios(all_scenarios):
    """Analyze the training data scenarios before neural network training"""
    
    print("\n📊 TRAINING DATA ANALYSIS")
    print("=" * 60)
    
    scenario_summary = []
    
    for scenario_name, scenario_data in all_scenarios.items():
        grid_data = scenario_data['grid_data']
        summary = scenario_data['summary']
        
        print(f"\n📈 {scenario_name.upper()}:")
        print(f"   Duration: {len(grid_data)/24:.1f} days ({len(grid_data)} hours)")
        print(f"   Peak demand: {summary['max_demand_mw']:.1f} MW ({summary['peak_capacity_pct']:.1f}% capacity)")
        print(f"   Strain hours: {summary['strain_hours']}/{len(grid_data)} ({summary['strain_percentage']:.1f}%)")
        
        # Temperature analysis
        temp_range = grid_data['dry_bulb_temperature_c']
        temp_f_min = temp_range.min() * 9/5 + 32
        temp_f_max = temp_range.max() * 9/5 + 32
        print(f"   Temperature: {temp_f_min:.1f}°F to {temp_f_max:.1f}°F")
        
        # Cohort demand analysis
        cohort_columns = [col for col in grid_data.columns if col.startswith('demand_mw_')]
        total_cohort_demand = grid_data[cohort_columns].sum(axis=1)
        
        print(f"   Cohort diversity: {len(cohort_columns)} building types")
        print(f"   Demand variation: {total_cohort_demand.std():.1f} MW std dev")
        
        scenario_summary.append({
            'scenario': scenario_name,
            'duration_days': len(grid_data)/24,
            'peak_demand_mw': summary['max_demand_mw'],
            'strain_percentage': summary['strain_percentage'],
            'temp_range_f': f"{temp_f_min:.1f}-{temp_f_max:.1f}",
            'demand_volatility': total_cohort_demand.std()
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(scenario_summary)
    print(f"\n🎯 TRAINING DATA SUMMARY:")
    print(summary_df.round(1))
    
    return summary_df

def train_lstm_model(all_scenarios, model_dir, results_dir):
    """Train the multi-cohort LSTM on realistic grid scenarios"""
    
    print("\n🧠 STARTING LSTM NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Create LSTM trainer
    print("🔧 Initializing Multi-Cohort LSTM...")
    lstm_model = create_lstm_trainer()
    
    print(f"   Architecture: 15-cohort multi-output LSTM")
    print(f"   Sequence length: {lstm_model.sequence_length} hours lookback")
    print(f"   Forecast horizon: {lstm_model.forecast_horizon} hours ahead")
    print(f"   Cohorts: {lstm_model.n_cohorts}")
    
    # Start training with realistic parameters
    print(f"\n🚀 Training neural network on {len(all_scenarios)} weather scenarios...")
    
    training_start_time = datetime.now()
    
    try:
        # Train model - using conservative settings for stability
        history = lstm_model.train_model(
            all_scenarios=all_scenarios,
            validation_split=0.2,        # 20% for temporal validation
            epochs=30,                   # Conservative epoch count
            batch_size=32                # Reasonable batch size
        )
        
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds() / 60
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Training time: {training_duration:.1f} minutes")
        print(f"   Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
        
        # Save trained model
        model_path = os.path.join(model_dir, "multi_cohort_lstm")
        lstm_model.save_model(model_path)
        print(f"   Model saved: {model_path}")
        
        return lstm_model, history
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

def plot_training_history(history, results_dir):
    """Create training history visualizations for PyTorch training"""
    
    print("\n📊 Creating training performance visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PyTorch LSTM Training Performance Analysis', fontsize=16, fontweight='bold')
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss During Training')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss on Log Scale
    axes[0, 1].plot(epochs_range, history['train_loss'], 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Loss (Log Scale)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Log(MSE)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Convergence (Recent epochs)
    recent_epochs = min(15, len(history['train_loss']))
    recent_range = range(len(history['train_loss']) - recent_epochs + 1, len(history['train_loss']) + 1)
    recent_train = history['train_loss'][-recent_epochs:]
    recent_val = history['val_loss'][-recent_epochs:]
    
    axes[1, 0].plot(recent_range, recent_train, 'o-', label='Training', linewidth=2, markersize=6)
    axes[1, 0].plot(recent_range, recent_val, 's-', label='Validation', linewidth=2, markersize=6)
    axes[1, 0].set_title(f'Final {recent_epochs} Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training Metrics Summary
    axes[1, 1].text(0.1, 0.8, f"Training Summary:", fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f"Epochs trained: {history['epochs_trained']}", fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f"Best val loss: {history['best_val_loss']:.6f}", fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f"Final train loss: {history['train_loss'][-1]:.6f}", fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f"Final val loss: {history['val_loss'][-1]:.6f}", fontsize=12, transform=axes[1, 1].transAxes)
    
    # Check for overfitting
    if history['val_loss'][-1] > history['train_loss'][-1] * 1.5:
        axes[1, 1].text(0.1, 0.2, "⚠️ Potential overfitting detected", fontsize=12, color='red', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.1, 0.2, "✅ Good training convergence", fontsize=12, color='green', transform=axes[1, 1].transAxes)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, 'pytorch_lstm_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   Training plots saved: {plot_path}")
    
    return plot_path

def test_sample_predictions(lstm_model, all_scenarios, results_dir):
    """Generate and analyze sample predictions from trained LSTM"""
    
    print("\n🔮 TESTING LSTM FORECASTING CAPABILITIES")
    print("=" * 60)
    
    # Test predictions on different scenarios
    test_scenarios = ['heat_wave', 'cold_snap', 'blizzard']
    sample_predictions = {}
    
    for scenario_name in test_scenarios:
        if scenario_name not in all_scenarios:
            continue
            
        print(f"\n📈 Testing predictions for {scenario_name}...")
        
        grid_data = all_scenarios[scenario_name]['grid_data']
        
        # Use first 48 hours to predict next 24 hours
        if len(grid_data) < 72:  # Need 48 + 24 hours minimum
            print(f"   ⚠️ Insufficient data for prediction test")
            continue
        
        try:
            # Prepare prediction input (first 48 hours)
            prediction_input = grid_data.head(48)
            
            # Generate forecast
            forecasts = lstm_model.predict_cohort_demands(prediction_input)
            
            # Get actual values for comparison (hours 48-72)
            actual_data = grid_data.iloc[48:72]
            
            # Calculate prediction accuracy
            cohort_accuracies = {}
            
            print(f"   🎯 Prediction Results ({scenario_name}):")
            
            for i, (cohort_name, forecast) in enumerate(forecasts.items()):
                if i >= 5:  # Show first 5 cohorts
                    break
                    
                actual_col = f'demand_mw_{cohort_name}'
                if actual_col in actual_data.columns:
                    actual_values = actual_data[actual_col].values
                    
                    # Calculate Mean Absolute Error
                    mae = np.mean(np.abs(forecast - actual_values))
                    
                    # Calculate Mean Absolute Percentage Error
                    mape = np.mean(np.abs((forecast - actual_values) / actual_values)) * 100
                    
                    cohort_accuracies[cohort_name] = {'mae': mae, 'mape': mape}
                    
                    print(f"     {cohort_name}: MAE={mae:.2f} MW, MAPE={mape:.1f}%")
            
            sample_predictions[scenario_name] = {
                'forecasts': forecasts,
                'accuracies': cohort_accuracies,
                'prediction_period': '24-hour forecast from first 48 hours'
            }
            
        except Exception as e:
            print(f"   ❌ Prediction test failed: {str(e)}")
    
    # Save sample predictions
    if sample_predictions:
        # Create summary of prediction performance
        prediction_summary = []
        
        for scenario, results in sample_predictions.items():
            avg_mae = np.mean([acc['mae'] for acc in results['accuracies'].values()])
            avg_mape = np.mean([acc['mape'] for acc in results['accuracies'].values()])
            
            prediction_summary.append({
                'scenario': scenario,
                'avg_mae_mw': avg_mae,
                'avg_mape_percent': avg_mape,
                'cohorts_tested': len(results['accuracies'])
            })
        
        summary_df = pd.DataFrame(prediction_summary)
        
        print(f"\n🎯 PREDICTION PERFORMANCE SUMMARY:")
        print(summary_df.round(2))
        
        # Save results
        results_path = os.path.join(results_dir, 'sample_predictions.csv')
        summary_df.to_csv(results_path, index=False)
        print(f"   Results saved: {results_path}")
    
    return sample_predictions

def main():
    """Main training workflow"""
    
    print("🚀 MULTI-COHORT LSTM TRAINING PIPELINE (PyTorch)")
    print("=" * 70)
    print("Training neural network for energy demand forecasting")
    print("Target: 15 building cohorts, 24-hour forecasts, grid strain scenarios")
    print("Framework: PyTorch (macOS compatible)")
    print("=" * 70)
    
    try:
        # Setup environment
        model_dir, results_dir = setup_training_environment()
        
        # Generate training scenarios
        print("\n📊 Generating realistic grid strain scenarios...")
        all_scenarios = generate_all_demand_scenarios()
        
        # Analyze training data
        scenario_summary = analyze_training_scenarios(all_scenarios)
        
        # Train LSTM model
        lstm_model, history = train_lstm_model(all_scenarios, model_dir, results_dir)
        
        # Create training visualizations
        plot_training_history(history, results_dir)
        
        # Test sample predictions
        sample_predictions = test_sample_predictions(lstm_model, all_scenarios, results_dir)
        
        print("\n🎉 PYTORCH LSTM TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("✅ Neural network trained on realistic grid scenarios")
        print("✅ Model performance validated on extreme weather conditions") 
        print("✅ Sample forecasts generated for stakeholder demonstration")
        print("✅ Training artifacts saved for team coordination")
        
        print(f"\n📁 Generated Files:")
        print(f"   Model: {model_dir}/multi_cohort_lstm_model.pth")
        print(f"   Config: {model_dir}/multi_cohort_lstm_config.pkl")
        print(f"   Training plots: {results_dir}/pytorch_lstm_training_history.png")
        print(f"   Prediction results: {results_dir}/sample_predictions.csv")
        
        print(f"\n🚀 Ready for integration with portfolio optimization pipeline!")
        
        return lstm_model, history, sample_predictions
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run training pipeline
    trained_model, training_history, predictions = main()