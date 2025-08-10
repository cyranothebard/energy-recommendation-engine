#!/usr/bin/env python3
"""
Test script for LSTM integration dashboard functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dashboard.data_manager import DataManager
from dashboard.components import EnergyVisualizations

def test_lstm_data_loading():
    """Test LSTM data loading functionality."""
    print("Testing LSTM data loading...")
    
    data_manager = DataManager("data")
    integration_results = data_manager.load_lstm_integration_results()
    
    if integration_results is None:
        print("❌ Failed to load LSTM integration results")
        return False
    
    print(f"✅ Successfully loaded LSTM results with {len(integration_results)} weather scenarios")
    
    # Test summary metrics
    summary_metrics = data_manager.get_lstm_summary_metrics(integration_results)
    print(f"✅ Generated summary metrics for {len(summary_metrics)} scenarios")
    
    # Test performance metrics
    performance_metrics = data_manager.get_lstm_performance_metrics(integration_results)
    print(f"✅ Generated performance metrics with {len(performance_metrics)} categories")
    
    # Test recommendations
    recommendations = data_manager.get_lstm_recommendations(integration_results)
    print(f"✅ Generated {len(recommendations)} recommendations")
    
    # Test cohort analysis
    cohort_analysis = data_manager.get_lstm_cohort_analysis(integration_results, "heat_wave")
    print(f"✅ Generated cohort analysis for {len(cohort_analysis)} cohorts")
    
    return True

def test_visualization_creation():
    """Test visualization creation functionality."""
    print("\nTesting visualization creation...")
    
    data_manager = DataManager("data")
    integration_results = data_manager.load_lstm_integration_results()
    
    if integration_results is None:
        print("❌ Cannot test visualizations without data")
        return False
    
    try:
        # Test various chart types
        charts = [
            ("LSTM Forecast", EnergyVisualizations.create_lstm_forecast_chart(integration_results, "heat_wave")),
            ("Weather Comparison", EnergyVisualizations.create_weather_scenario_comparison(integration_results)),
            ("Building Heatmap", EnergyVisualizations.create_building_cohort_heatmap(integration_results, "heat_wave")),
            ("Strain Summary", EnergyVisualizations.create_strain_prediction_summary(integration_results)),
            ("Peak Analysis", EnergyVisualizations.create_peak_hour_analysis(integration_results)),
            ("Performance Validation", EnergyVisualizations.create_performance_validation_chart({})),
            ("Cohort Performance", EnergyVisualizations.create_cohort_performance_heatmap(integration_results, "heat_wave")),
            ("Grid Strain Timeline", EnergyVisualizations.create_grid_strain_timeline(integration_results)),
            ("Weather Summary", EnergyVisualizations.create_weather_scenario_summary(integration_results))
        ]
        
        for chart_name, chart in charts:
            if chart is not None and hasattr(chart, 'data'):
                print(f"✅ {chart_name} chart created successfully")
            else:
                print(f"❌ {chart_name} chart creation failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return False

def main():
    """Run all tests."""
    print("🧠 LSTM Integration Dashboard Test Suite")
    print("=" * 50)
    
    # Test data loading
    data_loading_success = test_lstm_data_loading()
    
    # Test visualization creation
    visualization_success = test_visualization_creation()
    
    print("\n" + "=" * 50)
    if data_loading_success and visualization_success:
        print("🎉 All tests passed! Dashboard is ready to run.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 