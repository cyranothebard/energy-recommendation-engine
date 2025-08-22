#!/usr/bin/env python3
"""
Debug script to test chart creation functions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dashboard.data_manager import DataManager
from dashboard.components import EnergyVisualizations

def debug_chart_creation():
    """Debug the chart creation process."""
    print("🔍 Debugging Chart Creation")
    print("=" * 50)
    
    # Initialize data manager
    data_manager = DataManager("data")
    
    # Load LSTM integration results
    print("\n1. Loading LSTM integration results...")
    integration_results = data_manager.load_lstm_integration_results()
    
    if integration_results is None:
        print("❌ Failed to load LSTM integration results")
        return False
    
    print(f"✅ LSTM data loaded successfully")
    
    # Test time series data transformation
    print("\n2. Testing time series data transformation...")
    scenario = "heat_wave"
    time_series_data = data_manager.get_lstm_time_series_data(integration_results, scenario)
    
    if time_series_data.empty:
        print("❌ Time series data is empty")
        return False
    
    print(f"✅ Time series data generated")
    print(f"   - Shape: {time_series_data.shape}")
    print(f"   - Columns: {list(time_series_data.columns)}")
    print(f"   - First few rows:")
    print(time_series_data.head())
    
    # Test building distribution data transformation
    print("\n3. Testing building distribution data transformation...")
    building_data = data_manager.get_lstm_building_distribution_data(integration_results, scenario)
    
    if building_data.empty:
        print("❌ Building distribution data is empty")
        return False
    
    print(f"✅ Building distribution data generated")
    print(f"   - Shape: {building_data.shape}")
    print(f"   - Columns: {list(building_data.columns)}")
    print(f"   - Data:")
    print(building_data)
    
    # Test chart creation
    print("\n4. Testing chart creation...")
    
    try:
        # Test energy consumption chart
        print("   - Creating energy consumption chart...")
        energy_chart = EnergyVisualizations.create_energy_consumption_chart(
            time_series_data, 
            f"Energy Consumption Over Time - {scenario.replace('_', ' ').title()}"
        )
        print(f"✅ Energy consumption chart created successfully")
        print(f"   - Chart type: {type(energy_chart)}")
        if hasattr(energy_chart, 'data'):
            print(f"   - Chart has {len(energy_chart.data)} traces")
        
    except Exception as e:
        print(f"❌ Energy consumption chart creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test building type chart
        print("   - Creating building type chart...")
        building_chart = EnergyVisualizations.create_building_type_chart(
            building_data, 
            f"Building Type Distribution - {scenario.replace('_', ' ').title()}"
        )
        print(f"✅ Building type chart created successfully")
        print(f"   - Chart type: {type(building_chart)}")
        if hasattr(building_chart, 'data'):
            print(f"   - Chart has {len(building_chart.data)} traces")
        
    except Exception as e:
        print(f"❌ Building type chart creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎯 Chart Creation Debug Complete!")
    return True

if __name__ == "__main__":
    success = debug_chart_creation()
    sys.exit(0 if success else 1) 