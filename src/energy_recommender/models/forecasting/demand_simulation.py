import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CohortDemandSimulator:
    """Generate realistic energy demand patterns for building cohorts"""
    
    def __init__(self):
        # Define cohort-specific energy characteristics
        self.cohort_profiles = {
            # Office Buildings - Peak weekday business hours
            'SmallOffice_Small': {
                'base_load_kw': 25, 'peak_multiplier': 2.5,
                'temp_sensitivity': 0.8, 'humidity_factor': 0.3,
                'schedule': 'office', 'weekend_reduction': 0.3
            },
            'SmallOffice_Medium-Small': {
                'base_load_kw': 45, 'peak_multiplier': 2.3,
                'temp_sensitivity': 0.9, 'humidity_factor': 0.3,
                'schedule': 'office', 'weekend_reduction': 0.3
            },
            'SmallOffice_Medium-Large': {
                'base_load_kw': 85, 'peak_multiplier': 2.1,
                'temp_sensitivity': 1.0, 'humidity_factor': 0.4,
                'schedule': 'office', 'weekend_reduction': 0.3
            },
            
            # Retail - Peak during shopping hours, higher weekend usage
            'RetailStandalone_Small': {
                'base_load_kw': 35, 'peak_multiplier': 1.8,
                'temp_sensitivity': 1.2, 'humidity_factor': 0.5,
                'schedule': 'retail', 'weekend_reduction': 0.8
            },
            'RetailStandalone_Medium-Small': {
                'base_load_kw': 65, 'peak_multiplier': 1.9,
                'temp_sensitivity': 1.3, 'humidity_factor': 0.5,
                'schedule': 'retail', 'weekend_reduction': 0.8
            },
            'RetailStandalone_Medium-Large': {
                'base_load_kw': 120, 'peak_multiplier': 1.7,
                'temp_sensitivity': 1.4, 'humidity_factor': 0.6,
                'schedule': 'retail', 'weekend_reduction': 0.8
            },
            
            # Restaurants - Peak meal times, consistent 7-day operation
            'FullServiceRestaurant_Small': {
                'base_load_kw': 40, 'peak_multiplier': 2.8,
                'temp_sensitivity': 0.6, 'humidity_factor': 0.4,
                'schedule': 'restaurant', 'weekend_reduction': 1.1
            },
            'FullServiceRestaurant_Medium-Small': {
                'base_load_kw': 75, 'peak_multiplier': 2.6,
                'temp_sensitivity': 0.7, 'humidity_factor': 0.4,
                'schedule': 'restaurant', 'weekend_reduction': 1.1
            },
            
            # Strip Malls - Mixed retail with extended hours
            'RetailStripmall_Small': {
                'base_load_kw': 30, 'peak_multiplier': 1.6,
                'temp_sensitivity': 1.1, 'humidity_factor': 0.5,
                'schedule': 'retail_extended', 'weekend_reduction': 0.9
            },
            'RetailStripmall_Medium-Small': {
                'base_load_kw': 55, 'peak_multiplier': 1.7,
                'temp_sensitivity': 1.2, 'humidity_factor': 0.5,
                'schedule': 'retail_extended', 'weekend_reduction': 0.9
            },
            'RetailStripmall_Medium-Large': {
                'base_load_kw': 95, 'peak_multiplier': 1.5,
                'temp_sensitivity': 1.3, 'humidity_factor': 0.6,
                'schedule': 'retail_extended', 'weekend_reduction': 0.9
            },
            
            # Warehouses - Consistent operation, less weather sensitive
            'Warehouse_Medium-Small': {
                'base_load_kw': 50, 'peak_multiplier': 1.3,
                'temp_sensitivity': 0.4, 'humidity_factor': 0.2,
                'schedule': 'industrial', 'weekend_reduction': 0.7
            },
            'Warehouse_Medium-Large': {
                'base_load_kw': 150, 'peak_multiplier': 1.2,
                'temp_sensitivity': 0.5, 'humidity_factor': 0.2,
                'schedule': 'industrial', 'weekend_reduction': 0.7
            },
            
            # Hotels - 24/7 operation with guest comfort priority
            'LargeHotel_Large': {
                'base_load_kw': 250, 'peak_multiplier': 1.4,
                'temp_sensitivity': 1.8, 'humidity_factor': 0.8,
                'schedule': 'hotel', 'weekend_reduction': 1.0
            },
            'SmallHotel_Medium-Small': {
                'base_load_kw': 80, 'peak_multiplier': 1.5,
                'temp_sensitivity': 1.6, 'humidity_factor': 0.7,
                'schedule': 'hotel', 'weekend_reduction': 1.0
            }
        }
        
        # Operating schedule patterns
        self.schedules = {
            'office': {
                'weekday_pattern': [0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 0.9, 0.7, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3],
                'weekend_pattern': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
            },
            'retail': {
                'weekday_pattern': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2],
                'weekend_pattern': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.3, 0.2]
            },
            'restaurant': {
                'weekday_pattern': [0.3, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8, 0.7, 0.6, 0.7, 1.0, 1.0, 0.8, 0.6, 0.5, 0.6, 0.9, 1.0, 0.9, 0.7, 0.6, 0.5, 0.4],
                'weekend_pattern': [0.3, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.7, 0.6, 0.7, 1.0, 1.0, 1.0, 0.8, 0.7, 0.6, 0.4]
            },
            'hotel': {
                'weekday_pattern': [0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.6],
                'weekend_pattern': [0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.7, 0.7]
            },
            'industrial': {
                'weekday_pattern': [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8],
                'weekend_pattern': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6]
            },
            'retail_extended': {
                'weekday_pattern': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2],
                'weekend_pattern': [0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2]
            }
        }
    
    def calculate_weather_impact(self, weather_data, cohort_profile):
        """Calculate weather-driven energy demand adjustments"""
        
        # Temperature impact (cooling and heating loads)
        temp_c = weather_data['dry_bulb_temperature_c']
        temp_impact = np.where(
            temp_c > 22,  # Above 72°F = cooling load
            (temp_c - 22) * cohort_profile['temp_sensitivity'] * 0.05,  # 5% per degree
            np.where(
                temp_c < 10,  # Below 50°F = heating load  
                (10 - temp_c) * cohort_profile['temp_sensitivity'] * 0.03,  # 3% per degree
                0  # Comfortable range
            )
        )
        
        # Humidity impact (additional HVAC load)
        humidity_pct = weather_data['relative_humidity_pct']
        humidity_impact = np.where(
            humidity_pct > 70,
            (humidity_pct - 70) * cohort_profile['humidity_factor'] * 0.002,  # 0.2% per humidity %
            0
        )
        
        # Solar radiation impact (reduced lighting needs, increased cooling)
        radiation = weather_data['global_horizontal_radiation_wm2']
        solar_impact = (radiation / 1000) * 0.02  # 2% reduction per 1000 W/m² (lighting offset)
        
        total_weather_impact = 1 + temp_impact + humidity_impact - solar_impact
        return np.clip(total_weather_impact, 0.5, 2.0)  # Reasonable bounds
    
    def generate_cohort_demand(self, weather_data, cohort_name):
        """Generate realistic hourly demand for a specific building cohort"""
        
        profile = self.cohort_profiles[cohort_name]
        schedule_type = profile['schedule']
        
        # Base demand pattern
        hours = len(weather_data)
        base_demand = np.full(hours, profile['base_load_kw'])
        
        # Apply operating schedule
        demand_multipliers = []
        for _, row in weather_data.iterrows():
            is_weekend = row['date_time'].dayofweek >= 5
            hour = row['date_time'].hour
            
            if is_weekend:
                schedule_mult = self.schedules[schedule_type]['weekend_pattern'][hour]
                schedule_mult *= profile['weekend_reduction']
            else:
                schedule_mult = self.schedules[schedule_type]['weekday_pattern'][hour]
            
            demand_multipliers.append(schedule_mult * profile['peak_multiplier'])
        
        scheduled_demand = base_demand * np.array(demand_multipliers)
        
        # Apply weather impacts
        weather_multipliers = self.calculate_weather_impact(weather_data, profile)
        final_demand = scheduled_demand * weather_multipliers
        
        # Add realistic noise
        noise = np.random.normal(1.0, 0.05, hours)  # 5% random variation
        final_demand *= noise
        
        return final_demand.round(1)
    
# Add to demand_simulation.py

class GridDemandSimulator(CohortDemandSimulator):
    """Generate aggregate grid-level demand using actual building counts"""
    
    def __init__(self):
        super().__init__()
        
        # Your actual cohort counts from analysis
        self.cohort_building_counts = {
            'SmallOffice_Small': 867,
            'SmallOffice_Medium-Small': 845, 
            'RetailStandalone_Medium-Small': 614,
            'RetailStandalone_Small': 533,
            'FullServiceRestaurant_Small': 514,
            'SmallOffice_Medium-Large': 510,
            'RetailStandalone_Medium-Large': 502,
            'RetailStripmall_Medium-Small': 359,
            'Warehouse_Medium-Large': 351,
            'RetailStripmall_Medium-Large': 316,
            'FullServiceRestaurant_Medium-Small': 293,
            'Warehouse_Medium-Small': 272,
            'LargeHotel_Large': 247,
            'RetailStripmall_Small': 239,
            'SmallHotel_Medium-Small': 206
        }
        
        # Total buildings in analysis: 6,668 buildings
        self.total_buildings = sum(self.cohort_building_counts.values())
        
        # Regional grid capacity calibrated to building portfolio scope
        self.regional_capacity_mw = 1700  # Calibrated to building portfolio scope
        
        # Extreme load factors for different weather scenarios
        self.extreme_load_factors = {
            'heat_wave': 1.4,       # 40% spike from simultaneous AC usage
            'cold_snap': 1.35,      # 35% spike from emergency heating  
            'blizzard': 1.25,       # 25% spike from heating + potential outages
            'normal_summer': 1.0,   # Normal conditions
            'normal_winter': 1.0,   # Normal conditions
            'shoulder_season': 1.0  # Normal conditions
        }
    
    def generate_grid_demand_scenario(self, weather_data, scenario_name):
        """Generate complete grid demand with all cohorts for a weather scenario"""
        
        print(f"🏢 Generating aggregate demand for {scenario_name} scenario...")
        print(f"Processing {self.total_buildings:,} buildings across {len(self.cohort_building_counts)} cohorts")
        
        # Initialize results DataFrame
        grid_data = weather_data.copy()
        cohort_demands = {}
        
        # Generate demand for each cohort
        for cohort_name, building_count in self.cohort_building_counts.items():
            print(f"  📊 {cohort_name}: {building_count} buildings")
            
            # Per-building demand
            per_building_demand = self.generate_cohort_demand(weather_data, cohort_name)
            
            # Aggregate to cohort level
            cohort_aggregate = per_building_demand * building_count
            cohort_demands[cohort_name] = cohort_aggregate
            
            # Add to grid data
            grid_data[f'demand_mw_{cohort_name}'] = (cohort_aggregate / 1000).round(2)  # Convert kW to MW
        
        # Calculate total grid demand
        total_demand_mw = sum(cohort_demands.values()) / 1000  # Convert to MW
        
        # Apply extreme load factor based on scenario type
        scenario_key = scenario_name.split()[0].lower()  # Extract first word and lowercase
        load_factor = self.extreme_load_factors.get(scenario_key, 1.0)
        total_demand_mw *= load_factor
        
        grid_data['total_grid_demand_mw'] = total_demand_mw.round(1)
        
        # Calculate grid strain metrics using regional capacity
        grid_data['grid_capacity_pct'] = (grid_data['total_grid_demand_mw'] / self.regional_capacity_mw * 100).round(1)
        grid_data['strain_alert'] = grid_data['grid_capacity_pct'] >= 85  # Strain threshold
        
        # Add scenario metadata
        grid_data['weather_scenario'] = scenario_name
        
        return grid_data, cohort_demands

# Integration with existing weather scenarios
def generate_all_demand_scenarios():
    """Generate complete grid demand for all weather scenarios"""
    
    from ...data.synthetic_weather import scenarios  # Your existing weather data
    
    grid_simulator = GridDemandSimulator()
    all_scenarios = {}
    
    scenario_mapping = {
        'normal_summer': 'Normal Summer Conditions',
        'heat_wave': 'Extreme Heat Wave (99°F+)', 
        'normal_winter': 'Normal Winter Conditions',
        'cold_snap': 'Extreme Cold Snap (10°F)',
        'blizzard': 'Major Blizzard Event',
        'shoulder_season': 'Spring Transition Period'
    }
    
    for scenario_key, weather_data in scenarios.items():
        scenario_name = scenario_mapping[scenario_key]
        
        print(f"\n🌡️ Processing {scenario_name}...")
        grid_data, cohort_demands = grid_simulator.generate_grid_demand_scenario(
            weather_data, scenario_name
        )
        
        # Generate summary statistics
        max_demand = grid_data['total_grid_demand_mw'].max()
        avg_demand = grid_data['total_grid_demand_mw'].mean()
        peak_capacity_pct = grid_data['grid_capacity_pct'].max()
        strain_hours = grid_data['strain_alert'].sum()
        
        print(f"  ⚡ Peak demand: {max_demand:.1f} MW ({peak_capacity_pct:.1f}% of capacity)")
        print(f"  📊 Average demand: {avg_demand:.1f} MW")
        print(f"  🚨 Grid strain hours: {strain_hours}/{len(grid_data)} ({strain_hours/len(grid_data)*100:.1f}%)")
        
        all_scenarios[scenario_key] = {
            'grid_data': grid_data,
            'cohort_demands': cohort_demands,
            'summary': {
                'max_demand_mw': max_demand,
                'avg_demand_mw': avg_demand,
                'peak_capacity_pct': peak_capacity_pct,
                'strain_hours': strain_hours,
                'strain_percentage': strain_hours/len(grid_data)*100
            }
        }
    
    return all_scenarios

# LSTM Training Data Preparation
def prepare_lstm_training_data(all_scenarios, sequence_length=48, forecast_horizon=24):
    """Prepare time series data for multi-output LSTM training"""
    
    # Initialize grid simulator to get cohort names
    grid_simulator = GridDemandSimulator()
    training_sequences = []
    
    for scenario_name, scenario_data in all_scenarios.items():
        grid_data = scenario_data['grid_data']
        
        # Features for LSTM input
        feature_columns = [
            # Weather features
            'dry_bulb_temperature_c', 'relative_humidity_pct', 
            'wind_speed_ms', 'global_horizontal_radiation_wm2',
            # Temporal features  
            'hour_of_day', 'day_of_week', 'month',
            # Historical demand (lagged features)
        ] + [col for col in grid_data.columns if col.startswith('demand_mw_')]
        
        # Create sequences for training
        for i in range(sequence_length, len(grid_data) - forecast_horizon):
            # Input sequence (48 hours of history)
            input_sequence = grid_data.iloc[i-sequence_length:i][feature_columns].values
            
            # Output targets (next 24 hours for each cohort)
            output_targets = []
            for cohort in grid_simulator.cohort_building_counts.keys():
                cohort_target = grid_data.iloc[i:i+forecast_horizon][f'demand_mw_{cohort}'].values
                output_targets.append(cohort_target)
            
            training_sequences.append({
                'input_sequence': input_sequence,
                'output_targets': np.array(output_targets),  # Shape: (15 cohorts, 24 hours)
                'scenario': scenario_name
            })
    
    return training_sequences


if __name__ == "__main__":
    """Main execution block for testing demand simulation"""
    
    print("🚀 Starting Energy Demand Simulation...")
    
    try:
        # Import your existing synthetic weather data
        from ...data.synthetic_weather import scenarios
        
        # Test basic cohort simulation first
        print("\n📊 Testing single cohort simulation...")
        simulator = CohortDemandSimulator()
        
        # Use your heat wave scenario for testing (most interesting case)
        test_scenario_name = 'heat_wave'
        weather_df = scenarios[test_scenario_name].copy()
        
        print(f"   Using {test_scenario_name} scenario from your synthetic weather data")
        print(f"   Weather data period: {weather_df['date_time'].min()} to {weather_df['date_time'].max()}")
        print(f"   Temperature range: {weather_df['dry_bulb_temperature_c'].min():.1f}°C to {weather_df['dry_bulb_temperature_c'].max():.1f}°C")
        
        # Test single cohort
        test_cohort = 'SmallOffice_Small'
        demand = simulator.generate_cohort_demand(weather_df, test_cohort)
        print(f"✅ Generated demand for {test_cohort}: {demand[:5]}... (showing first 5 hours)")
        print(f"   Peak demand: {demand.max():.1f} kW, Average: {demand.mean():.1f} kW")
        
        # Test grid-level simulation
        print(f"\n🏢 Testing grid-level simulation with {test_scenario_name} scenario...")
        grid_sim = GridDemandSimulator()
        
        grid_data, cohort_demands = grid_sim.generate_grid_demand_scenario(weather_df, test_scenario_name.replace('_', ' ').title())
        
        print(f"✅ Grid simulation complete!")
        print(f"   Total grid demand range: {grid_data['total_grid_demand_mw'].min():.1f} - {grid_data['total_grid_demand_mw'].max():.1f} MW")
        print(f"   Peak capacity usage: {grid_data['grid_capacity_pct'].max():.1f}%")
        print(f"   Strain alert hours: {grid_data['strain_alert'].sum()}/{len(grid_data)}")
        
        # Test with all available scenarios
        print(f"\n🌡️ Testing all available weather scenarios...")
        available_scenarios = list(scenarios.keys())
        print(f"   Available scenarios: {', '.join(available_scenarios)}")
        
        scenario_results = {}
        for scenario_key in available_scenarios:  # Test all scenarios
            print(f"\n   Processing {scenario_key}...")
            scenario_weather = scenarios[scenario_key]
            scenario_grid_data, _ = grid_sim.generate_grid_demand_scenario(
                scenario_weather, 
                scenario_key.replace('_', ' ').title()
            )
            
            peak_demand = scenario_grid_data['total_grid_demand_mw'].max()
            peak_capacity = scenario_grid_data['grid_capacity_pct'].max()
            strain_hours = scenario_grid_data['strain_alert'].sum()
            
            scenario_results[scenario_key] = {
                'peak_demand_mw': peak_demand,
                'peak_capacity_pct': peak_capacity,
                'strain_hours': strain_hours
            }
            
            print(f"     Peak: {peak_demand:.1f} MW ({peak_capacity:.1f}% capacity), Strain: {strain_hours} hours")
        
        # Show comparison of scenarios
        print(f"\n📊 Scenario Comparison Summary:")
        for scenario, results in scenario_results.items():
            print(f"   {scenario:15} | {results['peak_demand_mw']:6.1f} MW | {results['peak_capacity_pct']:5.1f}% | {results['strain_hours']:2d} strain hours")
        
        # Show sample of results
        print(f"\n📋 Sample Results from {test_scenario_name} (first 3 hours):")
        display_cols = ['date_time', 'total_grid_demand_mw', 'grid_capacity_pct', 'strain_alert']
        if all(col in grid_data.columns for col in display_cols):
            print(grid_data[display_cols].head(3))
        
        print(f"\n🎯 All tests completed successfully!")
        print(f"   ✅ Using your synthetic weather data from src/energy_recommender/data/synthetic_weather.py")
        print(f"   ✅ {len(available_scenarios)} weather scenarios available for full analysis")
        print(f"   ✅ Ready to run full scenario generation with: generate_all_demand_scenarios()")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure synthetic_weather.py is available at src/energy_recommender/data/synthetic_weather.py")
        print("   You may need to run this from the project root directory")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()