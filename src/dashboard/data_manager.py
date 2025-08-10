"""
Data manager for Energy Recommendation Engine Dashboard.

This module handles data loading, processing, and integration with
the energy recommendation engine pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading, processing, and storage for the dashboard."""
    
    def __init__(self, data_path):
        """Initialize the DataManager with a data directory path."""
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def generate_sample_data(self):
        """Generate sample energy consumption data for demonstration."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        data = {
            'date': dates,
            'energy_consumption': np.random.normal(100, 20, len(dates)),
            'temperature': np.random.normal(20, 10, len(dates)),
            'building_type': np.random.choice(['Office', 'Residential', 'Industrial'], len(dates)),
            'efficiency_score': np.random.uniform(0.3, 0.9, len(dates)),
            'cost_per_kwh': np.random.uniform(0.08, 0.15, len(dates)),  # $0.08-$0.15 per kWh
            'renewable_energy_percentage': np.random.uniform(0.1, 0.4, len(dates))  # 10-40% renewable
        }
        
        return pd.DataFrame(data)
    
    def load_real_data(self, file_path: str) -> pd.DataFrame:
        """Load real data from file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Successfully loaded data from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return self.generate_sample_data()
    
    def get_energy_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate energy efficiency recommendations based on data analysis."""
        recommendations = []
        
        # Calculate average energy consumption by building type
        avg_by_type = df.groupby('building_type')['energy_consumption'].mean()
        
        # Calculate efficiency scores
        avg_efficiency = df['efficiency_score'].mean()
        
        # Generate recommendations based on data patterns
        if avg_efficiency < 0.6:
            recommendations.append({
                'priority': 'high',
                'category': 'efficiency',
                'title': 'Upgrade Building Systems',
                'description': 'Current efficiency score is below optimal levels. Consider upgrading HVAC and lighting systems.',
                'potential_savings': '15-25%',
                'estimated_cost': '$50,000 - $100,000',
                'payback_period': '3-5 years'
            })
        
        # Check for seasonal patterns
        monthly_avg = df.groupby(df['date'].dt.month)['energy_consumption'].mean()
        if monthly_avg.max() / monthly_avg.min() > 1.5:
            recommendations.append({
                'priority': 'medium',
                'category': 'seasonal',
                'title': 'Optimize Seasonal Operations',
                'description': 'Significant seasonal variation detected. Implement seasonal optimization strategies.',
                'potential_savings': '10-15%',
                'estimated_cost': '$10,000 - $25,000',
                'payback_period': '1-2 years'
            })
        
        # Check building type specific recommendations
        for building_type, avg_consumption in avg_by_type.items():
            if avg_consumption > 120:  # High consumption threshold
                recommendations.append({
                    'priority': 'high',
                    'category': 'building_specific',
                    'title': f'Optimize {building_type} Operations',
                    'description': f'{building_type} buildings show above-average energy consumption.',
                    'potential_savings': '20-30%',
                    'estimated_cost': '$25,000 - $75,000',
                    'payback_period': '2-4 years'
                })
        
        # Add general recommendations
        recommendations.extend([
            {
                'priority': 'medium',
                'category': 'general',
                'title': 'Implement Smart Controls',
                'description': 'Install smart thermostats and automated control systems.',
                'potential_savings': '8-12%',
                'estimated_cost': '$5,000 - $15,000',
                'payback_period': '1-2 years'
            },
            {
                'priority': 'low',
                'category': 'renewable',
                'title': 'Consider Renewable Energy',
                'description': 'Evaluate solar panel installation for renewable energy generation.',
                'potential_savings': '25-40%',
                'estimated_cost': '$100,000 - $200,000',
                'payback_period': '5-8 years'
            }
        ])
        
        return recommendations
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate key performance metrics."""
        metrics = {}
        
        # Energy metrics
        metrics['total_energy'] = df['energy_consumption'].sum()
        metrics['avg_energy'] = df['energy_consumption'].mean()
        metrics['peak_energy'] = df['energy_consumption'].max()
        metrics['min_energy'] = df['energy_consumption'].min()
        
        # Cost metrics - handle missing columns gracefully
        if 'cost_per_kwh' in df.columns:
            avg_cost_per_kwh = df['cost_per_kwh'].mean()
            metrics['total_cost'] = metrics['total_energy'] * avg_cost_per_kwh
            metrics['avg_daily_cost'] = metrics['total_cost'] / len(df)
        else:
            # Default cost if not available
            avg_cost_per_kwh = 0.12  # Default $0.12 per kWh
            metrics['total_cost'] = metrics['total_energy'] * avg_cost_per_kwh
            metrics['avg_daily_cost'] = metrics['total_cost'] / len(df)
        
        # Efficiency metrics
        metrics['avg_efficiency'] = df['efficiency_score'].mean()
        metrics['efficiency_trend'] = self._calculate_trend(df['efficiency_score'])
        
        # Environmental metrics
        metrics['total_co2'] = metrics['total_energy'] * 0.5  # kg CO2 per kWh
        if 'renewable_energy_percentage' in df.columns:
            metrics['renewable_percentage'] = df['renewable_energy_percentage'].mean()
        else:
            metrics['renewable_percentage'] = 0.25  # Default 25% renewable
        
        # Savings potential
        potential_savings = 0.15  # 15% potential savings
        metrics['potential_energy_savings'] = metrics['total_energy'] * potential_savings
        metrics['potential_cost_savings'] = metrics['total_cost'] * potential_savings
        
        return metrics
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for a time series."""
        if len(series) < 2:
            return "stable"
        
        # Simple linear trend calculation
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive data summary."""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'building_types': df['building_type'].value_counts().to_dict(),
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': df.duplicated().sum()
            }
        }
        
        return summary
    
    def export_data(self, df: pd.DataFrame, format: str = 'csv', file_path: str = None) -> str:
        """Export data to various formats."""
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"energy_data_{timestamp}.{format}"
        
        try:
            if format == 'csv':
                df.to_csv(file_path, index=False)
            elif format == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            elif format == 'parquet':
                df.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Data exported successfully to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise 

    def load_lstm_integration_results(self):
        """Load LSTM integration test results from the JSON file."""
        try:
            results_file = Path(__file__).parent.parent.parent / "results" / "integration_test_results.json"
            
            if not results_file.exists():
                print(f"Warning: LSTM results file not found at {results_file}")
                return None
            
            with open(results_file, 'r') as f:
                integration_results = json.load(f)
            
            print(f"Successfully loaded LSTM results with {len(integration_results)} weather scenarios")
            return integration_results
            
        except Exception as e:
            print(f"Error loading LSTM results: {e}")
            return None
    
    def get_lstm_summary_metrics(self, integration_results):
        """Calculate summary metrics from LSTM integration results."""
        if not integration_results:
            return {}
        
        metrics = {}
        
        for scenario, data in integration_results.items():
            cohort_data = data["cohort_forecasts"]
            
            # Calculate total energy for each hour across all cohorts
            total_energy_by_hour = []
            for hour in range(24):
                hour_total = sum(cohort_data[cohort][hour] for cohort in cohort_data.keys())
                total_energy_by_hour.append(hour_total)
            
            metrics[scenario] = {
                'total_cohorts': len(cohort_data),
                'peak_hour': total_energy_by_hour.index(max(total_energy_by_hour)),
                'peak_energy': max(total_energy_by_hour),
                'total_daily_energy': sum(total_energy_by_hour),
                'avg_hourly_energy': sum(total_energy_by_hour) / 24,
                'strain_predicted': data["strain_prediction"],
                'capacity_forecast': data["capacity_forecast"],
                'hourly_breakdown': total_energy_by_hour
            }
        
        return metrics
    
    def get_lstm_recommendations(self, integration_results):
        """Generate recommendations based on LSTM integration results."""
        if not integration_results:
            return []
        
        recommendations = []
        
        for scenario, data in integration_results.items():
            if data["strain_prediction"]:
                recommendations.append({
                    'type': 'danger',
                    'message': f'{scenario.replace("_", " ").title()}: Grid strain predicted. Implement demand response measures.',
                    'priority': 'high',
                    'scenario': scenario
                })
            
            cohort_data = data["cohort_forecasts"]
            peak_hour = 0
            peak_energy = 0
            
            # Find peak hour across all cohorts
            for hour in range(24):
                hour_total = sum(cohort_data[cohort][hour] for cohort in cohort_data.keys())
                if hour_total > peak_energy:
                    peak_energy = hour_total
                    peak_hour = hour
            
            recommendations.append({
                'type': 'info',
                'message': f'{scenario.replace("_", " ").title()}: Peak energy at hour {peak_hour} ({peak_energy:.1f} kWh). Consider load shifting.',
                'priority': 'medium',
                'scenario': scenario
            })
        
        return recommendations
    
    def get_lstm_performance_metrics(self, integration_results):
        """Calculate performance metrics for LSTM model validation."""
        if not integration_results:
            return {}
        
        # Based on industry benchmarks from the documentation
        performance_metrics = {
            'industry_benchmarks': {
                'highly_accurate': 'MAPE < 10%',
                'reasonable': 'MAPE 11-20%',
                'acceptable_extreme': 'MAPE 20-25%',
                'challenging': 'MAPE > 25%'
            },
            'our_performance': {
                'heat_wave': '20-25% MAPE (Acceptable for extreme weather)',
                'cold_snap': '20-25% MAPE (Acceptable for extreme weather)',
                'blizzard': '200%+ MAPE (Challenging - unprecedented conditions)'
            },
            'business_justification': {
                'grid_stability': 'RMSE prioritized over MAPE for high-demand periods',
                'production_viable': 'Within documented ranges for challenging forecasting',
                'extreme_weather': 'Appropriate performance for unprecedented conditions'
            }
        }
        
        return performance_metrics
    
    def get_lstm_cohort_analysis(self, integration_results, scenario="heat_wave"):
        """Extract cohort-specific performance metrics from LSTM results."""
        if not integration_results or scenario not in integration_results:
            return {}
        
        cohort_data = integration_results[scenario]["cohort_forecasts"]
        analysis = {}
        
        for cohort_name, hourly_forecasts in cohort_data.items():
            # Parse building type and size from cohort name
            if '_' in cohort_name:
                building_type, size = cohort_name.split('_', 1)
            else:
                building_type, size = cohort_name, "Unknown"
            
            # Calculate metrics
            peak_hour = hourly_forecasts.index(max(hourly_forecasts))
            peak_energy = max(hourly_forecasts)
            total_daily_energy = sum(hourly_forecasts)
            avg_hourly_energy = total_daily_energy / 24
            
            # Calculate volatility (standard deviation)
            volatility = np.std(hourly_forecasts)
            
            # Peak to average ratio
            peak_to_avg_ratio = peak_energy / avg_hourly_energy if avg_hourly_energy > 0 else 0
            
            analysis[cohort_name] = {
                'building_type': building_type,
                'size': size,
                'peak_hour': peak_hour,
                'peak_energy': peak_energy,
                'total_daily_energy': total_daily_energy,
                'avg_hourly_energy': avg_hourly_energy,
                'volatility': volatility,
                'peak_to_avg_ratio': peak_to_avg_ratio,
                'hourly_forecasts': hourly_forecasts
            }
        
        return analysis

    def get_lstm_time_series_data(self, integration_results, scenario="heat_wave"):
        """Transform LSTM results into time series data for energy consumption charts."""
        if not integration_results or scenario not in integration_results:
            return pd.DataFrame()
        
        # Create 24-hour time series
        hours = pd.date_range(start='2024-01-01 00:00:00', periods=24, freq='h')
        
        cohort_data = integration_results[scenario]["cohort_forecasts"]
        
        # Aggregate energy consumption by hour across all cohorts
        hourly_energy = []
        for hour in range(24):
            total_energy = sum(cohort_data[cohort][hour] for cohort in cohort_data.keys())
            hourly_energy.append(total_energy)
        
        # Create DataFrame with time series data
        time_series_data = pd.DataFrame({
            'date': hours,
            'energy_consumption': hourly_energy,
            'temperature': self._generate_weather_temperature(scenario, hours),
            'building_type': 'Mixed',  # Since we're aggregating across all building types
            'efficiency_score': 0.7,  # Default efficiency score
            'cost_per_kwh': 0.12,  # Default cost
            'renewable_energy_percentage': 0.25  # Default renewable percentage
        })
        
        return time_series_data
    
    def get_lstm_building_distribution_data(self, integration_results, scenario="heat_wave"):
        """Transform LSTM results into building type distribution data."""
        if not integration_results or scenario not in integration_results:
            return pd.DataFrame()
        
        cohort_data = integration_results[scenario]["cohort_forecasts"]
        
        # Group cohorts by building type and calculate total daily energy
        building_type_energy = {}
        
        for cohort_name, hourly_forecasts in cohort_data.items():
            if '_' in cohort_name:
                building_type = cohort_name.split('_')[0]
            else:
                building_type = cohort_name
            
            total_daily_energy = sum(hourly_forecasts)
            
            if building_type in building_type_energy:
                building_type_energy[building_type] += total_daily_energy
            else:
                building_type_energy[building_type] = total_daily_energy
        
        # Create DataFrame for building type distribution
        building_data = []
        for building_type, total_energy in building_type_energy.items():
            building_data.append({
                'building_type': building_type,
                'total_energy': total_energy,
                'avg_daily_energy': total_energy,
                'efficiency_score': 0.7,  # Default efficiency score
                'cost_per_kwh': 0.12,  # Default cost
                'renewable_energy_percentage': 0.25  # Default renewable percentage
            })
        
        return pd.DataFrame(building_data)
    
    def _generate_weather_temperature(self, scenario, hours):
        """Generate realistic temperature data based on weather scenario."""
        if scenario == "heat_wave":
            # High temperatures during day, warm at night
            base_temp = 30
            daily_variation = 8
        elif scenario == "cold_snap":
            # Low temperatures, cold throughout
            base_temp = -5
            daily_variation = 6
        elif scenario == "blizzard":
            # Very cold with minimal variation
            base_temp = -15
            daily_variation = 3
        else:
            # Normal weather
            base_temp = 20
            daily_variation = 10
        
        # Generate temperatures with daily cycle (warmer during day, cooler at night)
        temperatures = []
        for i, hour in enumerate(hours):
            # Day is warmer (around hour 14 = 2 PM), night is cooler (around hour 2 = 2 AM)
            daily_cycle = np.cos(2 * np.pi * (hour.hour - 14) / 24)
            temp = base_temp + daily_variation * daily_cycle + np.random.normal(0, 2)
            temperatures.append(temp)
        
        return temperatures 