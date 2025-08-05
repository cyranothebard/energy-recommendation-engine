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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Manages data operations for the dashboard."""
    
    def __init__(self, data_path: str = None):
        """Initialize the data manager."""
        self.data_path = data_path or "data"
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def generate_sample_data(self, days: int = 365) -> pd.DataFrame:
        """Generate comprehensive sample data for demonstration."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        # Generate realistic energy consumption patterns
        base_consumption = 100
        seasonal_pattern = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 5, len(dates))
        
        energy_consumption = base_consumption + seasonal_pattern + weekly_pattern + noise
        
        # Generate temperature data with seasonal patterns
        base_temp = 20
        temp_seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        temp_noise = np.random.normal(0, 3, len(dates))
        temperature = base_temp + temp_seasonal + temp_noise
        
        # Generate building types with realistic distribution
        building_types = np.random.choice(
            ['Office', 'Residential', 'Industrial', 'Commercial'],
            size=len(dates),
            p=[0.4, 0.3, 0.2, 0.1]
        )
        
        # Generate efficiency scores with some correlation to building type
        efficiency_scores = np.random.uniform(0.3, 0.9, len(dates))
        
        # Add some correlation between efficiency and building type
        for i, building_type in enumerate(building_types):
            if building_type == 'Industrial':
                efficiency_scores[i] *= 0.8  # Industrial buildings tend to be less efficient
            elif building_type == 'Residential':
                efficiency_scores[i] *= 1.1  # Residential buildings tend to be more efficient
        
        # Generate additional features
        data = {
            'date': dates,
            'energy_consumption': energy_consumption,
            'temperature': temperature,
            'building_type': building_types,
            'efficiency_score': np.clip(efficiency_scores, 0.1, 0.95),
            'humidity': np.random.uniform(30, 80, len(dates)),
            'occupancy': np.random.uniform(0.1, 1.0, len(dates)),
            'cost_per_kwh': np.random.uniform(0.08, 0.15, len(dates)),
            'renewable_energy_percentage': np.random.uniform(0, 0.3, len(dates))
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
        
        # Cost metrics
        avg_cost_per_kwh = df['cost_per_kwh'].mean()
        metrics['total_cost'] = metrics['total_energy'] * avg_cost_per_kwh
        metrics['avg_daily_cost'] = metrics['total_cost'] / len(df)
        
        # Efficiency metrics
        metrics['avg_efficiency'] = df['efficiency_score'].mean()
        metrics['efficiency_trend'] = self._calculate_trend(df['efficiency_score'])
        
        # Environmental metrics
        metrics['total_co2'] = metrics['total_energy'] * 0.5  # kg CO2 per kWh
        metrics['renewable_percentage'] = df['renewable_energy_percentage'].mean()
        
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