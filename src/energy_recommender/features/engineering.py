"""
Building feature engineering module for energy recommendation system.

This module handles the extraction and engineering of building characteristics
from NREL metadata for machine learning pipeline consumption.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


def engineer_building_features_comprehensive(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Systematically extract and engineer ALL building features with proper null handling.
    
    Args:
        metadata_df: DataFrame with building metadata from NREL dataset
        
    Returns:
        pd.DataFrame: Engineered features with building_id as index
        
    Raises:
        ValueError: If required columns are missing from metadata
    """
    # Extract key columns with error handling
    try:
        building_type_col = [col for col in metadata_df.columns if 'building_type' in col.lower()][0]
        sqft_col = [col for col in metadata_df.columns if 'sqft' in col.lower()][0] 
        vintage_col = [col for col in metadata_df.columns if 'vintage' in col.lower()][0]
        hvac_col = [col for col in metadata_df.columns if 'hvac' in col.lower()][0]
    except IndexError as e:
        raise ValueError(f"Required building metadata columns not found: {e}")
    
    features = pd.DataFrame()
    features['building_id'] = metadata_df.index
    
    # Clean and fill missing values
    building_type_clean = metadata_df[building_type_col].fillna('Unknown')
    hvac_clean = metadata_df[hvac_col].fillna('Unknown') 
    vintage_clean = metadata_df[vintage_col].fillna('Unknown')
    sqft_clean = metadata_df[sqft_col].fillna('Unknown')
    
    # Original categorical features (cleaned)
    features['building_type'] = building_type_clean
    features['hvac_system'] = hvac_clean
    features['vintage'] = vintage_clean
    features['sqft_category'] = sqft_clean
    
    # Data quality reporting
    print("Missing data check:")
    print(f"Building type nulls: {metadata_df[building_type_col].isna().sum()}")
    print(f"HVAC nulls: {metadata_df[hvac_col].isna().sum()}")
    print(f"Vintage nulls: {metadata_df[vintage_col].isna().sum()}")
    
    # Systematic one-hot encoding for ALL building types
    building_types = building_type_clean.unique()
    print(f"\nCreating features for {len(building_types)} building types: {building_types}")
    
    for btype in building_types:
        if pd.notna(btype):  # Additional safety check
            clean_name = str(btype).replace(' ', '_').replace('-', '_').lower()
            features[f'is_{clean_name}'] = (building_type_clean == btype).astype(int)
    
    # Systematic encoding for HVAC systems (top 10 to avoid feature explosion)
    hvac_systems = hvac_clean.value_counts().head(10).index
    print(f"Creating features for top {len(hvac_systems)} HVAC systems")
    
    for hvac in hvac_systems:
        if pd.notna(hvac):
            clean_name = str(hvac).replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').lower()[:20]
            features[f'hvac_{clean_name}'] = (hvac_clean == hvac).astype(int)
    
    return features


def engineer_compliance_features(building_features_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Engineer features for compliance prediction modeling.
    
    Args:
        building_features_df: DataFrame with building characteristics
        seed: Random seed for reproducible synthetic data
        
    Returns:
        pd.DataFrame: Features augmented with recommendation context
    """
    # Start with building features
    compliance_features = building_features_df.copy()
    
    # Add synthetic recommendation context for testing
    np.random.seed(seed)
    n_buildings = len(building_features_df)
    
    # Recommendation characteristics (would come from recommendation system in production)
    compliance_features['reduction_magnitude'] = np.random.uniform(0.05, 0.30, n_buildings)  # 5-30% reduction
    compliance_features['advance_notice_hours'] = np.random.choice([2, 4, 8, 24], n_buildings)
    compliance_features['duration_hours'] = np.random.choice([1, 2, 4], n_buildings)
    compliance_features['time_of_day'] = np.random.choice([10, 14, 16, 18], n_buildings)  # Peak hours
    compliance_features['outside_temp'] = np.random.normal(75, 15, n_buildings)  # Temperature impact
    
    return compliance_features