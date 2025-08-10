"""
Compliance prediction module for energy recommendation system.

This module handles the prediction of building compliance with energy
reduction recommendations based on building and context features.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def create_compliance_target(features_df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create realistic compliance probability based on building and recommendation characteristics.
    
    This function uses domain expertise to model realistic compliance behavior:
    - Office buildings are more compliant than other types
    - Larger reduction requests have lower compliance
    - More advance notice improves compliance
    - Extreme temperatures reduce compliance
    
    Args:
        features_df: DataFrame with building and recommendation features
        seed: Random seed for reproducible results
        
    Returns:
        Tuple of (binary_compliance, compliance_probability)
        - binary_compliance: Binary array (0/1) of actual compliance
        - compliance_probability: Continuous probability scores [0,1]
    """
    np.random.seed(seed)
    
    # Base compliance probability (industry baseline ~65%)
    base_compliance = 0.65
    compliance_prob = np.full(len(features_df), base_compliance)
    
    # Building type effects - office buildings more compliant
    if 'is_officesmall' in features_df.columns:
        compliance_prob += features_df['is_officesmall'] * 0.15
    
    # Reduce compliance for larger reduction requests (exponential penalty)
    compliance_prob -= features_df['reduction_magnitude'] * 1.2
    
    # More advance notice = better compliance
    compliance_prob += (features_df['advance_notice_hours'] / 24) * 0.1
    
    # Extreme temperatures reduce compliance (comfort priority)
    temp_penalty = np.abs(features_df['outside_temp'] - 72) / 100
    compliance_prob -= temp_penalty
    
    # Add realistic noise and clip to valid probability range
    compliance_prob += np.random.normal(0, 0.05, len(features_df))
    compliance_prob = np.clip(compliance_prob, 0.1, 0.9)
    
    # Convert to binary outcome using probability
    binary_compliance = (np.random.random(len(features_df)) < compliance_prob).astype(int)
    
    return binary_compliance, compliance_prob


def predict_compliance_probability(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict compliance for building-recommendation pairs.
    
    This is a wrapper function that adds compliance predictions to the feature DataFrame.
    In production, this would use a trained ML model instead of rule-based logic.
    
    Args:
        features_df: DataFrame with building and recommendation features
        
    Returns:
        pd.DataFrame: Original features plus compliance predictions
    """
    features_with_compliance = features_df.copy()
    
    # Generate compliance predictions
    binary_compliance, compliance_prob = create_compliance_target(features_df)
    
    # Add to DataFrame
    features_with_compliance['binary_compliance'] = binary_compliance
    features_with_compliance['compliance_probability'] = compliance_prob
    
    # Report summary statistics
    avg_compliance = binary_compliance.mean()
    print(f"Compliance prediction summary:")
    print(f"  Average compliance rate: {avg_compliance:.1%}")
    print(f"  Compliance probability range: {compliance_prob.min():.3f} - {compliance_prob.max():.3f}")
    
    return features_with_compliance