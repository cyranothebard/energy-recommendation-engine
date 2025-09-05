
"""
Portfolio optimization module for energy recommendation system.

This module handles the selection and coordination of buildings for
grid-level energy reduction recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def optimize_building_portfolio(compliance_features_df: pd.DataFrame, 
                              target_reduction_pct: float = 0.15,
                              selection_pct: float = 0.3,
                              seed: int = 42) -> pd.DataFrame:
    """
    Optimize building selection for maximum grid impact with realistic constraints.
    
    This function implements a greedy optimization strategy that:
    1. Simulates realistic building consumption baselines
    2. Calculates absolute impact potential per building
    3. Selects top performers for recommendation targeting
    4. Reports grid-level performance metrics
    
    Args:
        compliance_features_df: DataFrame with building features and compliance predictions
        target_reduction_pct: Grid reduction target (default 15%)
        selection_pct: Percentage of buildings to select for recommendations
        seed: Random seed for reproducible building size simulation
        
    Returns:
        pd.DataFrame: Selected buildings with impact calculations
    """
    np.random.seed(seed)
    features_df = compliance_features_df.copy()
    
    # Simulate realistic building sizes (kW baseline consumption)
    # Log-normal distribution reflects real-world building size variation
    features_df['baseline_consumption_kw'] = np.random.lognormal(4, 1, len(features_df))
    
    # Calculate ABSOLUTE impact per building (kW reduction)
    features_df['absolute_impact_kw'] = (
        features_df['reduction_magnitude'] * 
        features_df['baseline_consumption_kw'] *
        features_df['compliance_probability']
    )
    
    # Grid-level baseline metrics
    total_baseline_consumption = features_df['baseline_consumption_kw'].sum()
    target_reduction_kw = total_baseline_consumption * target_reduction_pct
    
    # Calculate impact efficiency for selection prioritization
    features_df['impact_per_request'] = features_df['absolute_impact_kw']
    
    # Portfolio selection: top buildings by absolute impact
    n_selected = int(len(features_df) * selection_pct)
    selected_buildings = features_df.nlargest(n_selected, 'impact_per_request')
    
    # Calculate portfolio performance metrics
    total_expected_reduction_kw = selected_buildings['absolute_impact_kw'].sum()
    portfolio_reduction_percentage = total_expected_reduction_kw / total_baseline_consumption
    goal_achievement = (portfolio_reduction_percentage / target_reduction_pct) * 100
    
    # Performance reporting
    print("Portfolio Optimization Results:")
    print(f"Total grid baseline: {total_baseline_consumption:,.0f} kW")
    print(f"Target reduction needed: {target_reduction_kw:,.0f} kW ({target_reduction_pct:.1%})")
    print(f"Expected portfolio reduction: {total_expected_reduction_kw:,.0f} kW ({portfolio_reduction_percentage:.1%})")
    print(f"Goal achievement: {goal_achievement:.1f}%")
    print(f"Buildings selected: {len(selected_buildings):,}/{len(features_df):,}")
    print(f"Average compliance of selected: {selected_buildings['compliance_probability'].mean():.3f}")
    
    return selected_buildings


def calculate_grid_impact_metrics(portfolio_results: pd.DataFrame, 
                                total_buildings: int) -> Dict[str, float]:
    """
    Calculate comprehensive grid impact metrics for portfolio performance evaluation.
    
    Args:
        portfolio_results: DataFrame with selected buildings and impact calculations
        total_buildings: Total number of buildings in the grid
        
    Returns:
        Dict with performance metrics for reporting and evaluation
    """
    total_baseline = portfolio_results['baseline_consumption_kw'].sum() / portfolio_results['absolute_impact_kw'].sum() * 100
    
    metrics = {
        'total_reduction_kw': portfolio_results['absolute_impact_kw'].sum(),
        'average_building_impact': portfolio_results['absolute_impact_kw'].mean(),
        'selection_efficiency': len(portfolio_results) / total_buildings,
        'compliance_rate': portfolio_results['compliance_probability'].mean(),
        'load_diversity': portfolio_results['baseline_consumption_kw'].std() / portfolio_results['baseline_consumption_kw'].mean()
    }
    
    return metrics


def run_portfolio_scenarios(compliance_features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Run multiple portfolio optimization scenarios for comparison.
    
    Args:
        compliance_features_df: DataFrame with building features and compliance predictions
        
    Returns:
        Dict with results for conservative, standard, and emergency scenarios
    """
    scenarios = {
        'conservative': {'selection_pct': 0.2, 'target_reduction_pct': 0.05},
        'standard': {'selection_pct': 0.3, 'target_reduction_pct': 0.10},
        'emergency': {'selection_pct': 0.5, 'target_reduction_pct': 0.15}
    }
    
    results = {}
    for scenario_name, params in scenarios.items():
        print(f"\n=== {scenario_name.upper()} SCENARIO ===")
        results[scenario_name] = optimize_building_portfolio(
            compliance_features_df, 
            **params
        )
    
    return results