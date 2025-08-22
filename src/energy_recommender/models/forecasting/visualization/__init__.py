"""
Energy Recommendation Project - Visualization Module

This module provides comprehensive visualization capabilities for the energy
recommendation system, including model performance analysis, business impact
assessment, and stakeholder reporting.
"""

from .report_generator import generate_comprehensive_report
from .plot_templates import (
    create_peak_demand_heatmap,
    create_cohort_consumption_patterns,
    create_prediction_accuracy_analysis,
    create_cost_savings_analysis,
    create_decision_impact_timeline
)

__version__ = "1.0.0"
__author__ = "Energy Recommendation Team"

__all__ = [
    'generate_comprehensive_report',
    'create_peak_demand_heatmap',
    'create_cohort_consumption_patterns', 
    'create_prediction_accuracy_analysis',
    'create_cost_savings_analysis',
    'create_decision_impact_timeline'
]
