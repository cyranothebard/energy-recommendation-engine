#!/usr/bin/env python3
"""
Generate Comprehensive Visualizations for Energy Recommendation System

This script creates executive-ready visualizations for stakeholder reports.
Run this after training your LSTM model to generate publication-quality charts.

Usage:
    python generate_report_visualizations.py
    python generate_report_visualizations.py --format png,pdf
    python generate_report_visualizations.py --output-dir custom/path
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.energy_recommender.models.forecasting.visualization.report_generator import (
    setup_report_environment,
    generate_comprehensive_report
)

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Generate comprehensive energy recommendation visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_report_visualizations.py
    python generate_report_visualizations.py --format png,pdf --dpi 600
    python generate_report_visualizations.py --output-dir reports/executive_summary
        """
    )
    
    parser.add_argument('--output-dir', 
                       default=None,
                       help='Output directory for reports (default: src/energy_recommender/models/forecasting/results/reports)')
    
    parser.add_argument('--format', 
                       default='png',
                       help='Output formats: png, pdf, svg (comma-separated)')
    
    parser.add_argument('--dpi', 
                       type=int, 
                       default=300,
                       help='Image resolution (default: 300)')
    
    args = parser.parse_args()
    
    print("📊 ENERGY RECOMMENDATION SYSTEM - VISUALIZATION GENERATOR")
    print("=" * 65)
    print("Creating executive-ready visualizations and analysis reports")
    print("=" * 65)
    
    # Setup environment
    if args.output_dir is None:
        # Default to results/reports within the project
        default_output = os.path.join(project_root, "src", "energy_recommender", 
                                     "models", "forecasting", "results", "reports")
        output_dirs = setup_report_environment(default_output)
    else:
        output_dirs = setup_report_environment(args.output_dir)
    
    # Parse formats
    include_formats = [fmt.strip() for fmt in args.format.split(',')]
    
    print(f"\n🎯 Configuration:")
    print(f"   Output directory: {output_dirs['output_dir']}")
    print(f"   Image formats: {', '.join(include_formats)}")
    print(f"   Resolution: {args.dpi} DPI")
    
    # Generate comprehensive report
    success = generate_comprehensive_report(
        output_dirs=output_dirs,
        include_formats=include_formats,
        dpi=args.dpi
    )
    
    if success:
        print(f"\n🎉 SUCCESS! All visualizations generated successfully!")
        print(f"📁 Files saved to: {output_dirs['output_dir']}")
        print(f"\n📋 Generated Reports:")
        print(f"   • Peak Demand Heatmap - Grid strain risk by hour/weather")
        print(f"   • Cohort Consumption Patterns - Building type energy profiles") 
        print(f"   • Prediction Accuracy Analysis - LSTM model performance")
        print(f"   • Cost Savings Analysis - Business impact quantification")
        print(f"   • Decision Impact Timeline - 6-hour advance warning system")
        print(f"   • Executive Summary - Key findings and recommendations")
        
        print(f"\n🚀 Ready for stakeholder presentation!")
        return 0
    else:
        print(f"\n❌ Report generation failed!")
        print(f"   Please ensure LSTM model is trained first by running:")
        print(f"   python train_lstm.py")
        return 1

if __name__ == "__main__":
    exit(main())
