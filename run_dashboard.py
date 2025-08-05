#!/usr/bin/env python3
"""
Main script to run the Energy Recommendation Engine Dashboard.

This script initializes and runs the Dash application with proper
configuration, error handling, and logging.
"""

import sys
import os
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dashboard.config import DashboardConfig, get_constants
from dashboard.app import app
from dashboard.data_manager import DataManager


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, DashboardConfig.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(DashboardConfig.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def validate_environment():
    """Validate the environment and configuration."""
    logger = logging.getLogger(__name__)
    
    # Validate configuration
    if not DashboardConfig.validate_config():
        logger.error("Configuration validation failed")
        return False
    
    # Check if data directory exists
    if not os.path.exists(DashboardConfig.DATA_PATH):
        logger.info(f"Creating data directory: {DashboardConfig.DATA_PATH}")
        os.makedirs(DashboardConfig.DATA_PATH, exist_ok=True)
    
    # Check if required packages are installed
    try:
        import dash
        import plotly
        import pandas
        import numpy
        logger.info("All required packages are available")
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        return False
    
    return True


def initialize_data():
    """Initialize data for the dashboard."""
    logger = logging.getLogger(__name__)
    
    try:
        data_manager = DataManager(DashboardConfig.DATA_PATH)
        
        # Generate sample data if no real data is available
        sample_data = data_manager.generate_sample_data()
        
        # Calculate metrics
        metrics = data_manager.calculate_metrics(sample_data)
        recommendations = data_manager.get_energy_recommendations(sample_data)
        
        logger.info(f"Initialized dashboard with {len(sample_data)} data points")
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize data: {e}")
        return False


def main():
    """Main function to run the dashboard."""
    print("=" * 60)
    print("Energy Recommendation Engine Dashboard")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Energy Recommendation Engine Dashboard")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)
    
    # Initialize data
    if not initialize_data():
        logger.error("Data initialization failed. Exiting.")
        sys.exit(1)
    
    # Get constants
    constants = get_constants()
    
    # Display startup information
    print(f"Dashboard Version: {constants['VERSION']}")
    print(f"Server Host: {DashboardConfig.HOST}")
    print(f"Server Port: {DashboardConfig.PORT}")
    print(f"Debug Mode: {DashboardConfig.DEBUG}")
    print(f"Data Path: {DashboardConfig.DATA_PATH}")
    print("-" * 60)
    
    try:
        # Run the Dash application
        logger.info(f"Starting dashboard server on {DashboardConfig.HOST}:{DashboardConfig.PORT}")
        app.run(
            debug=DashboardConfig.DEBUG,
            host=DashboardConfig.HOST,
            port=DashboardConfig.PORT
        )
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        print("\nDashboard stopped by user")
        
    except Exception as e:
        logger.error(f"Dashboard failed to start: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 