"""
Configuration settings for Energy Recommendation Engine Dashboard.

This module manages all configuration settings, environment variables,
and application constants.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DashboardConfig:
    """Configuration class for the dashboard application."""
    
    # Server Configuration
    HOST = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    PORT = int(os.getenv('DASHBOARD_PORT', 8051))
    DEBUG = os.getenv('DASHBOARD_DEBUG', 'True').lower() == 'true'
    
    # Data Configuration
    DATA_PATH = os.getenv('DATA_PATH', 'data')
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 300))  # 5 minutes
    
    # Database Configuration (if using database)
    DATABASE_URL = os.getenv('DATABASE_URL', None)
    
    # API Configuration
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))
    
    # Security Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'dashboard.log')
    
    # Dashboard Theme and Styling
    THEME = os.getenv('DASHBOARD_THEME', 'bootstrap')
    PRIMARY_COLOR = os.getenv('PRIMARY_COLOR', '#007bff')
    SECONDARY_COLOR = os.getenv('SECONDARY_COLOR', '#6c757d')
    
    # Chart Configuration
    CHART_HEIGHT = int(os.getenv('CHART_HEIGHT', 400))
    CHART_TEMPLATE = os.getenv('CHART_TEMPLATE', 'plotly_white')
    
    # Data Refresh Configuration
    AUTO_REFRESH_INTERVAL = int(os.getenv('AUTO_REFRESH_INTERVAL', 300000))  # 5 minutes in milliseconds
    
    # Feature Flags
    ENABLE_REAL_TIME_UPDATES = os.getenv('ENABLE_REAL_TIME_UPDATES', 'False').lower() == 'true'
    ENABLE_DATA_EXPORT = os.getenv('ENABLE_DATA_EXPORT', 'True').lower() == 'true'
    ENABLE_FILTERS = os.getenv('ENABLE_FILTERS', 'True').lower() == 'true'
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration."""
        if cls.DATABASE_URL:
            return {'url': cls.DATABASE_URL}
        return {}
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration for caching."""
        return {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'db': int(os.getenv('REDIS_DB', 0)),
            'password': os.getenv('REDIS_PASSWORD', None)
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Check required directories
        if not os.path.exists(cls.DATA_PATH):
            try:
                os.makedirs(cls.DATA_PATH, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create data directory: {e}")
        
        # Check port range
        if not (1024 <= cls.PORT <= 65535):
            errors.append(f"Port {cls.PORT} is not in valid range (1024-65535)")
        
        # Check timeout values
        if cls.CACHE_TIMEOUT <= 0:
            errors.append("Cache timeout must be positive")
        
        if cls.API_TIMEOUT <= 0:
            errors.append("API timeout must be positive")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Dashboard-specific constants
DASHBOARD_CONSTANTS = {
    'APP_TITLE': 'Energy Recommendation Engine Dashboard',
    'APP_DESCRIPTION': 'Interactive dashboard for energy efficiency analysis and recommendations',
    'VERSION': '1.0.0',
    'AUTHOR': 'Energy Recommendation Engine Team',
    
    # Chart colors
    'COLORS': {
        'primary': '#007bff',
        'secondary': '#6c757d',
        'success': '#28a745',
        'danger': '#dc3545',
        'warning': '#ffc107',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    },
    
    # Energy efficiency thresholds
    'EFFICIENCY_THRESHOLDS': {
        'excellent': 0.8,
        'good': 0.6,
        'fair': 0.4,
        'poor': 0.2
    },
    
    # Energy consumption thresholds (kWh)
    'CONSUMPTION_THRESHOLDS': {
        'low': 50,
        'medium': 100,
        'high': 150,
        'very_high': 200
    },
    
    # Cost per kWh (default values)
    'DEFAULT_COST_PER_KWH': 0.12,
    
    # CO2 emission factor (kg CO2 per kWh)
    'CO2_EMISSION_FACTOR': 0.5,
    
    # Potential savings percentages
    'POTENTIAL_SAVINGS': {
        'conservative': 0.10,  # 10%
        'moderate': 0.15,      # 15%
        'aggressive': 0.25     # 25%
    }
}


def get_config() -> DashboardConfig:
    """Get dashboard configuration instance."""
    return DashboardConfig


def get_constants() -> Dict[str, Any]:
    """Get dashboard constants."""
    return DASHBOARD_CONSTANTS 