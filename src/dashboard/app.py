"""
Main Dash application for Energy Recommendation Engine Dashboard.

This module provides the main dashboard interface with interactive
visualizations for energy efficiency analysis and recommendations.
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import dashboard components and data manager
from src.dashboard.components import EnergyVisualizations
from src.dashboard.data_manager import DataManager

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    title="Energy Recommendation Engine Dashboard",
    suppress_callback_exceptions=True
)

# Add custom CSS for enhanced visual design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom CSS for enhanced visual design */
            .bg-gradient-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            }
            
            .shadow-lg {
                box-shadow: 0 1rem 3rem rgba(0,0,0,.175) !important;
            }
            
            .opacity-75 {
                opacity: 0.75 !important;
            }
            
            .opacity-25 {
                opacity: 0.25 !important;
            }
            
            .display-4 {
                font-size: 2.5rem;
                font-weight: 300;
                line-height: 1.2;
            }
            
            .fs-5 {
                font-size: 1.25rem !important;
            }
            
            .me-3 {
                margin-right: 1rem !important;
            }
            
            .me-2 {
                margin-right: 0.5rem !important;
            }
            
            .my-3 {
                margin-top: 1rem !important;
                margin-bottom: 1rem !important;
            }
            
            .mt-2 {
                margin-top: 0.5rem !important;
            }
            
            .py-5 {
                padding-top: 3rem !important;
                padding-bottom: 3rem !important;
            }
            
            .mb-3 {
                margin-bottom: 1rem !important;
            }
            
            .badge {
                font-size: 0.75rem;
                padding: 0.5rem 0.75rem;
            }
            
            .shadow-sm {
                box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,.075) !important;
            }
            
            .text-primary {
                color: #0d6efd !important;
            }
            
            .fw-bold {
                font-weight: 700 !important;
            }
            
            .mb-1 {
                margin-bottom: 0.25rem !important;
            }
            
            .mb-2 {
                margin-bottom: 0.5rem !important;
            }
            
            .mb-3 {
                margin-bottom: 1rem !important;
            }
            
            .mb-4 {
                margin-bottom: 1.5rem !important;
            }
            
            .mb-md-0 {
                margin-bottom: 0 !important;
            }
            
            .p-3 {
                padding: 1rem !important;
            }
            
            .h-100 {
                height: 100% !important;
            }
            
            .d-flex {
                display: flex !important;
            }
            
            .flex-column {
                flex-direction: column !important;
            }
            
            .justify-content-center {
                justify-content: center !important;
            }
            
            .fa-2x {
                font-size: 2em;
            }
            
            .fs-6 {
                font-size: 1rem !important;
            }
            
            .border-start {
                border-left: 1px solid !important;
            }
            
            .border-primary {
                border-color: #0d6efd !important;
            }
            
            .border-4 {
                border-width: 0.25rem !important;
            }
            
            .ps-3 {
                padding-left: 1rem !important;
            }
            
            .text-success {
                color: #198754 !important;
            }
            
            .text-info {
                color: #0dcaf0 !important;
            }
            
            .text-muted {
                color: #6c757d !important;
            }
            
            .w-100 {
                width: 100% !important;
            }
            
            /* Responsive improvements */
            @media (max-width: 768px) {
                .display-4 {
                    font-size: 2rem;
                }
                .fs-5 {
                    font-size: 1rem !important;
                }
                .py-5 {
                    padding-top: 2rem !important;
                    padding-bottom: 2rem !important;
                }
                .mb-3 {
                    margin-bottom: 1rem !important;
                }
                .mb-md-0 {
                    margin-bottom: 1rem !important;
                }
            }
            
            /* Enhanced card hover effects */
            .card {
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 0.5rem 1rem rgba(0,0,0,.15) !important;
            }
            
            /* Better spacing for mobile */
            @media (max-width: 576px) {
                .container-fluid {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                .py-4 {
                    padding-top: 2rem !important;
                    padding-bottom: 2rem !important;
                }
            }
            
            /* CSS Grid Layout Implementation */
            .grid-row {
                display: grid !important;
                grid-template-columns: repeat(3, 1fr) !important;
                gap: 1.5rem !important;
                margin-bottom: 2rem !important;
            }
            
            .grid-row .card {
                width: 100% !important;
                max-width: none !important;
                margin: 0 !important;
                height: fit-content !important;
            }
            
            .grid-row .card .card-body {
                height: auto !important;
                display: flex !important;
                flex-direction: column !important;
            }
            
            .grid-row .card .card-body .dcc-graph {
                flex: 1 !important;
                min-height: 300px !important;
            }
            
            /* Full-width rows */
            .full-width-row {
                margin-bottom: 2rem !important;
            }
            
            .full-width-row .card {
                width: 100% !important;
                max-width: none !important;
            }
            
            /* Responsive breakpoints */
            @media (max-width: 1200px) {
                .grid-row {
                    grid-template-columns: repeat(2, 1fr) !important;
                    gap: 1rem !important;
                }
            }
            
            @media (max-width: 768px) {
                .grid-row {
                    grid-template-columns: 1fr !important;
                    gap: 1rem !important;
                }
            }
            
            /* Card styling improvements */
            .card {
                border: none !important;
                box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,.075) !important;
                transition: all 0.3s ease !important;
                border-radius: 0.5rem !important;
            }
            
            .card:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 0.5rem 1rem rgba(0,0,0,.15) !important;
            }
            
            /* Remove Bootstrap's default row/col behavior for grid rows */
            .grid-row .row {
                display: block !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            
            .grid-row .col-md-4 {
                display: block !important;
                width: 100% !important;
                max-width: none !important;
                flex: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# LSTM data will be loaded directly from integration results

# Layout
app.layout = dbc.Container([
    # Enhanced Header with better visual hierarchy
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.H1([
                                html.I(className="fas fa-bolt me-3"),
                                "Energy Recommendation Engine"
                            ], className="mb-3 fw-bold display-4"),
                            html.P([
                                html.I(className="fas fa-robot me-2"),
                                "AI-Powered Grid Stability & Energy Optimization"
                            ], className="mb-0 fs-5 opacity-75"),
                            html.Hr(className="my-3 opacity-25"),
                            html.Div([
                                html.Span("Dashboard v1.0", className="badge bg-light text-dark me-2"),
                                html.Span("Real-time Analytics", className="badge bg-success me-2"),
                                html.Span("LSTM Forecasting", className="badge bg-info")
                            ], className="mt-2")
                        ], className="text-center")
                    ], className="py-5")
                ])
            ], className="bg-gradient-primary text-white border-0 shadow-lg")
        ])
    ], className="mb-4"),
    
    # Load LSTM data on demand via a button (removed auto-interval loader)
    
    # Enhanced Key Metrics Section with better visual design
    dbc.Row([
        dbc.Col([
            html.H3([
                html.I(className="fas fa-chart-line me-2"),
                "Key Performance Metrics"
            ], className="mb-4 fw-bold text-primary")
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-leaf fa-2x text-success mb-2"),
                                    html.H4("Total Energy Saved", className="text-muted fw-bold mb-2"),
                                    html.H2("1,234 kWh", id="energy-saved", className="text-success fw-bold mb-1"),
                                    html.Small("+12% from last month", className="text-success")
                                ], className="text-center p-3")
                            ], className="h-100 d-flex flex-column justify-content-center")
                        ], width={"size": 12, "md": 4}, className="mb-3 mb-md-0"),
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-dollar-sign fa-2x text-info mb-2"),
                                    html.H4("Cost Reduction", className="text-muted fw-bold mb-2"),
                                    html.H2("$567", id="cost-reduction", className="text-info fw-bold mb-1"),
                                    html.Small("+8% from last month", className="text-info")
                                ], className="text-center p-3")
                            ], className="h-100 d-flex flex-column justify-content-center")
                        ], width={"size": 12, "md": 4}, className="mb-3 mb-md-0"),
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-tachometer-alt fa-2x text-warning mb-2"),
                                    html.H4("Efficiency Score", className="text-muted fw-bold mb-2"),
                                    html.H2("78%", id="efficiency-score", className="text-warning fw-bold mb-1"),
                                    html.Small("+5% from last month", className="text-warning")
                                ], className="text-center p-3")
                            ], className="h-100 d-flex flex-column justify-content-center")
                        ], width={"size": 12, "md": 4})
                    ])
                ])
            ], className="bg-white border-0 shadow-sm")
        ])
    ], className="mb-5"),
    
    # Enhanced LSTM Integration Results Section Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3([
                    html.I(className="fas fa-brain me-2"),
                    "LSTM Integration Results"
                ], className="mb-3 fw-bold text-primary"),
                html.P([
                    html.I(className="fas fa-chart-area me-2"),
                    "AI-powered energy demand forecasting across different weather scenarios"
                ], className="text-muted mb-4 fs-6")
            ], className="border-start border-primary border-4 ps-3")
        ])
    ], className="mb-4"),
    
    # Prominent Load button for LSTM data
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Load LSTM Data",
                id="load-lstm-btn",
                color="success",
                className="mb-3"
            )
        ], width="auto")
    ], className="mb-2"),


    

    
    # Enhanced LSTM Controls Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4([
                        html.I(className="fas fa-sliders-h me-2"),
                        "LSTM Analysis Controls"
                    ], className="mb-4 fw-bold text-info"),
                    dbc.Row([
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-cloud-sun me-2"),
                                "Weather Scenario:"
                            ], className="form-label fw-bold text-muted"),
                            dcc.Dropdown(
                                id="scenario-dropdown",
                                options=[
                                    {"label": "🌡️ Heat Wave", "value": "heat_wave"},
                                    {"label": "❄️ Cold Snap", "value": "cold_snap"},
                                    {"label": "🌨️ Blizzard", "value": "blizzard"}
                                ],
                                value="heat_wave",
                                clearable=False,
                                className="mb-3"
                            )
                        ], width={"size": 12, "md": 4}, className="mb-3 mb-md-0"),
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Chart Type:"
                            ], className="form-label fw-bold text-muted"),
                            dcc.Dropdown(
                                id="chart-type-dropdown",
                                options=[
                                    {"label": "📊 24-Hour Forecast", "value": "forecast"},
                                    {"label": "🔥 Building Cohort Heatmap", "value": "heatmap"},
                                    {"label": "⏰ Peak Hour Analysis", "value": "peak"},
                                    {"label": "📈 Cohort Performance Analysis", "value": "cohort_performance"},
                                    {"label": "⚡ Grid Strain Timeline", "value": "strain_timeline"}
                                ],
                                value="forecast",
                                clearable=False,
                                className="mb-3"
                            )
                        ], width={"size": 12, "md": 4}, className="mb-3 mb-md-0"),
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-play me-2"),
                                "Actions:"
                            ], className="form-label fw-bold text-muted"),
                            dbc.Button([
                                html.I(className="fas fa-chart-line me-2"),
                                "Show Comparison"
                            ], 
                                id="compare-scenarios-btn",
                                color="primary",
                                outline=True,
                                className="w-100"
                            )
                        ], width={"size": 12, "md": 4})
                    ])
                ])
            ], className="bg-white border-0 shadow-sm")
        ])
    ], className="mb-5"),
    
    # Responsive Grid Layout for All Cards (3x3 at desktop, full rows for specified charts)
    
    # Row 1: LSTM Energy Forecast (Full Width)
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-chart-line me-2"),
                    "LSTM Energy Forecast"
                ], className="mb-3 fw-bold text-primary"),
                html.P("AI-powered 24-hour energy demand prediction", className="text-muted mb-3"),
                dcc.Graph(id="lstm-forecast-chart", style={"height": "500px", "width": "100%"}, config={"responsive": True})
            ])
        ], className="bg-white border-0 shadow-sm")
    ], className="full-width-row"),
    
    # Row 2: CSS Grid - Performance Validation, Grid Strain, LSTM Summary
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-chart-line me-2"),
                    "LSTM Performance Validation"
                ], className="mb-3 fw-bold text-success"),
                html.P([
                    html.I(className="fas fa-award me-2"),
                    "Model performance against industry benchmarks for commercial building energy forecasting"
                ], className="text-muted mb-3"),
                dcc.Graph(id="performance-validation-chart", style={"height": "400px"})
            ])
        ], className="bg-white border-0 shadow-sm"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-bolt me-2"),
                    "Grid Strain Summary"
                ], className="mb-3 fw-bold text-warning"),
                html.P("Real-time grid stress indicators", className="text-muted mb-3"),
                dcc.Graph(id="strain-summary-chart", style={"height": "400px", "width": "100%"}, config={"responsive": True})
            ])
        ], className="bg-white border-0 shadow-sm"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-clipboard-list me-2"),
                    "LSTM Integration Summary"
                ], className="mb-3 fw-bold text-info"),
                html.P("Key metrics and insights from LSTM analysis", className="text-muted mb-3"),
                html.Div(id="lstm-summary-content", className="p-3")
            ])
        ], className="bg-white border-0 shadow-sm")
    ], className="grid-row"),
    
    # Row 3: Building Cohort Heatmap (Full Width)
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-building me-2"),
                    "Building Cohort Heatmap"
                ], className="mb-3 fw-bold text-success"),
                html.P("Energy consumption patterns across building types and time periods", className="text-muted mb-3"),
                dcc.Graph(id="lstm-building-heatmap", style={"height": "450px", "width": "100%"}, config={"responsive": True})
            ])
        ], className="bg-white border-0 shadow-sm")
    ], className="full-width-row"),
    
    # Row 4: CSS Grid - Weather Comparison, Weather Summary, Building Distribution
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-cloud-sun me-2"),
                    "Weather Scenario Comparison"
                ], className="mb-3 fw-bold text-info"),
                html.P("Energy demand patterns across different weather conditions", className="text-muted mb-3"),
                dcc.Graph(id="scenario-comparison-chart", style={"height": "400px"})
            ])
        ], className="bg-white border-0 shadow-sm"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-thermometer-half me-2"),
                    "Weather Scenario Summary"
                ], className="mb-3 fw-bold text-warning"),
                html.P("Aggregated weather impact analysis", className="text-muted mb-3"),
                dcc.Graph(id="weather-scenario-summary-chart", style={"height": "400px", "width": "100%"}, config={"responsive": True})
            ])
        ], className="bg-white border-0 shadow-sm"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-building me-2"),
                    "Building Type Distribution"
                ], className="mb-3 fw-bold text-success"),
                html.P("Energy consumption by building category", className="text-muted mb-3"),
                dcc.Graph(id="building-type-chart", style={"height": "400px", "width": "100%"}, config={"responsive": True})
            ])
        ], className="bg-white border-0 shadow-sm")
    ], className="grid-row"),
    
    # Row 5: Energy Consumption Over Time (Full Width)
    html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4([
                    html.I(className="fas fa-bolt me-2"),
                    "Energy Consumption Over Time"
                ], className="mb-3 fw-bold text-primary"),
                html.P("Historical energy usage trends and patterns", className="text-muted mb-3"),
                dcc.Graph(id="energy-consumption-chart", style={"height": "400px"})
            ])
        ], className="bg-white border-0 shadow-sm")
    ], className="full-width-row"),
    

    
], fluid=True, className="py-4")

# Callback to update energy consumption chart
@app.callback(
    Output("energy-consumption-chart", "figure"),
    [Input("load-lstm-btn", "n_clicks"),
     Input("scenario-dropdown", "value")]
)
def update_energy_consumption_chart(n_clicks, scenario):
    """Update energy consumption chart based on LSTM data and scenario."""
    try:
        if not n_clicks or not scenario:
            return go.Figure().add_annotation(text="Please load LSTM data and select a scenario", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Load data directly
        data_manager = DataManager("data")
        lstm_data = data_manager.load_lstm_integration_results()
        
        if not lstm_data or scenario not in lstm_data:
            return go.Figure().add_annotation(text="No LSTM data available for selected scenario", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Get LSTM time series data for the selected scenario
        time_series_data = data_manager.get_lstm_time_series_data(lstm_data, scenario)
        
        if time_series_data is not None and not time_series_data.empty:
            # Create chart with LSTM data
            return EnergyVisualizations.create_energy_consumption_chart(
                time_series_data, f"Energy Consumption - {scenario.replace('_', ' ').title()}"
            )
        
        # Fallback to sample data if no LSTM data
        sample_data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=24, freq='H'),
            'energy_consumption': np.random.normal(800, 200, 24)
        })
        return EnergyVisualizations.create_energy_consumption_chart(sample_data, "Energy Consumption Over Time")
        
    except Exception as e:
        print(f"Error updating energy consumption chart: {e}")
        # Return empty chart on error
        return go.Figure().add_annotation(text="Chart loading...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback to update building type chart
@app.callback(
    Output("building-type-chart", "figure"),
    [Input("load-lstm-btn", "n_clicks"),
     Input("scenario-dropdown", "value")]
)
def update_building_type_chart(n_clicks, scenario):
    """Update building type chart based on LSTM data and scenario."""
    try:
        if not n_clicks or not scenario:
            return go.Figure().add_annotation(text="Please load LSTM data and select a scenario", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Load data directly
        data_manager = DataManager("data")
        lstm_data = data_manager.load_lstm_integration_results()
        
        if not lstm_data or scenario not in lstm_data:
            return go.Figure().add_annotation(text="No LSTM data available for selected scenario", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Get LSTM building distribution data for the selected scenario
        building_data = data_manager.get_lstm_building_distribution_data(lstm_data, scenario)
        
        if building_data is not None and not building_data.empty:
            # Create chart with LSTM data
            return EnergyVisualizations.create_building_type_chart(
                building_data, f"Building Distribution - {scenario.replace('_', ' ').title()}"
            )
        
        # Fallback to sample data if no LSTM data
        sample_data = pd.DataFrame({
            'building_type': ['Office', 'Energy', 'Residential', 'Industrial', 'Hospitality'],
            'energy_consumption': np.random.normal(500, 150, 5)
        })
        return EnergyVisualizations.create_building_type_chart(sample_data, "Building Type Distribution")
        
    except Exception as e:
        print(f"Error updating building type chart: {e}")
        # Return empty chart on error
        return go.Figure().add_annotation(text="Chart loading...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# LSTM data loading is now handled directly in the update_all_lstm_components callback

# LSTM Integration Results Callbacks
@app.callback(
    Output('lstm-forecast-chart', 'figure'),
    Output('lstm-building-heatmap', 'figure'),
    Output('lstm-summary-content', 'children'),
    Input('load-lstm-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_all_lstm_components(n_clicks):
    """Update all LSTM components when the load button is clicked"""
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Load data directly
        data_manager = DataManager('data')
        lstm_data = data_manager.load_lstm_integration_results()
        
        if not lstm_data:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Get default scenario (first available)
        default_scenario = list(lstm_data.keys())[0] if lstm_data else None
        
        # Create all components
        forecast_chart = EnergyVisualizations.create_lstm_forecast_chart(lstm_data, default_scenario)
        heatmap = EnergyVisualizations.create_building_cohort_heatmap(lstm_data, default_scenario)
        summary = create_lstm_summary_content(lstm_data)
        
        return forecast_chart, heatmap, summary
        
    except Exception as e:
        print(f"Error updating LSTM components: {e}")
        return dash.no_update, dash.no_update, dash.no_update

def create_lstm_metrics(lstm_data):
    """Create LSTM metrics display"""
    if not lstm_data:
        return html.Div("No LSTM data available", className="text-muted")
    
    total_scenarios = len(lstm_data)
    strain_count = 0
    total_cohorts = 0
    peak_energy = 0
    
    for scenario_name, scenario_data in lstm_data.items():
        # Check for strain_prediction (boolean) or strain_predicted
        strain_predicted = scenario_data.get('strain_prediction', False) or scenario_data.get('strain_predicted', False)
        strain_count += 1 if strain_predicted else 0
        
        # Count cohorts from cohort_forecasts
        cohort_forecasts = scenario_data.get('cohort_forecasts', {})
        if isinstance(cohort_forecasts, dict):
            total_cohorts += len(cohort_forecasts)
        
        # Get capacity forecast as peak energy
        capacity = scenario_data.get('capacity_forecast', 0)
        if isinstance(capacity, (int, float)):
            peak_energy = max(peak_energy, capacity)
    
    return html.Div([
        html.H5("LSTM Integration Metrics", className="mb-3"),
        html.Div([
            html.Div([
                html.H4(f"{total_scenarios}", className="text-primary mb-1"),
                html.P("Weather Scenarios", className="text-muted mb-0")
            ], className="text-center p-3 border rounded"),
            html.Div([
                html.H4(f"{total_cohorts}", className="text-success mb-1"),
                html.P("Total Cohorts", className="text-muted mb-0")
            ], className="text-center p-3 border rounded"),
            html.Div([
                html.H4(f"{strain_count}", className="text-warning mb-1"),
                html.P("Strain Events", className="text-muted mb-0")
            ], className="text-center p-3 border rounded"),
            html.Div([
                html.H4(f"{peak_energy:.1f}", className="text-danger mb-1"),
                html.P("Peak Energy (kWh)", className="text-muted mb-0")
            ], className="text-center p-3 border rounded")
        ], className="row g-3")
    ])

def create_lstm_summary_content(lstm_data):
    """Create LSTM summary content"""
    if not lstm_data:
        return html.Div("No LSTM data available", className="text-muted")
    
    summary_items = []
    
    for scenario_name, scenario_data in lstm_data.items():
        cohort_count = len(scenario_data.get('cohort_forecasts', {}))
        capacity = scenario_data.get('capacity_forecast', 'N/A')
        strain_predicted = scenario_data.get('strain_prediction', False) or scenario_data.get('strain_predicted', False)
        
        scenario_summary = html.Div([
            html.H6(f"🌤️ {scenario_name.replace('_', ' ').title()}", className="fw-bold mb-2"),
            html.Ul([
                html.Li(f"Total Cohorts: {cohort_count}"),
                html.Li(f"Capacity Forecast: {capacity:.1f} kWh" if isinstance(capacity, (int, float)) else f"Capacity Forecast: {capacity}"),
                html.Li(f"Strain Predicted: {'Yes' if strain_predicted else 'No'}"),
                html.Li(f"Model Accuracy: 20-25% MAPE")
            ], className="list-unstyled small")
        ], className="mb-3 p-2 border rounded")
        
        summary_items.append(scenario_summary)
    
    return html.Div([
        html.H5("LSTM Integration Summary", className="mb-3"),
        html.Div(summary_items)
    ])

# Callback to update strain summary chart
@app.callback(
    Output("strain-summary-chart", "figure"),
    Input("load-lstm-btn", "n_clicks")
)
def update_strain_summary(n_clicks):
    """Update strain summary chart."""
    try:
        if not n_clicks:
            return go.Figure().add_annotation(text="Please load LSTM data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Load data directly
        data_manager = DataManager("data")
        lstm_data = data_manager.load_lstm_integration_results()
        
        if lstm_data:
            return EnergyVisualizations.create_strain_prediction_summary(lstm_data)
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    except Exception as e:
        print(f"Error updating strain summary: {e}")
        return go.Figure().add_annotation(text="Chart loading...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback to update scenario comparison chart
@app.callback(
    Output("scenario-comparison-chart", "figure"),
    Input("load-lstm-btn", "n_clicks")
)
def update_scenario_comparison(n_clicks):
    """Update scenario comparison chart."""
    try:
        if not n_clicks:
            return go.Figure().add_annotation(text="Please load LSTM data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Load data directly
        data_manager = DataManager("data")
        lstm_data = data_manager.load_lstm_integration_results()
        
        if lstm_data:
            return EnergyVisualizations.create_weather_scenario_comparison(lstm_data)
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    except Exception as e:
        print(f"Error updating scenario comparison: {e}")
        return go.Figure().add_annotation(text="Chart loading...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback to update performance validation chart
@app.callback(
    Output("performance-validation-chart", "figure"),
    Input("load-lstm-btn", "n_clicks")
)
def update_performance_validation(n_clicks):
    """Update performance validation chart."""
    try:
        if not n_clicks:
            return go.Figure().add_annotation(text="Please load LSTM data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Generate sample performance metrics
        performance_metrics = {
            'MAPE': 22.5,
            'RMSE': 156.8,
            'MAE': 134.2,
            'R2': 0.78
        }
        return EnergyVisualizations.create_performance_validation_chart(performance_metrics)
    except Exception as e:
        print(f"Error updating performance validation: {e}")
        return go.Figure().add_annotation(text="Chart loading...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback to update weather scenario summary chart
@app.callback(
    Output("weather-scenario-summary-chart", "figure"),
    Input("load-lstm-btn", "n_clicks")
)
def update_weather_scenario_summary(n_clicks):
    """Update weather scenario summary chart."""
    try:
        if not n_clicks:
            return go.Figure().add_annotation(text="Please load LSTM data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Load data directly
        data_manager = DataManager("data")
        lstm_data = data_manager.load_lstm_integration_results()
        
        if lstm_data:
            return EnergyVisualizations.create_weather_scenario_summary(lstm_data)
        return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    except Exception as e:
        print(f"Error updating weather scenario summary: {e}")
        return go.Figure().add_annotation(text="Chart loading...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

# Callback to toggle comparison view
@app.callback(
    Output("compare-scenarios-btn", "children"),
    Input("compare-scenarios-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_comparison_view(n_clicks):
    """Toggle comparison button text."""
    if n_clicks and n_clicks % 2 == 1:
        return "Hide Comparison"
    return "Show Comparison"

# Callback to toggle comparison chart visibility
@app.callback(
    Output("scenario-comparison-chart", "style"),
    Input("compare-scenarios-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_comparison_chart_visibility(n_clicks):
    """Toggle comparison chart visibility."""
    if n_clicks and n_clicks % 2 == 1:
        return {"display": "block"}
    return {"display": "none"}

if __name__ == "__main__":
    app.run_server(debug=True) 