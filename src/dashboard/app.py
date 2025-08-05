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

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Energy Recommendation Engine Dashboard",
    suppress_callback_exceptions=True
)

# Sample data for demonstration
def generate_sample_data():
    """Generate sample energy consumption data for demonstration."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    data = {
        'date': dates,
        'energy_consumption': np.random.normal(100, 20, len(dates)),
        'temperature': np.random.normal(20, 10, len(dates)),
        'building_type': np.random.choice(['Office', 'Residential', 'Industrial'], len(dates)),
        'efficiency_score': np.random.uniform(0.3, 0.9, len(dates))
    }
    
    return pd.DataFrame(data)

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Energy Recommendation Engine Dashboard", 
                   className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Key Metrics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Total Energy Saved", className="text-success"),
                            html.H2("1,234 kWh", id="energy-saved")
                        ]),
                        dbc.Col([
                            html.H4("Cost Reduction", className="text-info"),
                            html.H2("$567", id="cost-reduction")
                        ]),
                        dbc.Col([
                            html.H4("Efficiency Score", className="text-warning"),
                            html.H2("78%", id="efficiency-score")
                        ])
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Energy Consumption Over Time"),
                dbc.CardBody([
                    dcc.Graph(id="energy-consumption-chart")
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Building Type Distribution"),
                dbc.CardBody([
                    dcc.Graph(id="building-type-chart")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recommendations"),
                dbc.CardBody([
                    html.Ul([
                        html.Li("Upgrade HVAC system to reduce energy consumption by 15%"),
                        html.Li("Install LED lighting to save $200/month"),
                        html.Li("Implement smart thermostats for better temperature control"),
                        html.Li("Consider solar panel installation for renewable energy")
                    ])
                ])
            ])
        ])
    ]),
    
    # Hidden div for storing data
    dcc.Store(id="data-store")
    
], fluid=True)

# Callbacks
@app.callback(
    Output("energy-consumption-chart", "figure"),
    Input("data-store", "data")
)
def update_energy_consumption_chart(data):
    """Update energy consumption chart."""
    if data is None:
        df = generate_sample_data()
    else:
        df = pd.DataFrame(data)
    
    fig = px.line(
        df, 
        x='date', 
        y='energy_consumption',
        title="Daily Energy Consumption",
        labels={'energy_consumption': 'Energy Consumption (kWh)', 'date': 'Date'}
    )
    fig.update_layout(height=400)
    return fig

@app.callback(
    Output("building-type-chart", "figure"),
    Input("data-store", "data")
)
def update_building_type_chart(data):
    """Update building type distribution chart."""
    if data is None:
        df = generate_sample_data()
    else:
        df = pd.DataFrame(data)
    
    building_counts = df['building_type'].value_counts()
    
    fig = px.pie(
        values=building_counts.values,
        names=building_counts.index,
        title="Building Type Distribution"
    )
    fig.update_layout(height=400)
    return fig

@app.callback(
    Output("data-store", "data"),
    Input("_pages_location", "pathname")
)
def load_data(pathname):
    """Load data for the dashboard."""
    df = generate_sample_data()
    return df.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 