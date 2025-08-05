"""
Dashboard components for Energy Recommendation Engine.

This module contains reusable components and visualization functions
for the dashboard interface.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc


class EnergyVisualizations:
    """Class containing energy-related visualization functions."""
    
    @staticmethod
    def create_energy_consumption_chart(df, title="Energy Consumption Over Time"):
        """Create a line chart for energy consumption over time."""
        fig = px.line(
            df,
            x='date',
            y='energy_consumption',
            title=title,
            labels={'energy_consumption': 'Energy Consumption (kWh)', 'date': 'Date'},
            template='plotly_white'
        )
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified'
        )
        return fig
    
    @staticmethod
    def create_efficiency_score_chart(df, title="Efficiency Score Distribution"):
        """Create a histogram for efficiency scores."""
        fig = px.histogram(
            df,
            x='efficiency_score',
            title=title,
            labels={'efficiency_score': 'Efficiency Score', 'count': 'Count'},
            nbins=20,
            template='plotly_white'
        )
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig
    
    @staticmethod
    def create_temperature_correlation_chart(df, title="Energy vs Temperature"):
        """Create a scatter plot showing correlation between temperature and energy."""
        fig = px.scatter(
            df,
            x='temperature',
            y='energy_consumption',
            color='building_type',
            title=title,
            labels={'temperature': 'Temperature (°C)', 'energy_consumption': 'Energy Consumption (kWh)'},
            template='plotly_white'
        )
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig
    
    @staticmethod
    def create_building_type_chart(df, title="Building Type Distribution"):
        """Create a pie chart for building type distribution."""
        building_counts = df['building_type'].value_counts()
        
        fig = px.pie(
            values=building_counts.values,
            names=building_counts.index,
            title=title,
            template='plotly_white'
        )
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig
    
    @staticmethod
    def create_energy_heatmap(df, title="Energy Consumption Heatmap"):
        """Create a heatmap showing energy consumption patterns."""
        # Reshape data for heatmap
        df_heatmap = df.copy()
        df_heatmap['month'] = df_heatmap['date'].dt.month
        df_heatmap['day_of_week'] = df_heatmap['date'].dt.dayofweek
        
        heatmap_data = df_heatmap.groupby(['month', 'day_of_week'])['energy_consumption'].mean().unstack()
        
        fig = px.imshow(
            heatmap_data,
            title=title,
            labels=dict(x="Day of Week", y="Month", color="Energy Consumption (kWh)"),
            aspect="auto",
            template='plotly_white'
        )
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig


class DashboardComponents:
    """Class containing reusable dashboard UI components."""
    
    @staticmethod
    def create_metric_card(title, value, color="primary", icon=None):
        """Create a metric card component."""
        card_content = [
            html.H4(title, className=f"text-{color}"),
            html.H2(value, id=f"{title.lower().replace(' ', '-')}")
        ]
        
        if icon:
            card_content.insert(0, html.I(className=f"fas fa-{icon} fa-2x mb-2"))
        
        return dbc.Card([
            dbc.CardBody([
                html.Div(card_content, className="text-center")
            ])
        ])
    
    @staticmethod
    def create_chart_card(title, chart_id, height=400):
        """Create a card containing a chart."""
        return dbc.Card([
            dbc.CardHeader(title),
            dbc.CardBody([
                dcc.Graph(id=chart_id, style={'height': height})
            ])
        ])
    
    @staticmethod
    def create_recommendations_card(recommendations):
        """Create a card displaying energy efficiency recommendations."""
        recommendation_items = [
            html.Li(rec) for rec in recommendations
        ]
        
        return dbc.Card([
            dbc.CardHeader("Energy Efficiency Recommendations"),
            dbc.CardBody([
                html.Ul(recommendation_items, className="list-unstyled")
            ])
        ])
    
    @staticmethod
    def create_filters_section():
        """Create a filters section for the dashboard."""
        return dbc.Card([
            dbc.CardHeader("Filters"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Date Range"),
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            calendar_orientation="horizontal"
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Building Type"),
                        dcc.Dropdown(
                            id="building-type-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Office", "value": "office"},
                                {"label": "Residential", "value": "residential"},
                                {"label": "Industrial", "value": "industrial"}
                            ],
                            value="all",
                            clearable=False
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Efficiency Score Range"),
                        dcc.RangeSlider(
                            id="efficiency-range",
                            min=0,
                            max=1,
                            step=0.1,
                            value=[0, 1],
                            marks={i/10: str(i/10) for i in range(0, 11, 2)}
                        )
                    ], width=4)
                ])
            ])
        ])


class DataProcessor:
    """Class for processing and filtering dashboard data."""
    
    @staticmethod
    def filter_data_by_date_range(df, start_date, end_date):
        """Filter dataframe by date range."""
        if start_date and end_date:
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            return df[mask]
        return df
    
    @staticmethod
    def filter_data_by_building_type(df, building_type):
        """Filter dataframe by building type."""
        if building_type and building_type != "all":
            return df[df['building_type'].str.lower() == building_type.lower()]
        return df
    
    @staticmethod
    def filter_data_by_efficiency_range(df, min_efficiency, max_efficiency):
        """Filter dataframe by efficiency score range."""
        mask = (df['efficiency_score'] >= min_efficiency) & (df['efficiency_score'] <= max_efficiency)
        return df[mask]
    
    @staticmethod
    def calculate_summary_metrics(df):
        """Calculate summary metrics for the dashboard."""
        return {
            'total_energy': df['energy_consumption'].sum(),
            'avg_energy': df['energy_consumption'].mean(),
            'total_cost': df['energy_consumption'].sum() * 0.12,  # Assuming $0.12/kWh
            'avg_efficiency': df['efficiency_score'].mean(),
            'energy_saved': df['energy_consumption'].sum() * 0.15,  # Assuming 15% potential savings
            'cost_reduction': df['energy_consumption'].sum() * 0.12 * 0.15
        } 