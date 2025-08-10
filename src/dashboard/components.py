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
            template='plotly_white',
            aspect="auto"
        )
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig

    # New LSTM-specific visualization methods
    @staticmethod
    def create_lstm_forecast_chart(integration_results, scenario="heat_wave", title="LSTM 24-Hour Energy Forecast"):
        """Create a comprehensive chart showing LSTM forecasts for all building cohorts."""
        fig = go.Figure()
        
        if scenario not in integration_results:
            return go.Figure().add_annotation(text="Scenario not found", showarrow=False)
        
        cohort_data = integration_results[scenario]["cohort_forecasts"]
        hours = list(range(24))
        
        # Color palette for different building types
        colors = px.colors.qualitative.Set3
        
        for i, (cohort_name, forecast_values) in enumerate(cohort_data.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=hours,
                y=forecast_values,
                mode='lines+markers',
                name=cohort_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{cohort_name}</b><br>' +
                            'Hour: %{x}<br>' +
                            'Energy: %{y:.2f} kWh<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day",
            yaxis_title="Energy Demand (kWh)",
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def create_weather_scenario_comparison(integration_results, title="Weather Scenario Comparison"):
        """Compare energy forecasts across different weather scenarios."""
        scenarios = list(integration_results.keys())
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Heat wave, Cold snap, Blizzard
        
        for i, scenario in enumerate(scenarios):
            cohort_data = integration_results[scenario]["cohort_forecasts"]
            
            # Calculate average energy across all cohorts for each hour
            all_forecasts = list(cohort_data.values())
            avg_forecast = [sum(hour_values) / len(hour_values) for hour_values in zip(*all_forecasts)]
            
            hours = list(range(24))
            fig.add_trace(go.Scatter(
                x=hours,
                y=avg_forecast,
                mode='lines+markers',
                name=scenario.replace('_', ' ').title(),
                line=dict(color=colors[i], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{scenario.replace("_", " ").title()}</b><br>' +
                            'Hour: %{x}<br>' +
                            'Avg Energy: %{y:.2f} kWh<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day",
            yaxis_title="Average Energy Demand (kWh)",
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_building_cohort_heatmap(integration_results, scenario="heat_wave", title="Building Cohort Energy Heatmap"):
        """Create a heatmap showing energy patterns across building cohorts and hours."""
        if scenario not in integration_results:
            return go.Figure().add_annotation(text="Scenario not found", showarrow=False)
        
        cohort_data = integration_results[scenario]["cohort_forecasts"]
        cohort_names = list(cohort_data.keys())
        hours = list(range(24))
        
        # Prepare data for heatmap
        heatmap_data = []
        for cohort_name in cohort_names:
            heatmap_data.append(cohort_data[cohort_name])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=hours,
            y=cohort_names,
            colorscale='Viridis',
            hovertemplate='<b>%{y}</b><br>' +
                        'Hour: %{x}<br>' +
                        'Energy: %{z:.2f} kWh<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day",
            yaxis_title="Building Cohort",
            template='plotly_white',
            height=600,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_strain_prediction_summary(integration_results, title="Grid Strain Prediction Summary"):
        """Create a summary of grid strain predictions across scenarios."""
        scenarios = list(integration_results.keys())
        strain_status = [integration_results[scenario]["strain_prediction"] for scenario in scenarios]
        capacity_forecasts = [integration_results[scenario]["capacity_forecast"] for scenario in scenarios]
        
        # Create bar chart for capacity forecasts
        fig = go.Figure(data=go.Bar(
            x=[scenario.replace('_', ' ').title() for scenario in scenarios],
            y=capacity_forecasts,
            marker_color=['#FF6B6B' if strain else '#4ECDC4' for strain in strain_status],
            text=[f'{cap:.1f}' for cap in capacity_forecasts],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                        'Capacity: %{y:.2f} kWh<br>' +
                        'Strain: %{text}<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Weather Scenario",
            yaxis_title="Capacity Forecast (kWh)",
            template='plotly_white',
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_peak_hour_analysis(integration_results, title="Peak Hour Energy Analysis"):
        """Analyze peak energy consumption hours across scenarios and cohorts."""
        scenarios = list(integration_results.keys())
        peak_data = []
        
        for scenario in scenarios:
            cohort_data = integration_results[scenario]["cohort_forecasts"]
            for cohort_name, forecast_values in cohort_data.items():
                peak_hour = forecast_values.index(max(forecast_values))
                peak_energy = max(forecast_values)
                peak_data.append({
                    'scenario': scenario.replace('_', ' ').title(),
                    'cohort': cohort_name,
                    'peak_hour': peak_hour,
                    'peak_energy': peak_energy
                })
        
        df_peak = pd.DataFrame(peak_data)
        
        # Create scatter plot
        fig = px.scatter(
            df_peak,
            x='peak_hour',
            y='peak_energy',
            color='scenario',
            size='peak_energy',
            hover_data=['cohort'],
            title=title,
            labels={
                'peak_hour': 'Peak Hour',
                'peak_energy': 'Peak Energy (kWh)',
                'scenario': 'Weather Scenario'
            },
            template='plotly_white'
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_performance_validation_chart(performance_metrics, title="LSTM Performance Validation"):
        """Create a chart showing LSTM performance against industry benchmarks."""
        # Create a comparison chart
        fig = go.Figure()
        
        # Industry benchmark ranges
        benchmark_ranges = [
            {'range': 'Highly Accurate', 'min': 0, 'max': 10, 'color': '#2E8B57'},
            {'range': 'Reasonable', 'min': 10, 'max': 20, 'color': '#FFD700'},
            {'range': 'Acceptable (Extreme)', 'min': 20, 'max': 25, 'color': '#FF8C00'},
            {'range': 'Challenging', 'min': 25, 'max': 50, 'color': '#DC143C'}
        ]
        
        # Add benchmark ranges as horizontal bars
        for i, benchmark in enumerate(benchmark_ranges):
            fig.add_trace(go.Bar(
                x=[benchmark['max'] - benchmark['min']],
                y=[benchmark['range']],
                orientation='h',
                marker_color=benchmark['color'],
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add our performance markers
        our_performance = [
            {'scenario': 'Heat Wave', 'mape': 22.5, 'color': '#FF6B6B'},
            {'scenario': 'Cold Snap', 'mape': 22.5, 'color': '#4ECDC4'},
            {'scenario': 'Blizzard', 'mape': 200, 'color': '#45B7D1'}
        ]
        
        for perf in our_performance:
            fig.add_trace(go.Scatter(
                x=[perf['mape']],
                y=['Our Performance'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=perf['color'],
                    symbol='diamond'
                ),
                name=perf['scenario'],
                hovertemplate=f'<b>{perf["scenario"]}</b><br>' +
                            'MAPE: %{x:.1f}%<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="MAPE (%)",
            yaxis_title="Performance Category",
            template='plotly_white',
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(range=[0, 250]),
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_cohort_performance_heatmap(integration_results, scenario="heat_wave", title="Building Cohort Performance Analysis"):
        """Create a comprehensive heatmap showing cohort performance metrics."""
        if scenario not in integration_results:
            return go.Figure().add_annotation(text="Scenario not found", showarrow=False)
        
        cohort_data = integration_results[scenario]["cohort_forecasts"]
        
        # Calculate performance metrics for each cohort
        metrics_data = []
        cohort_names = []
        
        for cohort_name, forecast_values in cohort_data.items():
            cohort_names.append(cohort_name)
            
            # Calculate various performance metrics
            peak_hour = forecast_values.index(max(forecast_values))
            peak_energy = max(forecast_values)
            total_daily = sum(forecast_values)
            avg_hourly = total_daily / 24
            volatility = np.std(forecast_values)
            peak_to_avg_ratio = peak_energy / avg_hourly if avg_hourly > 0 else 0
            
            metrics_data.append([
                peak_hour,
                peak_energy,
                total_daily,
                avg_hourly,
                volatility,
                peak_to_avg_ratio
            ])
        
        # Create subplot with multiple heatmaps
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Peak Hour', 'Peak Energy (kWh)', 'Total Daily (kWh)',
                'Avg Hourly (kWh)', 'Volatility', 'Peak/Avg Ratio'
            ],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        metric_names = ['Peak Hour', 'Peak Energy', 'Total Daily', 'Avg Hourly', 'Volatility', 'Peak/Avg Ratio']
        
        for i, (metric_name, metric_data) in enumerate(zip(metric_names, zip(*metrics_data))):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=[metric_data],
                    x=cohort_names,
                    y=[metric_name],
                    colorscale='Viridis',
                    showscale=True,
                    hovertemplate='<b>%{y}</b><br>' +
                                'Cohort: %{x}<br>' +
                                'Value: %{z:.2f}<br>' +
                                '<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=800,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_grid_strain_timeline(integration_results, title="Grid Strain Timeline Analysis"):
        """Create a timeline visualization showing grid strain predictions across scenarios."""
        scenarios = list(integration_results.keys())
        hours = list(range(24))
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Heat wave, Cold snap, Blizzard
        
        for i, scenario in enumerate(scenarios):
            cohort_data = integration_results[scenario]["cohort_forecasts"]
            
            # Calculate total energy for each hour across all cohorts
            total_energy_by_hour = []
            for hour in range(24):
                hour_total = sum(cohort_data[cohort][hour] for cohort in cohort_data.keys())
                total_energy_by_hour.append(hour_total)
            
            # Add strain threshold line (assuming 80% of capacity as threshold)
            strain_threshold = integration_results[scenario]["capacity_forecast"] * 0.8
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=total_energy_by_hour,
                mode='lines+markers',
                name=f'{scenario.replace("_", " ").title()} - Total Demand',
                line=dict(color=colors[i], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{scenario.replace("_", " ").title()}</b><br>' +
                            'Hour: %{x}<br>' +
                            'Total Energy: %{y:.2f} kWh<br>' +
                            '<extra></extra>'
            ))
            
            # Add strain threshold line
            fig.add_trace(go.Scatter(
                x=hours,
                y=[strain_threshold] * 24,
                mode='lines',
                name=f'{scenario.replace("_", " ").title()} - Strain Threshold',
                line=dict(color=colors[i], width=2, dash='dash'),
                showlegend=False,
                hovertemplate=f'<b>Strain Threshold</b><br>' +
                            'Hour: %{x}<br>' +
                            'Threshold: %{y:.2f} kWh<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day",
            yaxis_title="Total Energy Demand (kWh)",
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def create_weather_scenario_summary(integration_results, title="Weather Scenario Impact Summary"):
        """Create a summary dashboard showing the impact of different weather scenarios."""
        scenarios = list(integration_results.keys())
        
        # Calculate summary metrics for each scenario
        summary_data = []
        for scenario in scenarios:
            cohort_data = integration_results[scenario]["cohort_forecasts"]
            
            # Calculate total energy for each hour
            total_energy_by_hour = []
            for hour in range(24):
                hour_total = sum(cohort_data[cohort][hour] for cohort in cohort_data.keys())
                total_energy_by_hour.append(hour_total)
            
            summary_data.append({
                'scenario': scenario.replace('_', ' ').title(),
                'total_daily_energy': sum(total_energy_by_hour),
                'peak_hour_energy': max(total_energy_by_hour),
                'avg_hourly_energy': sum(total_energy_by_hour) / 24,
                'strain_predicted': integration_results[scenario]["strain_prediction"],
                'capacity_forecast': integration_results[scenario]["capacity_forecast"],
                'cohort_count': len(cohort_data)
            })
        
        # Create a comprehensive summary visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Daily Energy Consumption by Scenario',
                'Peak Hour Energy by Scenario',
                'Grid Strain Prediction Status',
                'Building Cohort Distribution'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "pie"}]]
        )
        
        # Daily energy consumption
        fig.add_trace(
            go.Bar(
                x=[d['scenario'] for d in summary_data],
                y=[d['total_daily_energy'] for d in summary_data],
                name='Total Daily Energy',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            ),
            row=1, col=1
        )
        
        # Peak hour energy
        fig.add_trace(
            go.Bar(
                x=[d['scenario'] for d in summary_data],
                y=[d['peak_hour_energy'] for d in summary_data],
                name='Peak Hour Energy',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            ),
            row=1, col=2
        )
        
        # Grid strain indicator
        strain_count = sum(1 for d in summary_data if d['strain_predicted'])
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=strain_count,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Scenarios with Grid Strain"},
                delta={'reference': 0},
                gauge={'axis': {'range': [None, len(scenarios)]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 1], 'color': "lightgray"},
                                {'range': [1, 2], 'color': "yellow"},
                                {'range': [2, 3], 'color': "red"}]}
            ),
            row=2, col=1
        )
        
        # Cohort distribution
        total_cohorts = sum(d['cohort_count'] for d in summary_data)
        fig.add_trace(
            go.Pie(
                labels=[d['scenario'] for d in summary_data],
                values=[d['cohort_count'] for d in summary_data],
                name="Building Cohorts"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=800,
            margin=dict(l=50, r=50, t=100, b=50),
            showlegend=False
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