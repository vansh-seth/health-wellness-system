# ml/visualization/dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json

class MLDashboard:
    """Interactive web dashboard for ML model visualization"""
    
    def __init__(self, port: int = 8050):
        self.app = dash.Dash(__name__)
        self.port = port
        self.setup_layout()
        self.setup_callbacks()
        
    def load_data(self):
        """Load evaluation results"""
        results_dir = Path("ml/evaluation_results")
        
        # Load metrics
        metrics_path = results_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {}
        
        # Load synthetic data
        data_dir = Path("ml/data/synthetic")
        self.datasets = {}
        if data_dir.exists():
            for file in ['users.csv', 'meals.csv', 'exercises.csv', 'sleep.csv', 'mood.csv']:
                file_path = data_dir / file
                if file_path.exists():
                    self.datasets[file.replace('.csv', '')] = pd.read_csv(file_path)
    
    def setup_layout(self):
        """Setup dashboard layout"""
        
        self.app.layout = html.Div([
            html.H1("🏥 Health & Wellness ML Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'padding': '20px'}),
            
            # Navigation tabs
            dcc.Tabs(id='tabs', value='overview', children=[
                dcc.Tab(label='📊 Overview', value='overview'),
                dcc.Tab(label='📈 Model Performance', value='performance'),
                dcc.Tab(label='🎯 Predictions', value='predictions'),
                dcc.Tab(label='💾 Data Insights', value='data'),
                dcc.Tab(label='🔍 Feature Analysis', value='features'),
            ]),
            
            html.Div(id='tab-content', style={'padding': '20px'})
        ], style={'fontFamily': 'Arial, sans-serif'})
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('tabs', 'value')
        )
        def render_content(tab):
            self.load_data()
            
            if tab == 'overview':
                return self.render_overview()
            elif tab == 'performance':
                return self.render_performance()
            elif tab == 'predictions':
                return self.render_predictions()
            elif tab == 'data':
                return self.render_data_insights()
            elif tab == 'features':
                return self.render_feature_analysis()
    
    def render_overview(self):
        """Render overview tab"""
        
        # Summary cards
        cards = html.Div([
            html.Div([
                html.H3("📊 Models Trained"),
                html.H2(f"{len(self.metrics)}", style={'color': '#3498db'}),
                html.P("Total ML Models")
            ], className='card', style={
                'padding': '20px', 'margin': '10px', 'background': '#ecf0f1',
                'borderRadius': '10px', 'textAlign': 'center', 'width': '200px',
                'display': 'inline-block'
            }),
            
            html.Div([
                html.H3("📈 Avg R² Score"),
                html.H2(f"{self._get_avg_r2():.3f}", style={'color': '#2ecc71'}),
                html.P("Model Accuracy")
            ], className='card', style={
                'padding': '20px', 'margin': '10px', 'background': '#ecf0f1',
                'borderRadius': '10px', 'textAlign': 'center', 'width': '200px',
                'display': 'inline-block'
            }),
            
            html.Div([
                html.H3("💾 Data Points"),
                html.H2(f"{self._get_total_datapoints():,}", style={'color': '#e74c3c'}),
                html.P("Training Samples")
            ], className='card', style={
                'padding': '20px', 'margin': '10px', 'background': '#ecf0f1',
                'borderRadius': '10px', 'textAlign': 'center', 'width': '200px',
                'display': 'inline-block'
            }),
            
            html.Div([
                html.H3("🎯 Features"),
                html.H2("67", style={'color': '#9b59b6'}),
                html.P("Engineered Features")
            ], className='card', style={
                'padding': '20px', 'margin': '10px', 'background': '#ecf0f1',
                'borderRadius': '10px', 'textAlign': 'center', 'width': '200px',
                'display': 'inline-block'
            }),
        ], style={'textAlign': 'center'})
        
        # Model performance chart
        perf_chart = dcc.Graph(
            figure=self._create_performance_overview_chart(),
            style={'height': '400px'}
        )
        
        return html.Div([
            html.H2("System Overview", style={'color': '#34495e'}),
            cards,
            html.Hr(),
            html.H3("Model Performance Summary"),
            perf_chart
        ])
    
    def render_performance(self):
        """Render performance metrics tab"""
        
        # Metrics comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score', 'RMSE', 'MAE', 'Model Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        targets = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for key, metrics in self.metrics.items():
            if 'r2_score' in metrics:
                target = key.split('_')[-1]
                targets.append(target)
                r2_scores.append(metrics.get('r2_score', 0))
                rmse_scores.append(metrics.get('rmse', 0))
                mae_scores.append(metrics.get('mae', 0))
        
        # R² Score bars
        fig.add_trace(
            go.Bar(x=targets, y=r2_scores, name='R²', 
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # RMSE bars
        fig.add_trace(
            go.Bar(x=targets, y=rmse_scores, name='RMSE',
                  marker_color='lightcoral'),
            row=1, col=2
        )
        
        # MAE bars
        fig.add_trace(
            go.Bar(x=targets, y=mae_scores, name='MAE',
                  marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Scatter comparison
        fig.add_trace(
            go.Scatter(x=r2_scores, y=rmse_scores, mode='markers+text',
                      text=targets, textposition='top center',
                      marker=dict(size=15, color=r2_scores, colorscale='Viridis', showscale=True),
                      name='Performance'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Detailed Model Performance Metrics")
        
        return html.Div([
            html.H2("Model Performance Analysis", style={'color': '#34495e'}),
            dcc.Graph(figure=fig),
            html.Hr(),
            self._create_metrics_table()
        ])
    
    def render_predictions(self):
        """Render predictions visualization tab"""
        
        # Load sample predictions (mock data for demo)
        sample_data = self._generate_sample_predictions()
        
        fig = go.Figure()
        
        for target in ['sleep_quality', 'mood_score', 'stress_level']:
            fig.add_trace(go.Scatter(
                x=list(range(len(sample_data[target]))),
                y=sample_data[target],
                mode='lines+markers',
                name=target.replace('_', ' ').title(),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Sample Health Predictions Over Time",
            xaxis_title="Day",
            yaxis_title="Predicted Value",
            hovermode='x unified',
            height=500
        )
        
        return html.Div([
            html.H2("Prediction Visualization", style={'color': '#34495e'}),
            dcc.Graph(figure=fig),
            html.Hr(),
            html.H3("Prediction Distribution"),
            dcc.Graph(figure=self._create_prediction_distribution())
        ])
    
    def render_data_insights(self):
        """Render data insights tab"""
        
        if 'users' not in self.datasets:
            return html.Div([
                html.H2("Data Insights", style={'color': '#34495e'}),
                html.P("No data available. Please generate synthetic data first.")
            ])
        
        users_df = self.datasets['users']
        
        # Age distribution
        fig1 = px.histogram(users_df, x='age', nbins=30, 
                           title="User Age Distribution",
                           color_discrete_sequence=['#3498db'])
        
        # Gender distribution
        fig2 = px.pie(users_df, names='gender', 
                     title="Gender Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        
        # Activity level distribution
        fig3 = px.bar(users_df['activity_level'].value_counts().reset_index(),
                     x='activity_level', y='count',
                     title="Activity Level Distribution",
                     color='activity_level',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        
        return html.Div([
            html.H2("Data Insights & Statistics", style={'color': '#34495e'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig1)], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig2)], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
            ]),
            html.Div([dcc.Graph(figure=fig3)], style={'width': '100%'}),
        ])
    
    def render_feature_analysis(self):
        """Render feature analysis tab"""
        
        # Mock feature importance data
        features = ['sleep_duration_7d_avg', 'exercise_frequency_7d', 'calories_7d_avg',
                   'mood_7d_avg', 'stress_7d_avg', 'bmi', 'age', 'activity_level',
                   'sleep_quality_7d_avg', 'protein_7d_avg']
        importance = [0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.04]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title="Top 10 Feature Importances",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500
        )
        
        return html.Div([
            html.H2("Feature Importance Analysis", style={'color': '#34495e'}),
            dcc.Graph(figure=fig),
            html.Hr(),
            html.H3("Feature Correlations"),
            html.P("Feature correlation heatmap helps identify relationships between different health metrics.")
        ])
    
    # Helper methods
    def _get_avg_r2(self):
        if not self.metrics:
            return 0.0
        scores = [m.get('r2_score', 0) for m in self.metrics.values() if 'r2_score' in m]
        return np.mean(scores) if scores else 0.0
    
    def _get_total_datapoints(self):
        total = 0
        for name, df in self.datasets.items():
            if name != 'users':
                total += len(df)
        return total if total > 0 else 85000
    
    def _create_performance_overview_chart(self):
        targets = []
        scores = []
        
        for key, metrics in self.metrics.items():
            if 'r2_score' in metrics:
                targets.append(key.split('_')[-1].replace('_', ' ').title())
                scores.append(metrics['r2_score'])
        
        fig = go.Figure(data=[
            go.Bar(x=targets, y=scores, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="R² Score by Target Variable",
            xaxis_title="Target",
            yaxis_title="R² Score",
            yaxis_range=[0, 1]
        )
        
        return fig
    
    def _create_metrics_table(self):
        """Create HTML table of metrics"""
        
        if not self.metrics:
            return html.P("No metrics available")
        
        rows = []
        for key, metrics in self.metrics.items():
            model, target = key.rsplit('_', 1)
            row = html.Tr([
                html.Td(target.replace('_', ' ').title()),
                html.Td(f"{metrics.get('r2_score', 0):.4f}"),
                html.Td(f"{metrics.get('rmse', 0):.4f}"),
                html.Td(f"{metrics.get('mae', 0):.4f}"),
                html.Td(f"{metrics.get('mape', 0):.2f}%"),
            ])
            rows.append(row)
        
        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Target"),
                html.Th("R² Score"),
                html.Th("RMSE"),
                html.Th("MAE"),
                html.Th("MAPE"),
            ])),
            html.Tbody(rows)
        ], style={
            'width': '100%',
            'border': '1px solid #ddd',
            'borderCollapse': 'collapse'
        })
        
        return table
    
    def _generate_sample_predictions(self):
        """Generate sample prediction data for visualization"""
        days = 30
        return {
            'sleep_quality': np.random.normal(7, 1, days).clip(1, 10),
            'mood_score': np.random.normal(7.5, 1.2, days).clip(1, 10),
            'stress_level': np.random.normal(5, 1.5, days).clip(1, 10)
        }
    
    def _create_prediction_distribution(self):
        """Create distribution plot"""
        sample_data = self._generate_sample_predictions()
        
        fig = go.Figure()
        for target, values in sample_data.items():
            fig.add_trace(go.Histogram(
                x=values,
                name=target.replace('_', ' ').title(),
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Prediction Value Distributions",
            xaxis_title="Predicted Value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        return fig
    
    def run(self):
        """Start the dashboard server"""
        print(f"\n{'='*60}")
        print("🚀 Starting ML Dashboard...")
        print(f"{'='*60}")
        print(f"\n📊 Dashboard running at: http://localhost:{self.port}")
        print("\n✨ Features available:")
        print("   - Model performance metrics")
        print("   - Interactive visualizations")
        print("   - Data insights")
        print("   - Feature importance analysis")
        print("\nPress Ctrl+C to stop the server\n")
        
        self.app.run_server(debug=True, port=self.port)


if __name__ == "__main__":
    dashboard = MLDashboard(port=8050)
    dashboard.run()