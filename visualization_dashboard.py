import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class CustomerSupportVisualizer:
    """
    Visualization class for Customer Support RL System
    """
    
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_wordcloud(self, column='error_description', title='Error Description Word Cloud'):
        """Create wordcloud from text data"""
        
        # Combine all text
        text = ' '.join(self.data[column].astype(str))
        
        # Create wordcloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_resolution_distribution(self):
        """Plot resolution type distribution"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Resolution type counts
        resolution_counts = self.data['resolution_type'].value_counts()
        colors = sns.color_palette("husl", len(resolution_counts))
        
        ax1.pie(resolution_counts.values, labels=resolution_counts.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Resolution Type Distribution', fontsize=14, fontweight='bold')
        
        # Severity distribution
        severity_counts = self.data['severity'].value_counts()
        severity_colors = ['#ff4444', '#ff8800', '#ffcc00', '#00cc00']
        
        ax2.bar(severity_counts.index, severity_counts.values, color=severity_colors[:len(severity_counts)])
        ax2.set_title('Case Severity Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Cases')
        
        plt.tight_layout()
        plt.show()
    
    def plot_customer_satisfaction_analysis(self):
        """Analyze and plot customer satisfaction metrics"""
        
        # Extract satisfaction scores
        satisfaction_mapping = {
            'Excellent - 5 stars': 5,
            'Very Good - 4 stars': 4,
            'Good - 4 stars': 4,
            'Satisfactory - 3 stars': 3,
            'Poor - 2 stars': 2,
            'Very Poor - 1 star': 1
        }
        
        self.data['satisfaction_score'] = self.data['customer_feedback'].map(satisfaction_mapping)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall satisfaction distribution
        ax1.hist(self.data['satisfaction_score'], bins=5, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Customer Satisfaction Distribution', fontweight='bold')
        ax1.set_xlabel('Satisfaction Score')
        ax1.set_ylabel('Frequency')
        ax1.axvline(self.data['satisfaction_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.data["satisfaction_score"].mean():.2f}')
        ax1.legend()
        
        # Satisfaction by resolution type
        resolution_satisfaction = self.data.groupby('resolution_type')['satisfaction_score'].mean().sort_values(ascending=False)
        ax2.barh(resolution_satisfaction.index, resolution_satisfaction.values, color='lightgreen')
        ax2.set_title('Average Satisfaction by Resolution Type', fontweight='bold')
        ax2.set_xlabel('Average Satisfaction Score')
        
        # Satisfaction by severity
        severity_satisfaction = self.data.groupby('severity')['satisfaction_score'].mean()
        ax3.bar(severity_satisfaction.index, severity_satisfaction.values, color='orange')
        ax3.set_title('Average Satisfaction by Severity', fontweight='bold')
        ax3.set_ylabel('Average Satisfaction Score')
        
        # Satisfaction by product
        product_satisfaction = self.data.groupby('product_name')['satisfaction_score'].mean().sort_values(ascending=False)
        ax4.barh(product_satisfaction.index, product_satisfaction.values, color='lightcoral')
        ax4.set_title('Average Satisfaction by Product', fontweight='bold')
        ax4.set_xlabel('Average Satisfaction Score')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create an interactive Dash dashboard"""
        
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Customer Support RL Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Key Metrics"),
                        dbc.CardBody([
                            html.H4(f"Total Cases: {len(self.data)}"),
                            html.H4(f"Average Satisfaction: {self.data['satisfaction_score'].mean():.2f}"),
                            html.H4(f"Critical Cases: {len(self.data[self.data['severity'] == 'Critical'])}")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dcc.Graph(
                        id='resolution-pie-chart',
                        figure=self.create_resolution_pie_chart()
                    )
                ], width=9)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='satisfaction-trend',
                        figure=self.create_satisfaction_trend()
                    )
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(
                        id='severity-bar-chart',
                        figure=self.create_severity_bar_chart()
                    )
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='product-performance',
                        figure=self.create_product_performance()
                    )
                ], width=12)
            ])
        ], fluid=True)
        
        return app
    
    def create_resolution_pie_chart(self):
        """Create pie chart for resolution types"""
        resolution_counts = self.data['resolution_type'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=resolution_counts.index,
            values=resolution_counts.values,
            hole=0.3,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title="Resolution Type Distribution",
            showlegend=True
        )
        
        return fig
    
    def create_satisfaction_trend(self):
        """Create satisfaction trend chart"""
        # Create satisfaction scores if not exists
        if 'satisfaction_score' not in self.data.columns:
            satisfaction_mapping = {
                'Excellent - 5 stars': 5,
                'Very Good - 4 stars': 4,
                'Good - 4 stars': 4,
                'Satisfactory - 3 stars': 3,
                'Poor - 2 stars': 2,
                'Very Poor - 1 star': 1
            }
            self.data['satisfaction_score'] = self.data['customer_feedback'].map(satisfaction_mapping)
        
        # Create index for trend
        self.data['case_index'] = range(len(self.data))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data['case_index'],
            y=self.data['satisfaction_score'],
            mode='lines+markers',
            name='Satisfaction Score',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Customer Satisfaction Trend",
            xaxis_title="Case Index",
            yaxis_title="Satisfaction Score",
            yaxis=dict(range=[0, 5])
        )
        
        return fig
    
    def create_severity_bar_chart(self):
        """Create severity distribution bar chart"""
        severity_counts = self.data['severity'].value_counts()
        
        colors = ['#ff4444', '#ff8800', '#ffcc00', '#00cc00']
        
        fig = go.Figure(data=[go.Bar(
            x=severity_counts.index,
            y=severity_counts.values,
            marker_color=colors[:len(severity_counts)]
        )])
        
        fig.update_layout(
            title="Case Severity Distribution",
            xaxis_title="Severity Level",
            yaxis_title="Number of Cases"
        )
        
        return fig
    
    def create_product_performance(self):
        """Create product performance comparison"""
        if 'satisfaction_score' not in self.data.columns:
            satisfaction_mapping = {
                'Excellent - 5 stars': 5,
                'Very Good - 4 stars': 4,
                'Good - 4 stars': 4,
                'Satisfactory - 3 stars': 3,
                'Poor - 2 stars': 2,
                'Very Poor - 1 star': 1
            }
            self.data['satisfaction_score'] = self.data['customer_feedback'].map(satisfaction_mapping)
        
        product_stats = self.data.groupby('product_name').agg({
            'satisfaction_score': ['mean', 'count'],
            'severity': lambda x: (x == 'Critical').sum()
        }).round(2)
        
        product_stats.columns = ['Avg_Satisfaction', 'Case_Count', 'Critical_Cases']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Satisfaction by Product', 'Case Count by Product'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Satisfaction chart
        fig.add_trace(
            go.Bar(x=product_stats.index, y=product_stats['Avg_Satisfaction'], 
                   name='Avg Satisfaction', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Case count chart
        fig.add_trace(
            go.Bar(x=product_stats.index, y=product_stats['Case_Count'], 
                   name='Case Count', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Product Performance Analysis")
        
        return fig
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        
        # Calculate metrics
        total_cases = len(self.data)
        
        # Satisfaction analysis
        satisfaction_mapping = {
            'Excellent - 5 stars': 5,
            'Very Good - 4 stars': 4,
            'Good - 4 stars': 4,
            'Satisfactory - 3 stars': 3,
            'Poor - 2 stars': 2,
            'Very Poor - 1 star': 1
        }
        
        self.data['satisfaction_score'] = self.data['customer_feedback'].map(satisfaction_mapping)
        avg_satisfaction = self.data['satisfaction_score'].mean()
        
        # Severity analysis
        critical_cases = len(self.data[self.data['severity'] == 'Critical'])
        high_severity_cases = len(self.data[self.data['severity'] == 'High'])
        
        # Resolution analysis
        resolution_counts = self.data['resolution_type'].value_counts()
        most_common_resolution = resolution_counts.index[0]
        
        # Product analysis
        product_satisfaction = self.data.groupby('product_name')['satisfaction_score'].mean()
        best_product = product_satisfaction.idxmax()
        worst_product = product_satisfaction.idxmin()
        
        print("=" * 60)
        print("CUSTOMER SUPPORT PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Total Cases Analyzed: {total_cases}")
        print(f"Average Customer Satisfaction: {avg_satisfaction:.2f}/5.0")
        print(f"Critical Cases: {critical_cases} ({critical_cases/total_cases*100:.1f}%)")
        print(f"High Severity Cases: {high_severity_cases} ({high_severity_cases/total_cases*100:.1f}%)")
        print(f"Most Common Resolution: {most_common_resolution}")
        print(f"Best Performing Product: {best_product} ({product_satisfaction[best_product]:.2f}/5.0)")
        print(f"Product Needing Attention: {worst_product} ({product_satisfaction[worst_product]:.2f}/5.0)")
        print("=" * 60)
        
        return {
            'total_cases': total_cases,
            'avg_satisfaction': avg_satisfaction,
            'critical_cases': critical_cases,
            'high_severity_cases': high_severity_cases,
            'most_common_resolution': most_common_resolution,
            'best_product': best_product,
            'worst_product': worst_product
        }

def main():
    """Main function to run visualizations"""
    
    # Initialize visualizer
    visualizer = CustomerSupportVisualizer('customer_support_data.csv')
    
    # Generate wordcloud
    print("Generating wordcloud...")
    visualizer.create_wordcloud()
    
    # Plot distributions
    print("Creating distribution plots...")
    visualizer.plot_resolution_distribution()
    
    # Customer satisfaction analysis
    print("Analyzing customer satisfaction...")
    visualizer.plot_customer_satisfaction_analysis()
    
    # Generate performance report
    print("Generating performance report...")
    report = visualizer.generate_performance_report()
    
    # Create interactive dashboard
    print("Creating interactive dashboard...")
    app = visualizer.create_interactive_dashboard()
    
    print("\nDashboard is ready! Run the following command to start:")
    print("python -c \"from visualization_dashboard import CustomerSupportVisualizer; app = CustomerSupportVisualizer('customer_support_data.csv').create_interactive_dashboard(); app.run_server(debug=True)\"")

if __name__ == "__main__":
    main() 