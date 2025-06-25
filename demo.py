#!/usr/bin/env python3
"""
Demo script for Customer Support Reinforcement Learning System
"""

import pandas as pd
import numpy as np
from customer_support_rl import CustomerSupportRL
import warnings
warnings.filterwarnings('ignore')

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def demo_basic_functionality():
    """Demonstrate basic RL system functionality"""
    
    print_header("CUSTOMER SUPPORT RL SYSTEM DEMO")
    
    # Initialize the system
    print("1. Initializing RL System...")
    rl_system = CustomerSupportRL('customer_support_data.csv')
    
    # Train the model
    print("\n2. Training the model...")
    model = rl_system.train_model(total_timesteps=3000)  # Shorter for demo
    
    # Evaluate the model
    print("\n3. Evaluating model performance...")
    evaluation = rl_system.evaluate_model(num_episodes=20)
    
    print(f"   Average Reward: {evaluation['avg_reward']:.2f}")
    print(f"   Prediction Accuracy: {evaluation['accuracy']:.2%}")
    
    return rl_system

def demo_predictions(rl_system):
    """Demonstrate prediction capabilities"""
    
    print_header("PREDICTION DEMONSTRATION")
    
    # Test cases
    test_cases = [
        {
            'error': 'Login failed - Invalid credentials',
            'product': 'WebApp Pro',
            'severity': 'High',
            'customer': 'Premium User'
        },
        {
            'error': 'Payment processing error - Transaction declined',
            'product': 'E-Commerce Suite',
            'severity': 'Critical',
            'customer': 'New Customer'
        },
        {
            'error': 'Database connection timeout',
            'product': 'Cloud Platform',
            'severity': 'Medium',
            'customer': 'Enterprise Client'
        },
        {
            'error': 'Email delivery failed - SMTP error',
            'product': 'Email Service',
            'severity': 'Medium',
            'customer': 'Marketing Team'
        },
        {
            'error': 'API rate limit exceeded',
            'product': 'API Gateway',
            'severity': 'High',
            'customer': 'Developer'
        }
    ]
    
    print("Making predictions for test cases:\n")
    
    for i, case in enumerate(test_cases, 1):
        prediction = rl_system.predict_resolution(
            error_description=case['error'],
            product_name=case['product'],
            severity=case['severity'],
            customer_type=case['customer']
        )
        
        print(f"Case {i}:")
        print(f"  Error: {case['error']}")
        print(f"  Product: {case['product']}")
        print(f"  Severity: {case['severity']}")
        print(f"  Customer: {case['customer']}")
        print(f"  Predicted Resolution: {prediction}")
        print()

def demo_data_analysis():
    """Demonstrate data analysis capabilities"""
    
    print_header("DATA ANALYSIS")
    
    # Load data
    data = pd.read_csv('customer_support_data.csv')
    
    # Basic statistics
    print(f"Total Cases: {len(data)}")
    print(f"Unique Products: {data['product_name'].nunique()}")
    print(f"Unique Resolution Types: {data['resolution_type'].nunique()}")
    
    # Severity distribution
    print(f"\nSeverity Distribution:")
    severity_counts = data['severity'].value_counts()
    for severity, count in severity_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {severity}: {count} cases ({percentage:.1f}%)")
    
    # Resolution type distribution
    print(f"\nTop Resolution Types:")
    resolution_counts = data['resolution_type'].value_counts().head(5)
    for resolution, count in resolution_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {resolution}: {count} cases ({percentage:.1f}%)")
    
    # Customer satisfaction analysis
    print(f"\nCustomer Satisfaction Analysis:")
    satisfaction_mapping = {
        'Excellent - 5 stars': 5,
        'Very Good - 4 stars': 4,
        'Good - 4 stars': 4,
        'Satisfactory - 3 stars': 3,
        'Poor - 2 stars': 2,
        'Very Poor - 1 star': 1
    }
    
    data['satisfaction_score'] = data['customer_feedback'].map(satisfaction_mapping)
    avg_satisfaction = data['satisfaction_score'].mean()
    print(f"  Average Satisfaction: {avg_satisfaction:.2f}/5.0")
    
    # Satisfaction by severity
    severity_satisfaction = data.groupby('severity')['satisfaction_score'].mean()
    print(f"\nSatisfaction by Severity:")
    for severity, score in severity_satisfaction.items():
        print(f"  {severity}: {score:.2f}/5.0")

def demo_performance_insights(rl_system):
    """Demonstrate performance insights"""
    
    print_header("PERFORMANCE INSIGHTS")
    
    # Analyze performance
    analysis = rl_system.analyze_performance()
    
    print("Key Performance Metrics:")
    print(f"  Total Cases Analyzed: {analysis['resolution_counts'].sum()}")
    print(f"  Average Customer Satisfaction: {analysis['avg_satisfaction']:.2f}/5.0")
    
    # Resolution performance
    print(f"\nResolution Type Performance:")
    for resolution, stats in analysis['resolution_performance'].iterrows():
        print(f"  {resolution}: {stats['mean']:.2f} avg score ({stats['count']} cases)")
    
    # Product performance
    print(f"\nProduct Performance:")
    product_satisfaction = analysis['resolution_performance'].groupby(level=0)['mean'].mean()
    for product, score in product_satisfaction.items():
        print(f"  {product}: {score:.2f}/5.0")

def demo_recommendations():
    """Demonstrate recommendation capabilities"""
    
    print_header("RECOMMENDATION SYSTEM")
    
    # Load data for recommendations
    data = pd.read_csv('customer_support_data.csv')
    
    # Example scenarios
    scenarios = [
        {
            'name': 'High Severity WebApp Issue',
            'product': 'WebApp Pro',
            'severity': 'High'
        },
        {
            'name': 'Critical Payment Issue',
            'product': 'E-Commerce Suite',
            'severity': 'Critical'
        },
        {
            'name': 'Medium Severity API Issue',
            'product': 'API Gateway',
            'severity': 'Medium'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        
        # Find similar cases
        similar_cases = data[
            (data['product_name'] == scenario['product']) &
            (data['severity'] == scenario['severity'])
        ]
        
        if len(similar_cases) == 0:
            similar_cases = data  # Use all data if no similar cases
        
        # Get satisfaction scores
        satisfaction_mapping = {
            'Excellent - 5 stars': 5,
            'Very Good - 4 stars': 4,
            'Good - 4 stars': 4,
            'Satisfactory - 3 stars': 3,
            'Poor - 2 stars': 2,
            'Very Poor - 1 star': 1
        }
        
        similar_cases['satisfaction_score'] = similar_cases['customer_feedback'].map(satisfaction_mapping)
        
        # Get top resolutions
        top_resolutions = similar_cases.groupby('resolution_type')['satisfaction_score'].mean().sort_values(ascending=False)
        
        print(f"  Top recommended resolutions:")
        for i, (resolution, score) in enumerate(top_resolutions.head(3).items(), 1):
            print(f"    {i}. {resolution} (Avg satisfaction: {score:.2f}/5.0)")

def main():
    """Main demo function"""
    
    try:
        # Basic functionality demo
        rl_system = demo_basic_functionality()
        
        # Prediction demo
        demo_predictions(rl_system)
        
        # Data analysis demo
        demo_data_analysis()
        
        # Performance insights demo
        demo_performance_insights(rl_system)
        
        # Recommendations demo
        demo_recommendations()
        
        print_header("DEMO COMPLETED")
        print("The Customer Support RL System is ready for use!")
        print("\nNext steps:")
        print("1. Run 'python api_interface.py' to start the API server")
        print("2. Run 'python visualization_dashboard.py' for visualizations")
        print("3. Use the system with your own data by updating the CSV file")
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        print("Please ensure all dependencies are installed and the CSV file exists.")

if __name__ == "__main__":
    main() 