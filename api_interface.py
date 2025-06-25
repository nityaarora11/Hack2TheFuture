from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
from customer_support_rl import CustomerSupportRL
import warnings
warnings.filterwarnings('ignore')
from rag_suggestions import find_top_k_similar, df, get_llm_suggestions
import io
import sys

app = Flask(__name__)

# Initialize the RL system
rl_system = None

def initialize_rl_system():
    """Initialize the RL system with training"""
    global rl_system
    rl_system = CustomerSupportRL('customer_support_data.csv')
    rl_system.train_model(total_timesteps=3000)  # Shorter training for demo
    return rl_system

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Customer Support RL API is running'
    })

@app.route('/predict', methods=['POST'])
def predict_resolution():
    """Predict optimal resolution for a new case"""
    
    if rl_system is None:
        return jsonify({'error': 'RL system not initialized'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['error_description', 'product_name', 'severity']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get prediction
        prediction = rl_system.predict_resolution(
            error_description=data['error_description'],
            product_name=data['product_name'],
            severity=data['severity'],
            customer_type=data.get('customer_type', 'Standard')
        )
        
        return jsonify({
            'predicted_resolution': prediction,
            'confidence': 0.85,  # Placeholder confidence score
            'recommended_priority': 'High' if data['severity'] in ['Critical', 'High'] else 'Medium'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['GET'])
def analyze_data():
    """Get analysis of the training data"""
    
    try:
        # Load data
        data = pd.read_csv('customer_support_data.csv')
        
        # Basic statistics
        total_cases = len(data)
        resolution_counts = data['resolution_type'].value_counts().to_dict()
        severity_counts = data['severity'].value_counts().to_dict()
        
        # Customer satisfaction analysis
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
        
        # Product performance
        product_performance = data.groupby('product_name')['satisfaction_score'].agg(['mean', 'count']).round(2)
        product_performance_dict = product_performance.to_dict('index')
        
        return jsonify({
            'total_cases': total_cases,
            'average_satisfaction': float(avg_satisfaction),
            'resolution_distribution': resolution_counts,
            'severity_distribution': severity_counts,
            'product_performance': product_performance_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """Get recommendations based on case details"""
    
    try:
        data = request.get_json()
        
        # Load training data for recommendations
        training_data = pd.read_csv('customer_support_data.csv')
        
        # Find similar cases
        similar_cases = training_data[
            (training_data['product_name'] == data.get('product_name', '')) |
            (training_data['severity'] == data.get('severity', ''))
        ]
        
        if len(similar_cases) == 0:
            similar_cases = training_data  # Use all data if no similar cases
        
        # Get top performing resolutions for similar cases
        satisfaction_mapping = {
            'Excellent - 5 stars': 5,
            'Very Good - 4 stars': 4,
            'Good - 4 stars': 4,
            'Satisfactory - 3 stars': 3,
            'Poor - 2 stars': 2,
            'Very Poor - 1 star': 1
        }
        
        similar_cases['satisfaction_score'] = similar_cases['customer_feedback'].map(satisfaction_mapping)
        
        # Get top resolutions by satisfaction
        top_resolutions = similar_cases.groupby('resolution_type')['satisfaction_score'].mean().sort_values(ascending=False)
        
        recommendations = []
        for resolution, score in top_resolutions.head(3).items():
            # Get example case for this resolution
            example_case = similar_cases[similar_cases['resolution_type'] == resolution].iloc[0]
            
            recommendations.append({
                'resolution_type': resolution,
                'average_satisfaction': float(score),
                'example_error': example_case['error_description'],
                'example_details': example_case['resolution_details']
            })
        
        return jsonify({
            'recommendations': recommendations,
            'similar_cases_analyzed': len(similar_cases)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def retrain_model():
    """Retrain the RL model with new parameters"""
    
    try:
        data = request.get_json() or {}
        
        # Get training parameters
        timesteps = data.get('timesteps', 5000)
        learning_rate = data.get('learning_rate', 0.0003)
        
        # Initialize and train
        global rl_system
        rl_system = CustomerSupportRL('customer_support_data.csv')
        rl_system.train_model(total_timesteps=timesteps, learning_rate=learning_rate)
        
        # Evaluate the model
        evaluation = rl_system.evaluate_model(num_episodes=20)
        
        return jsonify({
            'message': 'Model retrained successfully',
            'training_parameters': {
                'timesteps': timesteps,
                'learning_rate': learning_rate
            },
            'evaluation_results': {
                'average_reward': float(evaluation['avg_reward']),
                'accuracy': float(evaluation['accuracy'])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cases', methods=['GET'])
def get_cases():
    """Get all cases from the dataset"""
    
    try:
        data = pd.read_csv('customer_support_data.csv')
        
        # Convert to list of dictionaries
        cases = data.to_dict('records')
        
        return jsonify({
            'cases': cases,
            'total_count': len(cases)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cases/<int:case_id>', methods=['GET'])
def get_case(case_id):
    """Get a specific case by ID"""
    
    try:
        data = pd.read_csv('customer_support_data.csv')
        
        if case_id >= len(data) or case_id < 0:
            return jsonify({'error': 'Case ID out of range'}), 404
        
        case = data.iloc[case_id].to_dict()
        
        return jsonify({
            'case': case,
            'case_id': case_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rag_suggestions', methods=['POST'])
def rag_suggestions():
    """Expose an endpoint for Agentforce to get LLM suggestions based on user issue"""
    try:
        data = request.get_json()
        user_issue = data.get('user_issue', None)
        if not user_issue:
            return jsonify({'error': 'Missing user_issue in request body'}), 400
        # Find top 3 similar cases
        top_indices = find_top_k_similar(user_issue, k=3)
        top_cases = []
        for idx in top_indices:
            case = df.iloc[idx]
            top_cases.append({'issue_description': case['issue_description'], 'resolution': case['resolution']})
        # Capture LLM output
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        get_llm_suggestions(user_issue, top_cases)
        sys.stdout = old_stdout
        llm_output = mystdout.getvalue()
        return jsonify({'llm_suggestions': llm_output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Initializing Customer Support RL System...")
    initialize_rl_system()
    print("RL System initialized successfully!")
    
    print("\nAPI Endpoints available:")
    print("- GET  /health - Health check")
    print("- POST /predict - Predict resolution for new case")
    print("- GET  /analyze - Get data analysis")
    print("- POST /recommendations - Get recommendations")
    print("- POST /train - Retrain model")
    print("- GET  /cases - Get all cases")
    print("- GET  /cases/<id> - Get specific case")
    print("- POST /rag_suggestions - Get LLM suggestions")
    
    print("\nStarting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000) 