#!/usr/bin/env python3
"""
Test script for Customer Support Reinforcement Learning System
"""

import pandas as pd
import numpy as np
import os
import sys
from customer_support_rl import CustomerSupportRL, CustomerSupportEnvironment

def test_csv_loading():
    """Test if CSV file can be loaded correctly"""
    print("Testing CSV file loading...")
    
    try:
        data = pd.read_csv('customer_support_data.csv')
        print(f"✓ CSV loaded successfully with {len(data)} rows")
        print(f"  Columns: {list(data.columns)}")
        return True
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return False

def test_environment_creation():
    """Test if RL environment can be created"""
    print("\nTesting RL environment creation...")
    
    try:
        env = CustomerSupportEnvironment('customer_support_data.csv')
        print(f"✓ Environment created successfully")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Number of resolution types: {len(env.resolution_types)}")
        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False

def test_environment_interaction():
    """Test basic environment interactions"""
    print("\nTesting environment interactions...")
    
    try:
        env = CustomerSupportEnvironment('customer_support_data.csv')
        
        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset successful, observation shape: {obs.shape}")
        
        # Test step
        action = 0  # First action
        obs, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        
        return True
    except Exception as e:
        print(f"✗ Failed environment interaction: {e}")
        return False

def test_rl_system_initialization():
    """Test RL system initialization"""
    print("\nTesting RL system initialization...")
    
    try:
        rl_system = CustomerSupportRL('customer_support_data.csv')
        print(f"✓ RL system initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize RL system: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\nTesting data preprocessing...")
    
    try:
        # Load data
        data = pd.read_csv('customer_support_data.csv')
        
        # Test text preprocessing
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=10, stop_words='english')
        error_features = tfidf.fit_transform(data['error_description'])
        print(f"✓ Text preprocessing successful, features: {error_features.shape}")
        
        # Test categorical encoding
        from sklearn.preprocessing import LabelEncoder
        product_encoder = LabelEncoder()
        product_encoded = product_encoder.fit_transform(data['product_name'])
        print(f"✓ Categorical encoding successful, unique products: {len(product_encoder.classes_)}")
        
        # Test sentiment analysis
        from textblob import TextBlob
        sentiment_scores = []
        for feedback in data['customer_feedback']:
            blob = TextBlob(feedback)
            sentiment_scores.append(blob.sentiment.polarity)
        print(f"✓ Sentiment analysis successful, average sentiment: {np.mean(sentiment_scores):.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Failed data preprocessing: {e}")
        return False

def test_prediction_functionality():
    """Test prediction functionality"""
    print("\nTesting prediction functionality...")
    
    try:
        rl_system = CustomerSupportRL('customer_support_data.csv')
        
        # Train a quick model
        print("  Training model (short training for testing)...")
        rl_system.train_model(total_timesteps=1000)  # Very short for testing
        
        # Test prediction
        prediction = rl_system.predict_resolution(
            error_description="Test error description",
            product_name="WebApp Pro",
            severity="High",
            customer_type="Test Customer"
        )
        print(f"✓ Prediction successful: {prediction}")
        
        return True
    except Exception as e:
        print(f"✗ Failed prediction: {e}")
        return False

def test_performance_analysis():
    """Test performance analysis functionality"""
    print("\nTesting performance analysis...")
    
    try:
        rl_system = CustomerSupportRL('customer_support_data.csv')
        
        # Run analysis
        analysis = rl_system.analyze_performance()
        
        print(f"✓ Performance analysis successful")
        print(f"  Total cases: {analysis['resolution_counts'].sum()}")
        print(f"  Average satisfaction: {analysis['avg_satisfaction']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Failed performance analysis: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("CUSTOMER SUPPORT RL SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("CSV Loading", test_csv_loading),
        ("Environment Creation", test_environment_creation),
        ("Environment Interaction", test_environment_interaction),
        ("RL System Initialization", test_rl_system_initialization),
        ("Data Preprocessing", test_data_preprocessing),
        ("Prediction Functionality", test_prediction_functionality),
        ("Performance Analysis", test_performance_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 