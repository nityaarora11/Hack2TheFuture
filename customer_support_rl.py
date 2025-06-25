import pandas as pd
import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class CustomerSupportEnvironment(gym.Env):
    """
    Custom Gym Environment for Customer Support Reinforcement Learning
    """
    
    def __init__(self, csv_file_path, max_steps=100):
        super(CustomerSupportEnvironment, self).__init__()
        
        # Load and preprocess data
        self.data = pd.read_csv(csv_file_path)
        self.max_steps = max_steps
        self.current_step = 0
        
        # Preprocess data
        self._preprocess_data()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.resolution_types))
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.feature_dim,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _preprocess_data(self):
        """Preprocess the CSV data for RL training"""
        
        # Text preprocessing for error descriptions
        self.tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        error_features = self.tfidf.fit_transform(self.data['error_description'])
        
        # Encode categorical variables
        self.product_encoder = LabelEncoder()
        self.severity_encoder = LabelEncoder()
        self.resolution_encoder = LabelEncoder()
        
        product_encoded = self.product_encoder.fit_transform(self.data['product_name'])
        severity_encoded = self.severity_encoder.fit_transform(self.data['severity'])
        resolution_encoded = self.resolution_encoder.fit_transform(self.data['resolution_type'])
        
        # Get unique resolution types for action space
        self.resolution_types = self.data['resolution_type'].unique()
        
        # Customer sentiment analysis
        sentiment_scores = []
        for feedback in self.data['customer_feedback']:
            blob = TextBlob(feedback)
            sentiment_scores.append(blob.sentiment.polarity)
        
        # Create feature matrix
        self.features = np.hstack([
            error_features.toarray(),
            product_encoded.reshape(-1, 1),
            severity_encoded.reshape(-1, 1),
            np.array(sentiment_scores).reshape(-1, 1)
        ])
        
        self.feature_dim = self.features.shape[1]
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Create reward mapping based on customer feedback
        self.reward_mapping = {
            'Excellent - 5 stars': 10,
            'Very Good - 4 stars': 8,
            'Good - 4 stars': 6,
            'Satisfactory - 3 stars': 4,
            'Poor - 2 stars': 2,
            'Very Poor - 1 star': 0
        }
        
        # Create severity penalty mapping
        self.severity_penalty = {
            'Critical': 5,
            'High': 3,
            'Medium': 1,
            'Low': 0
        }
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.current_case_idx = np.random.randint(0, len(self.data))
        self.current_state = self.features[self.current_case_idx]
        return self.current_state
    
    def step(self, action):
        """Execute one step in the environment"""
        
        # Get current case data
        current_case = self.data.iloc[self.current_case_idx]
        
        # Calculate reward based on action and actual resolution
        actual_resolution = current_case['resolution_type']
        predicted_resolution = self.resolution_types[action]
        
        # Base reward from customer feedback
        feedback = current_case['customer_feedback']
        base_reward = self.reward_mapping.get(feedback, 5)
        
        # Penalty for incorrect resolution prediction
        if predicted_resolution != actual_resolution:
            base_reward -= 3
        
        # Severity-based penalty (higher severity should be resolved faster)
        severity = current_case['severity']
        severity_penalty = self.severity_penalty.get(severity, 0)
        
        # Time penalty
        time_penalty = self.current_step * 0.1
        
        total_reward = base_reward - severity_penalty - time_penalty
        
        # Move to next case
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if not done:
            self.current_case_idx = np.random.randint(0, len(self.data))
            self.current_state = self.features[self.current_case_idx]
        
        return self.current_state, total_reward, done, {}
    
    def get_case_info(self, case_idx):
        """Get detailed information about a specific case"""
        return self.data.iloc[case_idx]

class CustomerSupportRL:
    """
    Main class for Customer Support Reinforcement Learning
    """
    
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.env = None
        self.model = None
        self.training_history = []
        
    def create_environment(self):
        """Create the RL environment"""
        self.env = CustomerSupportEnvironment(self.csv_file_path)
        return self.env
    
    def train_model(self, total_timesteps=10000, learning_rate=0.0003):
        """Train the RL model"""
        
        if self.env is None:
            self.create_environment()
        
        # Create vectorized environment
        env = DummyVecEnv([lambda: self.env])
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
        # Train the model
        print("Training the RL model...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training completed!")
        
        return self.model
    
    def evaluate_model(self, num_episodes=100):
        """Evaluate the trained model"""
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        total_rewards = []
        correct_predictions = 0
        total_predictions = 0
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                # Check if prediction was correct
                current_case = self.env.get_case_info(self.env.current_case_idx)
                predicted_resolution = self.env.resolution_types[action]
                actual_resolution = current_case['resolution_type']
                
                if predicted_resolution == actual_resolution:
                    correct_predictions += 1
                total_predictions += 1
            
            total_rewards.append(episode_reward)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_reward = np.mean(total_rewards)
        
        print(f"Model Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Prediction Accuracy: {accuracy:.2%}")
        
        return {
            'avg_reward': avg_reward,
            'accuracy': accuracy,
            'total_rewards': total_rewards
        }
    
    def predict_resolution(self, error_description, product_name, severity, customer_type):
        """Predict optimal resolution for a new case"""
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Preprocess input
        error_features = self.env.tfidf.transform([error_description])
        product_encoded = self.env.product_encoder.transform([product_name])
        severity_encoded = self.env.severity_encoder.transform([severity])
        
        # Create feature vector (simplified - you might want to add more features)
        features = np.hstack([
            error_features.toarray(),
            product_encoded.reshape(1, 1),
            severity_encoded.reshape(1, 1),
            np.array([0.0]).reshape(1, 1)  # Placeholder for sentiment
        ])
        
        features = self.env.scaler.transform(features)
        
        # Get prediction
        action, _ = self.model.predict(features, deterministic=True)
        predicted_resolution = self.env.resolution_types[action]
        
        return predicted_resolution
    
    def predict_top_k_resolutions(self, error_description, product_name, severity, customer_type, k=3):
        """
        Predict the top-k best resolutions for a new case.
        Returns a list of (resolution_type, probability) tuples.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Preprocess input
        error_features = self.env.tfidf.transform([error_description])
        product_encoded = self.env.product_encoder.transform([product_name])
        severity_encoded = self.env.severity_encoder.transform([severity])

        features = np.hstack([
            error_features.toarray(),
            product_encoded.reshape(1, 1),
            severity_encoded.reshape(1, 1),
            np.array([0.0]).reshape(1, 1)  # Placeholder for sentiment
        ])
        features = self.env.scaler.transform(features)

        # Get action probabilities from the policy
        obs_tensor = self.model.policy.obs_to_tensor(features)[0]
        action_dist = self.model.policy.get_distribution(obs_tensor)
        action_probs = action_dist.distribution.probs.detach().cpu().numpy().flatten()

        # Get top-k actions
        top_k_indices = np.argsort(action_probs)[::-1][:k]
        top_k_resolutions = [(self.env.resolution_types[i], float(action_probs[i])) for i in top_k_indices]

        return top_k_resolutions
    
    def analyze_performance(self):
        """Analyze model performance and generate insights"""
        
        # Load data for analysis
        data = pd.read_csv(self.csv_file_path)
        
        # Resolution type distribution
        resolution_counts = data['resolution_type'].value_counts()
        
        # Severity distribution
        severity_counts = data['severity'].value_counts()
        
        # Customer feedback analysis
        feedback_scores = []
        for feedback in data['customer_feedback']:
            score = self.env.reward_mapping.get(feedback, 5)
            feedback_scores.append(score)
        
        data['feedback_score'] = feedback_scores
        
        # Performance by resolution type
        resolution_performance = data.groupby('resolution_type')['feedback_score'].agg(['mean', 'count'])
        
        print("\n=== Performance Analysis ===")
        print(f"Total Cases: {len(data)}")
        print(f"Average Customer Satisfaction: {np.mean(feedback_scores):.2f}")
        print(f"Most Common Resolution: {resolution_counts.index[0]} ({resolution_counts.iloc[0]} cases)")
        print(f"Highest Severity Cases: {severity_counts.get('Critical', 0)} Critical cases")
        
        print("\n=== Resolution Type Performance ===")
        for resolution, stats in resolution_performance.iterrows():
            print(f"{resolution}: {stats['mean']:.2f} avg score ({stats['count']} cases)")
        
        return {
            'resolution_counts': resolution_counts,
            'severity_counts': severity_counts,
            'resolution_performance': resolution_performance,
            'avg_satisfaction': np.mean(feedback_scores)
        }

def main():
    """Main function to run the RL system"""
    
    # Initialize the RL system
    rl_system = CustomerSupportRL('customer_support_data.csv')
    
    # Train the model
    print("=== Customer Support RL System ===")
    model = rl_system.train_model(total_timesteps=5000)
    
    # Evaluate the model
    print("\n=== Model Evaluation ===")
    evaluation_results = rl_system.evaluate_model(num_episodes=50)
    
    # Analyze performance
    print("\n=== Performance Analysis ===")
    analysis_results = rl_system.analyze_performance()
    
    # Test predictions
    print("\n=== Test Predictions ===")
    test_cases = [
        {
            'error': 'Login failed - Invalid credentials',
            'product': 'WebApp Pro',
            'severity': 'High',
            'customer': 'Premium User'
        },
        {
            'error': 'Payment processing error',
            'product': 'E-Commerce Suite',
            'severity': 'Critical',
            'customer': 'New Customer'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        prediction = rl_system.predict_resolution(
            case['error'], case['product'], case['severity'], case['customer']
        )
        print(f"Case {i}: Predicted resolution = {prediction}")
    
    print("\n=== RL System Ready for Production Use ===")

if __name__ == "__main__":
    main() 