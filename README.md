# Customer Support Reinforcement Learning System

A comprehensive reinforcement learning system designed to optimize customer support operations by learning from historical data and predicting optimal resolution strategies.

## 🚀 Features

### Core RL System
- **Custom Gym Environment**: Built specifically for customer support scenarios
- **PPO Algorithm**: Uses Proximal Policy Optimization for stable learning
- **Multi-modal Input Processing**: Handles text, categorical, and numerical data
- **Reward Optimization**: Balances customer satisfaction, resolution accuracy, and response time

### Data Processing
- **Text Analysis**: TF-IDF vectorization for error descriptions
- **Sentiment Analysis**: Customer feedback sentiment scoring
- **Feature Engineering**: Automatic encoding and normalization
- **Categorical Encoding**: Label encoding for product names, severity levels

### Visualization & Analytics
- **Interactive Dashboards**: Real-time performance monitoring
- **Word Clouds**: Error description analysis
- **Performance Metrics**: Resolution accuracy, customer satisfaction trends
- **Product Analysis**: Performance comparison across products

### API Interface
- **RESTful API**: Flask-based endpoints for predictions
- **Real-time Predictions**: Instant resolution recommendations
- **Model Retraining**: On-demand model updates
- **Data Analysis**: Comprehensive insights and statistics

## 📊 Sample Data Structure

The system processes CSV files with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `error_description` | Text description of the issue | "Login failed - Invalid credentials" |
| `product_name` | Name of the product/service | "WebApp Pro" |
| `customer_details` | Customer information | "Premium User - 2 years" |
| `severity` | Issue severity level | "High", "Critical", "Medium", "Low" |
| `resolution_type` | Type of resolution applied | "Password Reset", "Bug Fix" |
| `resolution_details` | Detailed resolution steps | "Reset password and enabled 2FA" |
| `customer_feedback` | Customer satisfaction rating | "Excellent - 5 stars" |

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Hack2Future
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Usage

Run the main RL system:
```bash
python customer_support_rl.py
```

This will:
- Load the sample CSV data
- Train the reinforcement learning model
- Evaluate model performance
- Generate predictions for test cases

### 2. Visualization Dashboard

Generate visualizations and reports:
```bash
python visualization_dashboard.py
```

### 3. API Server

Start the REST API server:
```bash
python api_interface.py
```

The API will be available at `http://localhost:5000`

## 📈 API Endpoints

### Health Check
```bash
GET /health
```

### Predict Resolution
```bash
POST /predict
Content-Type: application/json

{
    "error_description": "Payment processing failed",
    "product_name": "E-Commerce Suite",
    "severity": "Critical",
    "customer_type": "Premium"
}
```

### Get Data Analysis
```bash
GET /analyze
```

### Get Recommendations
```bash
POST /recommendations
Content-Type: application/json

{
    "product_name": "WebApp Pro",
    "severity": "High"
}
```

### Retrain Model
```bash
POST /train
Content-Type: application/json

{
    "timesteps": 5000,
    "learning_rate": 0.0003
}
```

## 🧠 How the RL System Works

### Environment Design
1. **State Space**: Feature vector containing:
   - TF-IDF features from error descriptions
   - Encoded product names and severity levels
   - Customer sentiment scores

2. **Action Space**: Discrete actions representing different resolution types

3. **Reward Function**: 
   - Base reward from customer satisfaction
   - Penalty for incorrect predictions
   - Time-based penalties for resolution speed
   - Severity-based urgency factors

### Training Process
1. **Data Preprocessing**: Text vectorization, encoding, normalization
2. **Environment Creation**: Custom Gym environment with customer support logic
3. **Model Training**: PPO algorithm with policy optimization
4. **Evaluation**: Accuracy and reward-based performance metrics

## 📊 Performance Metrics

The system tracks several key performance indicators:

- **Prediction Accuracy**: Percentage of correct resolution predictions
- **Average Reward**: Overall system performance score
- **Customer Satisfaction**: Average satisfaction scores by resolution type
- **Response Time**: Time-based penalties and optimization
- **Severity Handling**: Performance on critical vs. non-critical cases

## 🔧 Customization

### Adding New Features
1. **New Data Columns**: Update the preprocessing in `CustomerSupportEnvironment._preprocess_data()`
2. **Custom Rewards**: Modify the reward function in the `step()` method
3. **Additional Actions**: Extend the action space for new resolution types

### Model Parameters
Adjust training parameters in `CustomerSupportRL.train_model()`:
- `total_timesteps`: Training duration
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size for training
- `n_epochs`: Number of epochs per update

## 📝 Example Usage

### Python Script Example
```python
from customer_support_rl import CustomerSupportRL

# Initialize system
rl_system = CustomerSupportRL('customer_support_data.csv')

# Train model
rl_system.train_model(total_timesteps=5000)

# Make prediction
prediction = rl_system.predict_resolution(
    error_description="Database connection timeout",
    product_name="Cloud Platform",
    severity="High",
    customer_type="Enterprise"
)

print(f"Recommended resolution: {prediction}")
```

### API Example
```python
import requests

# Predict resolution
response = requests.post('http://localhost:5000/predict', json={
    'error_description': 'API authentication failed',
    'product_name': 'API Gateway',
    'severity': 'Critical'
})

result = response.json()
print(f"Predicted: {result['predicted_resolution']}")
```

## 🎯 Use Cases

1. **Automated Support Routing**: Route cases to appropriate resolution paths
2. **Resolution Recommendations**: Suggest optimal solutions for new cases
3. **Performance Optimization**: Identify and improve underperforming areas
4. **Resource Allocation**: Optimize support team allocation based on case complexity
5. **Quality Assurance**: Monitor and improve resolution quality

## 🔮 Future Enhancements

- **Multi-agent RL**: Multiple agents for different support tiers
- **Real-time Learning**: Continuous learning from new cases
- **Natural Language Processing**: Advanced text understanding
- **Integration APIs**: Connect with existing support systems
- **A/B Testing**: Compare different resolution strategies

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This system is designed for educational and demonstration purposes. For production use, additional testing, validation, and security measures should be implemented. 