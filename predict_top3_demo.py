from customer_support_rl import CustomerSupportRL

# Initialize the RL system
rl_system = CustomerSupportRL('customer_support_data.csv')

# Train the model (for demo, use a small number of timesteps)
print("Training the RL model...")
rl_system.train_model(total_timesteps=2000)

# Define a new customer issue
new_issue = {
    'error_description': 'I want to solve my issue',
    'product_name': 'Sales Cloud',
    'severity': 'NA',
    'customer_type': 'NA'
}

# Predict the top 3 best solutions
print("\nPredicting top 3 best solutions for the new issue:")
top3 = rl_system.predict_top_k_resolutions(
    error_description=new_issue['error_description'],
    product_name=new_issue['product_name'],
    severity=new_issue['severity'],
    customer_type=new_issue['customer_type'],
    k=3
)

for i, (resolution, prob) in enumerate(top3, 1):
    print(f"{i}. {resolution} (probability: {prob:.2f})") 