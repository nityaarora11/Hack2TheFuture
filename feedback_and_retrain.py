import pandas as pd
from customer_support_rl import CustomerSupportRL

CSV_PATH = 'customer_support_data.csv'

# Example: The issue and top 3 solutions (normally, you'd get these from your prediction script)
issue = {
    'error_description': '',  # Intentionally left blank for demo
    'product_name': '',       # Intentionally left blank for demo
    'severity': 'High',
    'customer_type': ''       # Intentionally left blank for demo
}

def prompt_for_missing_fields(issue):
    required_fields = [
        ('error_description', 'Please describe the issue the customer is facing: '),
        ('product_name', 'Please enter the product name: '),
        ('severity', 'Please enter the severity (Critical, High, Medium, Low): '),
        ('customer_type', 'Please enter the customer type or details: ')
    ]
    for key, prompt in required_fields:
        while not issue.get(key):
            value = input(prompt)
            if value.strip():
                issue[key] = value.strip()
            else:
                print(f"{key.replace('_', ' ').capitalize()} is required.")
    return issue

def add_feedback_to_csv(issue, solution, resolution_details, feedback, csv_path=CSV_PATH):
    new_row = {
        'error_description': issue['error_description'],
        'product_name': issue['product_name'],
        'customer_details': issue.get('customer_type', ''),
        'severity': issue['severity'],
        'resolution_type': solution,
        'resolution_details': resolution_details,
        'customer_feedback': feedback
    }
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Appended feedback for solution '{solution}' to CSV.")

# Prompt for missing issue details
issue = prompt_for_missing_fields(issue)

# Load or train the RL system
rl_system = CustomerSupportRL(CSV_PATH)
rl_system.train_model(total_timesteps=2000)  # Or load a pre-trained model

# Predict top 3 solutions for the issue
top3 = rl_system.predict_top_k_resolutions(
    error_description=issue['error_description'],
    product_name=issue['product_name'],
    severity=issue['severity'],
    customer_type=issue['customer_type'],
    k=3
)

# Each item in top3 is (resolution_type, probability)
top3_solutions = [(res, '') for res, prob in top3]  # You can fill in details if available

# Prompt user for feedback for each of the top 3 solutions
user_feedbacks = []
for i, (solution, details) in enumerate(top3_solutions, 1):
    print(f"\nSolution {i}: {solution}")
    print(f"Details: {details}")
    feedback = input("Please rate this solution (e.g., 'Excellent - 5 stars', 'Good - 4 stars', etc.): ")
    user_feedbacks.append(feedback)

# Append feedback for each of the top 3 solutions
for (solution, details), feedback in zip(top3_solutions, user_feedbacks):
    add_feedback_to_csv(issue, solution, details, feedback)

# Retrain the RL model
print("\nRetraining the RL model with new feedback...")
rl_system.train_model(total_timesteps=3000)
print("Retraining complete!") 