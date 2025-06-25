import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from customer_support_rl import CustomerSupportRL

CSV_PATH = 'customer_support_data.csv'
TOP_K_RETRIEVE = 5
TOP_K_PRESENT = 3

# Step 1: Retrieve similar cases using TF-IDF

def retrieve_similar_cases(user_issue, csv_path=CSV_PATH, top_k=TOP_K_RETRIEVE):
    df = pd.read_csv(csv_path)
    tfidf = TfidfVectorizer().fit(df['error_description'])
    tfidf_matrix = tfidf.transform(df['error_description'])
    query_vec = tfidf.transform([user_issue])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df['similarity'] = similarities
    top_indices = similarities.argsort()[::-1][:top_k]
    return df.iloc[top_indices]

# Step 2: Prompt for additional context if needed
def prompt_for_missing_fields(context):
    required_fields = [
        ('product_name', 'Please enter the product name: '),
        ('severity', 'Please enter the severity (Critical, High, Medium, Low): '),
        ('customer_type', 'Please enter the customer type or details: ')
    ]
    for key, prompt in required_fields:
        while not context.get(key):
            value = input(prompt)
            if value.strip():
                context[key] = value.strip()
            else:
                print(f"{key.replace('_', ' ').capitalize()} is required.")
    return context

# Step 3: RL ranking and feedback collection
def main():
    print("=== Customer Support Hybrid Retrieval + RL Suggester ===")
    user_issue = input("Describe the issue you are facing: ").strip()
    while not user_issue:
        user_issue = input("Issue description is required. Please describe the issue: ").strip()

    # Retrieve similar cases
    candidates = retrieve_similar_cases(user_issue, top_k=TOP_K_RETRIEVE)

    # Prompt for additional context
    context = {
        'error_description': user_issue,
        'product_name': '',
        'severity': '',
        'customer_type': ''
    }
    context = prompt_for_missing_fields(context)

    # Load and (quickly) train RL model
    rl_system = CustomerSupportRL(CSV_PATH)
    rl_system.train_model(total_timesteps=2000)

    # RL rank the candidates
    ranked = []
    for _, row in candidates.iterrows():
        # Use RL model to get a score for this candidate
        top_k = rl_system.predict_top_k_resolutions(
            error_description=context['error_description'],
            product_name=context['product_name'],
            severity=context['severity'],
            customer_type=context['customer_type'],
            k=len(rl_system.env.resolution_types)
        )
        # Find the probability for this candidate's resolution_type
        prob = 0.0
        for res_type, p in top_k:
            if res_type == row['resolution_type']:
                prob = p
                break
        ranked.append((row, prob))
    # Sort by RL score
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    top3 = ranked[:TOP_K_PRESENT]

    # Present suggestions and collect feedback
    def add_feedback_to_csv(issue, row, feedback, csv_path=CSV_PATH):
        new_row = {
            'error_description': issue['error_description'],
            'product_name': issue['product_name'],
            'customer_details': issue.get('customer_type', ''),
            'severity': issue['severity'],
            'resolution_type': row['resolution_type'],
            'resolution_details': row['resolution_details'],
            'customer_feedback': feedback
        }
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"Appended feedback for suggestion '{row['resolution_type']}' to CSV.")

    print("\nTop suggestions for your issue:")
    for i, (row, prob) in enumerate(top3, 1):
        print(f"\nSuggestion {i}: {row['resolution_type']}")
        print(f"Details: {row['resolution_details']}")
        print(f"Similarity: {row['similarity']:.2f}, RL Score: {prob:.2f}")
        feedback = input("Please rate this suggestion (e.g., 'Excellent - 5 stars', 'Not helpful', etc.): ")
        add_feedback_to_csv(context, row, feedback)

    print("\nThank you for your feedback! The model will use it to improve future suggestions.")

if __name__ == '__main__':
    main() 