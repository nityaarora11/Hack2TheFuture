import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = 'customer_support_data.csv'
TOP_K = 3

def suggest_solutions(user_issue, csv_path=CSV_PATH, top_k=TOP_K):
    # Load historical data
    df = pd.read_csv(csv_path)
    # Vectorize error descriptions
    tfidf = TfidfVectorizer().fit(df['error_description'])
    tfidf_matrix = tfidf.transform(df['error_description'])
    # Vectorize the user's issue
    query_vec = tfidf.transform([user_issue])
    # Compute similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df['similarity'] = similarities
    # Get top K most similar past issues
    top_matches = df.sort_values('similarity', ascending=False).head(top_k)
    return top_matches

def main():
    print("=== Customer Support Solution Suggester ===")
    user_issue = input("Describe the issue you are facing: ").strip()
    while not user_issue:
        user_issue = input("Issue description is required. Please describe the issue: ").strip()

    suggestions = suggest_solutions(user_issue)
    print("\nTop suggested solutions based on similar past issues:")
    for i, row in enumerate(suggestions.itertuples(), 1):
        print(f"\nSuggestion {i}:")
        print(f"Past Issue: {row.error_description}")
        print(f"Product: {row.product_name}")
        print(f"Resolution Type: {row.resolution_type}")
        print(f"Resolution Details: {row.resolution_details}")
        print(f"Customer Feedback: {row.customer_feedback}")
        print(f"Similarity Score: {row.similarity:.2f}")

if __name__ == '__main__':
    main() 