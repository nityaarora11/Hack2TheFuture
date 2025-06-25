import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# 1. Load your historical data
df = pd.read_csv('customer_support_data.csv', encoding='latin1')

# 2. Create embeddings for all historical issues
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use any suitable model
issue_texts = df['issue_description'].tolist()
embeddings = model.encode(issue_texts, show_progress_bar=True)
embeddings = np.array(embeddings, dtype='float32')  # Ensure correct dtype and shape
if embeddings.ndim == 1:
    embeddings = np.expand_dims(embeddings, axis=0)

# 3. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def find_top_k_similar(query, k=3):
    query_vec = model.encode([query])
    query_vec = np.array(query_vec, dtype='float32')
    if query_vec.ndim == 1:
        query_vec = np.expand_dims(query_vec, axis=0)
    D, I = index.search(query_vec, k)
    return I[0]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def llm_context_match(user_query, candidate_issue):
    API_KEY = "651192c5-37ff-440a-b930-7444c69f4422"
    headers = {
        "X-LLM-Provider": "Bedrock",
        "Authorization": f"API_KEY {API_KEY}",
        "hawking-tenant-id": "autnt-8de5c6a7-14ed-4c35-ac9c-856dbd917508",
        "x-client-feature-id": "api-key-exploratory",
        "x-caller-service": "prediction-services.graph-execution-service",
        "x-sfdc-tenant-id": "00DSB000007cALh",
        "x-sfdc-core-tenant-id": "core/falcontest1-core4sdb3/00DSB000007cALh2AM",
        "x-sfdc-app-context": "EinsteinGPT",
        "Content-Type": "application/json"
    }
    prompt = f"""User query: "{user_query}"
Candidate issue: "{candidate_issue}"
Does the candidate issue match the context/meaning of the user query? Answer yes or no."""
    data = {
        "prompt": prompt,
        "model": "llmgateway__BedrockAnthropicClaude35Sonnet"
    }
    response = requests.post(
        "https://bot-svc-llm.sfproxy.einstein.aws-dev4-uswest2.aws.sfdc.cl/v1.0/generations",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        result = response.json()
        return 'yes' in result.get('text', '').lower()
    else:
        return False

# 4. Function to get LLM suggestions
def get_llm_suggestions(user_issue, top_cases=None):
    API_KEY = "651192c5-37ff-440a-b930-7444c69f4422"
    headers = {
        "X-LLM-Provider": "Bedrock",
        "Authorization": f"API_KEY {API_KEY}",
        "hawking-tenant-id": "autnt-8de5c6a7-14ed-4c35-ac9c-856dbd917508",
        "x-client-feature-id": "api-key-exploratory",
        "x-caller-service": "prediction-services.graph-execution-service",
        "x-sfdc-tenant-id": "00DSB000007cALh",
        "x-sfdc-core-tenant-id": "core/falcontest1-core4sdb3/00DSB000007cALh2AM",
        "x-sfdc-app-context": "EinsteinGPT",
        "Content-Type": "application/json"
    }
    if top_cases:
        prompt = f"A Salesforce user reports: '{user_issue}'.\nHere are 3 similar past cases and their resolutions:\n"
        for i, case in enumerate(top_cases, 1):
            prompt += f"\nCase {i}:\nIssue: {case['issue_description']}\nResolution: {case['resolution']}\n"
        prompt += "\nBased on these, list the top 3 possible solutions or troubleshooting steps for the new issue."
    else:
        prompt = f"A Salesforce user reports: '{user_issue}'. List the top 3 possible solutions or troubleshooting steps."

    data = {
        "prompt": prompt,
        "model": "llmgateway__BedrockAnthropicClaude35Sonnet"
    }
    response = requests.post(
        "https://bot-svc-llm.sfproxy.einstein.aws-dev4-uswest2.aws.sfdc.cl/v1.0/generations",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        print("\nModel response:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    user_issue = input("Describe the Salesforce issue: ")
    full_issue = user_issue
    while True:
        user_vec = model.encode([full_issue])[0]
        top_indices = find_top_k_similar(full_issue, k=3)
        top_cases = []
        print("\nTop 3 similar past cases:")
        for i, idx in enumerate(top_indices):
            case = df.iloc[idx]
            print(f"\nCase {i+1}:")
            print("Issue:", case['issue_description'])
            print("Resolution:", case['resolution'])
            top_cases.append({'issue_description': case['issue_description'], 'resolution': case['resolution']})
        print("\nProceeding to LLM for best suggestions based on the top 3 similar cases...\n")
        get_llm_suggestions(full_issue, top_cases)
        break 