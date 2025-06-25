import pandas as pd
import requests

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

def rephrase_text(text):
    if pd.isna(text) or not str(text).strip():
        return text
    prompt = f"Rephrase the following for clarity and professionalism:\n\n{text}"
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
        return result.get('text', text)
    else:
        return text

chunk_size = 100
input_file = "customer_support_data.csv"
output_file = "customer_support_data_rephrased.csv"

reader = pd.read_csv(input_file, chunksize=chunk_size)
first_chunk = True

for chunk in reader:
    for col in ['case comments', 'subject', 'description']:
        if col in chunk.columns:
            chunk[col] = chunk[col].apply(rephrase_text)
    chunk.to_csv(output_file, mode='a', index=False, header=first_chunk)
    first_chunk = False 