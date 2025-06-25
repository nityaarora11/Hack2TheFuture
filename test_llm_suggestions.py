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

def get_suggestions(user_issue):
    data = {
        "prompt": f"A Salesforce user reports: '{user_issue}'. List the top 3 possible solutions or troubleshooting steps.",
        "model": "llmgateway__BedrockAnthropicClaude35Sonnet"
    }
    response = requests.post(
        "https://bot-svc-llm.sfproxy.einstein.aws-dev4-uswest2.aws.sfdc.cl/v1.0/generations",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        print("Model response:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    user_issue = input("Describe the Salesforce issue: ")
    get_suggestions(user_issue) 