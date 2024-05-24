import os
from openai import AzureOpenAI

# Read environment variables
API_KEY = os.getenv('API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')

client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_ENDPOINT
)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Which is the best LLM?"}
    ]
)

#print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)
