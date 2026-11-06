
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"
print(env_path.exists())
load_dotenv(env_path)

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=(
        os.environ["DATABRICKS_HOST"].rstrip("/")
        + "/ai-gateway/mlflow/v1"
    ),
)

response = client.responses.create(
  model="models.ai.gpt-5-6-terra",
  max_output_tokens=256,
  input=[
    {
      "role": "user",
      "content": [{"type": "input_text", "text": "Hello!"}]
    },
    {
      "role": "assistant",
      "content": [{"type": "output_text", "text": "Hello! How can I assist you today?"}]
    },
    {
      "role": "user",
      "content": [{"type": "input_text", "text": "What is Databricks?"}]
    }
  ]
)

print(response.output_text)