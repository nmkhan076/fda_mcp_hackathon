from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
print(ROOT)
load_dotenv(ROOT / ".env")

def model_client():
    return OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=(
        os.environ["DATABRICKS_HOST"].rstrip("/")
        + "/ai-gateway/mlflow/v1"
    ),
)
client = model_client()
response = client.chat.completions.create(
    model=os.environ["DATABRICKS_MODEL_ENDPOINT"],
    messages=[
        {"role": "user", "content": "Say hello."}
    ],
)
print(response.choices[0].message.content)