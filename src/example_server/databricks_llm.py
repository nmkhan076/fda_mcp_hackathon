"""Evidence-grounded Databricks Responses API helper shared by MCP tools."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[2] / '.env')


class LLMConfigurationError(RuntimeError):
    """Databricks credentials or endpoint are missing."""


async def grounded_completion(system_prompt: str, evidence: dict) -> dict:
    host = os.getenv('DATABRICKS_HOST', '').strip().rstrip('/')
    token = os.getenv('DATABRICKS_TOKEN', '').strip()
    model = os.getenv('DATABRICKS_MODEL_ENDPOINT', '').strip()
    missing = [k for k, v in {'DATABRICKS_HOST': host, 'DATABRICKS_TOKEN': token,
                               'DATABRICKS_MODEL_ENDPOINT': model}.items() if not v]
    if missing:
        raise LLMConfigurationError('Missing: ' + ', '.join(missing))
    if not host.startswith('https://'):
        raise LLMConfigurationError('DATABRICKS_HOST must start with https://')
    client = AsyncOpenAI(api_key=token, base_url=f'{host}/ai-gateway/mlflow/v1',
                         timeout=90.0, max_retries=1)
    response = await client.responses.create(
        model=model,
        max_output_tokens=2048,
        input=[
            {'role': 'system', 'content': system_prompt + '\nReturn only valid JSON.'},
            {'role': 'user', 'content': json.dumps(evidence, ensure_ascii=False)},
        ],
    )
    content = (response.output_text or '').strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    if not content:
        raise ValueError('Databricks returned no text')
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object from Databricks')
    return result
