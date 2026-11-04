# FDA MCP Server — GSA Hackathon Evaluation

This evaluation follows the provided sample's five sections: objectives, methodology, metrics, security, and lessons learned. It is **implemented but not yet executed** against a live Databricks model and MCP server. Do not present sample CDC scores as FDA scores.

## 1. Prerequisites

Start the existing FDA MCP server locally or use your authorized deployed `/mcp` URL. Ensure the selected Databricks Model Serving endpoint supports OpenAI-compatible chat completions and function/tool calling. The judge endpoint must support chat completions. Your Databricks access token must have serving-endpoint permissions. A deployed Databricks App may require separate OAuth authentication: this basic runner uses unauthenticated MCP transport and is intended for local testing or a separately authenticated endpoint; it does **not** bypass Databricks App access controls. For protected Apps, use a supported authenticated MCP client/service-principal flow before running remotely.

From the repository root (Windows PowerShell):

```powershell
uv sync --group dev
uv pip install -r eval/phoenix/requirements-eval.txt
$env:DATABRICKS_HOST = "https://YOUR-WORKSPACE.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "YOUR-TOKEN"  # prefer a secure secret store
$env:DATABRICKS_MODEL_ENDPOINT = "YOUR-MODEL-SERVING-ENDPOINT"
$env:DATABRICKS_JUDGE_ENDPOINT = "YOUR-JUDGE-ENDPOINT" # optional, defaults to agent endpoint
$env:MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
```

In another terminal, start the combined application using the project's existing documented command, and verify its `/mcp` endpoint is reachable. Never commit real credentials or an `.env` file.

## 2. Run

```powershell
uv run python eval/phoenix/create_dataset.py
uv run python eval/phoenix/run_experiment.py --dataset-name fda-eval-v01 --system-prompt-version v01 --limit 2 --repeats 1
uv run python eval/phoenix/run_experiment.py --dataset-name fda-eval-v01 --system-prompt-version v01 --repeats 5
```

For optional local Phoenix tracing, install `arize-phoenix`, `arize-phoenix-otel`, and `openinference-instrumentation-openai`, run the Phoenix server separately, then add `--phoenix`. The optional dataset upload is `uv run python eval/phoenix/create_dataset.py --upload-phoenix`; requires Phoenix configured and the compatible Phoenix client API. The local CSV dataset is authoritative for the runner; dataset upload is separate.

## 3. What gets measured

* **Correctness**: an LLM judge scores whether each answer follows the expected behavior (0/1).
* **Scope adherence**: an LLM judge scores appropriate answer/refusal (0/1).
* **Tool selection**: expected MCP tool appeared in calls; out-of-scope cases should have no tool calls (0/1).
* **Latency and tool count**: per case, per trial.
* **MCP lift**: correctness percentage points over the no-tool baseline.
* **Repeatability**: five trials per question by default. Inspect per-question variance in the CSV.

The 15 cases include label retrieval, CYP history, FDA PDF retrieval, missing metadata, out-of-domain requests, medical advice, and prompt-injection attempts. These are **behavioral expectations**, not verified fixed FDA ground-truth answers. Before formal submission, a domain reviewer should verify current FDA source excerpts and convert relevant cases to frozen evidence-backed expected facts. LLM-as-judge scores are not independent factual validation.

## 4. Results and submission

Output: `eval/phoenix/results/fda_eval_<timestamp>.csv` and `_summary.json`. Report counts, correctness, scope adherence, MCP lift, tool-selection accuracy, mean latency, and representative failure traces. Judge errors are unscored, not counted as passes. For formal results, human-review every failed case and sample passing cases. The `evaluation_report_template.md` provides the five-section write-up. Do not claim completed evaluation before running it.

## 5. Security and limitations

The intended dataset uses public FDA content, but review all logs and retrieved passages for PII/CUI before sending data to models or Phoenix. Protect Databricks tokens and restrict access to local Phoenix traces. Prompt injection and out-of-scope coverage are included, but retrieved-document injection, multi-turn attacks, and true FDA historical-change ground truth remain gaps. A `get_drug_label` result does not provide original FDA approval dates; do not conflate `effective_time` with approval. A label-history sequence is not itself a verified change log. The current `search_opioid_label_updates` tool covers the PDF only, not arbitrary websites. CYP record-details retrieval requires a valid record ID obtained from history, so it is not included as an independent fixed-ID test.
