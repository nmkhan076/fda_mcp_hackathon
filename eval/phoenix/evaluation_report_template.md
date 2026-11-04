# FDA MCP Server — Evaluation Report (complete after executing)

**Server:** FDA MCP Server | **Framework:** Databricks model + optional Arize Phoenix tracing | **Dataset:** fda-eval-v01 (15 items) | **Agent:** [endpoint] | **Judge:** [endpoint] | **Date:** [date]

## 1. Overview and Objectives
Evaluate whether the MCP agent retrieves FDA label and CYP history evidence accurately, selects the right tools, and stays within its public-information remit. Compare against a no-tool baseline.

## 2. Testing Methodology
Versioned prompt `prompts/system_prompt_v01.txt`, golden dataset `golden_fda_eval.csv`, five repeated trials per case, Databricks chat-completions tool-calling agent, separate LLM correctness/scope judge, optional Phoenix OpenInference tracing. Record model, endpoint, date, run settings, and any human-reviewed ground-truth evidence.

## 3. Performance Metrics
| Metric | No-tool baseline | MCP agent |
|---|---|---|
| Correctness (scored / total) | TBD | TBD |
| Scope adherence | TBD | TBD |
| Tool-selection rate | N/A | TBD |
| Mean latency (seconds) | TBD | TBD |
| Mean tool calls | 0 | TBD |

**MCP correctness lift:** TBD percentage points. **Repeated-trial consistency:** TBD. **Failure analysis:** [case, cause, tool trace, remediation]. Do not use the CDC sample's results.

## 4. Security Considerations
Public FDA sources only unless separately approved; no PII/CUI in evaluation traces; tokens from environment/secret store; out-of-scope and prompt-injection cases included; inspect tool-output prompt injection and authenticated Databricks deployment in future tests.

## 5. Lessons Learned & Recommendations
[What worked] [Failure patterns] [Judge/human-review discrepancies] [Next dataset and security improvements].
