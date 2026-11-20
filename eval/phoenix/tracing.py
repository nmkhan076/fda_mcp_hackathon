import os

from opentelemetry import trace

def setup_tracing(enabled=False):
    if not enabled:
        return trace.get_tracer("fda-mcp-evaluation")

    from phoenix.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor

    provider = register(
        project_name="fda-mcp-eval",
        endpoint=os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://localhost:6006/v1/traces"
        ),
    )

    OpenAIInstrumentor().instrument(
        tracer_provider=provider
    )

    return trace.get_tracer("fda-mcp-evaluation")