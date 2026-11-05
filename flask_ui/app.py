import asyncio
import json
import os
import traceback

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# from databricks.sdk import WorkspaceClient
# from databricks_mcp import DatabricksOAuthClientProvider
app = Flask(__name__)

port = os.environ.get("DATABRICKS_APP_PORT", "8000")

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")

print("Flask MCP URL:", MCP_URL, flush=True)


async def fetch_drug(drug_name):
    async with streamable_http_client(MCP_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("get_drug_label", {"drug_name": drug_name})

            if result.is_error:
                raise RuntimeError(str(result.content))

            # Your MCP server returns TextContent
            if result.structured_content is not None:
                data = result.structured_content
            else:
                text_blocks = [
                    block.text for block in result.content if block.type == "text"
                ]

                if not text_blocks:
                    raise ValueError("MCP returned no text response")

                data = json.loads(text_blocks[0])

            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object from MCP")

            return data


async def search_fda_evidence(query, source, top_k=3):
    """Call an FDA RAG retrieval tool through MCP."""

    tool_names = {
        "pdf": "search_opioid_label_updates",
        "web": "search_opioid_medications",
    }

    tool_name = tool_names.get(source)

    if not tool_name:
        raise ValueError("Invalid FDA source selected.")

    async with streamable_http_client(MCP_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                tool_name, {"query": query, "top_k": top_k}
            )

            if result.is_error:
                raise RuntimeError(str(result.content))

            if result.structured_content is not None:
                data = result.structured_content
            else:
                text_blocks = [
                    block.text for block in result.content if block.type == "text"
                ]

                if not text_blocks:
                    raise ValueError("MCP returned no text response.")

                data = json.loads(text_blocks[0])

            if not isinstance(data, dict):
                raise ValueError("Unexpected MCP response format.")

            if not data.get("success", False):
                raise RuntimeError(data.get("error", "FDA retrieval failed."))

            return data

async def fetch_cyp_history(drug_name, enzymes):
    async with streamable_http_client(MCP_URL) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(
            read_stream, write_stream
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                "get_drug_cyp_label_history",
                {
                    "drug_name": drug_name,
                    "cyp_targets": enzymes,
                },
            )

            if result.is_error:
                raise RuntimeError(str(result.content))

            if result.structured_content is not None:
                data = result.structured_content
            else:
                blocks = [
                    block.text
                    for block in result.content
                    if block.type == "text"
                ]
                data = json.loads(blocks[0]) if blocks else {}

            if not isinstance(data, dict):
                raise ValueError("Unexpected MCP response")

            return data
@app.route("/", methods=["GET", "POST"])
def home():
    drug_data = None
    rag_data = None
    error = None
    cyp_data = None
    cyp_enzymes = "CYP2D6, CYP3A4"
    drug_name = ""
    query = ""
    source = "pdf"

    if request.method == "POST":
        search_type = request.form.get("search_type", "drug")

        if search_type == "drug":
            drug_name = request.form.get("drug_name", "").strip()

            if not drug_name:
                error = "Please enter a drug name."
            else:
                try:
                    drug_data = asyncio.run(fetch_drug(drug_name))

                    if not drug_data.get("success", False):
                        error = drug_data.get("error", "Drug information unavailable.")
                        drug_data = None

                except Exception:
                    traceback.print_exc()
                    error = "Unable to retrieve drug information."

        elif search_type == "rag":
            query = request.form.get("query", "").strip()
            source = request.form.get("source", "pdf")

            if not query:
                error = "Please enter a question."
            else:
                try:
                    rag_data = asyncio.run(
                        search_fda_evidence(query=query, source=source, top_k=3)
                    )

                except Exception:
                    traceback.print_exc()
                    error = "Unable to retrieve FDA evidence."
        elif search_type == "cyp":

            cyp_drug = request.form.get(
                "cyp_drug", ""
            ).strip()

            cyp_enzymes = request.form.get(
            ).strip()

            enzymes = [
                e.strip()
                for e in cyp_enzymes.split(",")
                if e.strip()
            ]

            if not cyp_drug or not enzymes:
                error = "Enter a drug name and CYP enzymes."

            else:
                try:
                    cyp_data = asyncio.run(
                        fetch_cyp_history(cyp_drug, enzymes)
                    )

                    if not cyp_data.get("success"):
                        error = cyp_data.get(
                            "error", "Unable to retrieve CYP history."
                        )
                        cyp_data = None

                except Exception:
                    traceback.print_exc()
                    error = "CYP history retrieval failed."
    return render_template(
        "index.html",
        drug_data=drug_data,
        rag_data=rag_data,
        drug_name=drug_name,
        query=query,
        source=source,
        error=error,
        cyp_data=cyp_data,
        cyp_drug=cyp_drug,
        cyp_enzymes=cyp_enzymes,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
