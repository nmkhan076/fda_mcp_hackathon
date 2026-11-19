
import asyncio
import json

from fastmcp import Client

MCP_URL = "http://localhost:8000/mcp"


def print_result(result):
    """Display MCP tool responses in a readable format."""
    if result.is_error:
        print("Tool returned an error:")

    if result.structured_content is not None:
        print(json.dumps(result.structured_content, indent=2, default=str))
    else:
        for item in result.content:
            if hasattr(item, "text"):
                try:
                    print(json.dumps(json.loads(item.text), indent=2))
                except (ValueError, TypeError):
                    print(item.text)
            else:
                print(item)


async def run_tool(client, tool_name, arguments):
    """Invoke one MCP tool and display its response."""
    print("\n" + "=" * 70)
    print(f"CALLING MCP TOOL: {tool_name}")
    print("=" * 70)

    print("\nInput arguments:")
    print(json.dumps(arguments, indent=2))

    try:
        result = await client.call_tool(tool_name, arguments)

        print("\nMCP Tool Response:")
        print_result(result)

        if result.is_error:
            print(f"\nSTATUS: TOOL ERROR - {tool_name}")
        else:
            print(f"\nSTATUS: SUCCESS - {tool_name}")

        return result

    except Exception as e:
        print(f"\nSTATUS: FAILED - {tool_name}")
        print(f"Error: {e}")
        return None


async def main():

    async with Client(MCP_URL) as client:

        print("\nConnected to FDA MCP Server")

        # --------------------------------------------------
        # 1. Discover registered MCP tools
        # --------------------------------------------------
        tools = await client.list_tools()

        print("\nAVAILABLE MCP TOOLS:")
        available = {tool.name for tool in tools}

        for tool in tools:
            print(f" - {tool.name}")

        # --------------------------------------------------
        # 2. Define the demonstration tool calls
        # --------------------------------------------------
        test_cases = [
            {
                "name": "get_drug_label",
                "arguments": {
                    "drug_name": "Oxycodone"
                }
            },
            {
                "name": "get_latest_four_opioid_labels",
                "arguments": {}
            },
            {
                "name": "search_opioid_label_updates",
                "arguments": {
                    "query": "respiratory depression",
                    "top_k": 3
                }
            }
        ]

        # --------------------------------------------------
        # 3. Execute the first three MCP tools
        # --------------------------------------------------
        for test in test_cases:
            if test["name"] not in available:
                print(f"\nTool not registered: {test['name']}")
                continue

            await run_tool(
                client,
                test["name"],
                test["arguments"]
            )

        # --------------------------------------------------
        # 4. Test CYP details for one label record
        # --------------------------------------------------
        cyp_tool = "get_drug_cyp_label_details"

        if cyp_tool in available:

            # Replace with an actual record_id returned by
            # your label-history retrieval tool.
            record_id = None

            if record_id is not None:
                await run_tool(
                    client,
                    cyp_tool,
                    {
                        "record_id": record_id,
                        "cyp_targets": ["CYP2D6", "CYP3A4"]
                    }
                )
            else:
                print("\n" + "=" * 70)
                print(f"SKIPPED: {cyp_tool}")
                print("Set record_id to a valid label record identifier.")
        else:
            print(f"\nTool not registered: {cyp_tool}")

        print("\nMCP TEST SESSION COMPLETED")


if __name__ == "__main__":
    asyncio.run(main())
