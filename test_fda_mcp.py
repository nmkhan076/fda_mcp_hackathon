import asyncio

from fastmcp import Client


async def main():
    async with Client("http://localhost:8000/mcp") as client:
        # Discover available MCP tools
        tools = await client.list_tools()

        print("Available tools:")
        for tool in tools:
            print("-", tool.name)

        # Call the FDA PDF search tool
        result = await client.call_tool(
            "search_opioid_label_updates",
            {
                "query": "respiratory depression",
                "top_k": 3,
            },
        )

        print("\nFDA search result:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
