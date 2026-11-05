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
        drug_search_result = await client.call_tool(
            "get_drug_label",
            {
                "drug_name": "Oxycodone",
            },
        )

        print("\nOPenFDA search result:")
        print(drug_search_result)
        # Call the FDA PDF search tool
        pdf_search_result = await client.call_tool(
            "search_opioid_label_updates",
            {
                "query": "respiratory depression",
                "top_k": 3,
            },
        )

        print("\nFDA search result:")
        print(pdf_search_result)


if __name__ == "__main__":
    asyncio.run(main())
