import httpx
import asyncio

async def test_drugs():
    url = "https://api.fda.gov/drug/label.json"

    queries = [
        'openfda.brand_name:"Subutex"',
        #'openfda.generic_name:"tramadol hydrochloride"',
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        for query in queries:
            response = await client.get(
                url,
                params={"search": query, "limit": 5},
            )

            print("\nQuery:", query)
            print("Status:", response.status_code)

            if response.status_code == 200:
                for record in response.json().get("results", []):
                    print({
                        "brand": record.get("openfda", {}).get("brand_name"),
                        "set_id": record.get("set_id"),
                        "version": record.get("version"),
                    })

asyncio.run(test_drugs())