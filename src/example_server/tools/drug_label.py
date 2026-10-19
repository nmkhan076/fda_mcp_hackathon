"""Retrieve FDA drug labeling information from openFDA."""

from typing import Annotated

import httpx
from fastmcp import FastMCP
from pydantic import Field

OPENFDA_URL = "https://api.fda.gov/drug/label.json"


def register(mcp: FastMCP) -> None:
    """Register the openFDA drug-label tool."""

    @mcp.tool(
        name="get_drug_label",
        annotations={
            "title": "Search openFDA Drug Labels",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    )
    async def get_drug_label(
        drug_name: Annotated[
            str,
            Field(description="Drug brand name, such as Narcan"),
        ],
    ) -> dict:
        """Retrieve labeling information by drug brand name."""

        drug_name = drug_name.strip()

        if not drug_name:
            return {
                "success": False,
                "error": "Please provide a drug name.",
            }

        # Escape characters that could alter the search syntax.
        safe_name = drug_name.replace("\\", "\\\\").replace('"', '\\"')

        params = {
            "search": f'openfda.brand_name:"{safe_name}"',
            "limit": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    OPENFDA_URL,
                    params=params,
                )

                if response.status_code == 404:
                    return {
                        "success": False,
                        "error": "No matching FDA drug label found.",
                    }

                response.raise_for_status()
                payload = response.json()

            results = payload.get("results", [])

            if not results:
                return {
                    "success": False,
                    "error": "No matching FDA drug label found.",
                }

            label = results[0]
            openfda = label.get("openfda", {})

            return {
                "success": True,
                "brand_name": openfda.get("brand_name", []),
                "generic_name": openfda.get("generic_name", []),
                "manufacturer": openfda.get("manufacturer_name", []),
                "boxed_warning": label.get("boxed_warning", []),
                "warnings": label.get("warnings", []),
                "adverse_reactions": label.get("adverse_reactions", []),
                "set_id": label.get("set_id"),
                "effective_time": label.get("effective_time"),
                "source": str(response.url),
            }

        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "The openFDA request timed out.",
            }

        except httpx.HTTPStatusError as exc:
            return {
                "success": False,
                "error": (f"openFDA returned HTTP {exc.response.status_code}."),
            }

        except httpx.RequestError as exc:
            return {
                "success": False,
                "error": (f"Unable to connect to openFDA: {type(exc).__name__}"),
            }

        except ValueError:
            return {
                "success": False,
                "error": "Invalid JSON returned by openFDA.",
            }
