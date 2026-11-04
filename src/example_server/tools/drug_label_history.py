"""Historical openFDA drug labeling timeline filtered for CYP mentions."""
from datetime import datetime
import httpx
from fastmcp import FastMCP

URL = "https://api.fda.gov/drug/label.json"
SECTIONS = {
    "Clinical Pharmacology": "clinical_pharmacology",
    "Pharmacokinetics": "pharmacokinetics",
    "Drug Interactions": "drug_interactions",
    "Warnings and Precautions": "warnings_and_precautions",
    "Boxed Warning": "boxed_warning",
}

def formatted_date(value):
    try:
        return datetime.strptime(value[:8], "%Y%m%d").strftime("%B %d, %Y")
    except (ValueError, TypeError):
        return value or "Unknown"

async def fetch_all_label_records(
    search_query: str,
    client: httpx.AsyncClient,
) -> tuple[list[dict], int]:
    """Fetch matching openFDA labeling records using pagination."""

    records = []
    skip = 0
    limit = 1000
    total_matches = 0

    while True:
        response = await client.get(
            URL,
            params={
                "search": search_query,
                "limit": limit,
                "skip": skip,
            },
        )

        if response.status_code == 404:
            if skip == 0:
                return [], 0
            break

        response.raise_for_status()
        payload = response.json()

        batch = payload.get("results", [])
        total_matches = payload.get(
            "meta", {}
        ).get("results", {}).get("total", 0)

        if not batch:
            break

        records.extend(batch)
        skip += len(batch)

        if skip >= total_matches:
            break

        # Standard skip-based pagination boundary.
        if skip >= 26000:
            break

    # Deduplicate by unique labeling record ID.
    unique = {}

    for record in records:
        key = record.get("id") or (
            record.get("set_id"),
            str(record.get("version")),
            record.get("effective_time"),
        )
        unique[key] = record

    return list(unique.values()), total_matches


async def build_history(
    drug_name: str,
    cyp_targets: list[str],
) -> dict:
    """Retrieve matching drug labels and extract CYP evidence."""

    # 1. Validate inputs
    drug_name = drug_name.strip()

    targets = list(dict.fromkeys(
        t.strip().upper()
        for t in cyp_targets
        if t and t.strip()
    ))

    if not drug_name or not targets:
        return {
            "success": False,
            "error": "Provide drug name and CYP targets.",
        }

    if len(targets) > 20 or any(len(t) > 40 for t in targets):
        return {
            "success": False,
            "error": "Invalid CYP target list.",
        }

    safe_name = (
        drug_name.replace("\\", "\\\\")
        .replace('"', '\\"')
    )

    # 2. Search all labels matching the drug name
    # Match either generic name or brand name.
    search_query = (
        f'(openfda.generic_name:"{safe_name}"'
        f'+openfda.brand_name:"{safe_name}")'
    )

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            records, total_matches = await fetch_all_label_records(
                search_query,
                client,
            )

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "openFDA request timed out.",
        }

    except httpx.HTTPStatusError as exc:
        return {
            "success": False,
            "error": f"openFDA HTTP {exc.response.status_code}.",
        }

    except (httpx.RequestError, ValueError) as exc:
        return {
            "success": False,
            "error": f"Retrieval failed: {type(exc).__name__}.",
        }

    if not records:
        return {
            "success": False,
            "error": f"No labeling records found for {drug_name}.",
        }

    # 3. Sort by effective date and version
    def version_number(record):
        try:
            return int(record.get("version") or 0)
        except (ValueError, TypeError):
            return 0

    records.sort(
        key=lambda r: (
            r.get("effective_time") or "",
            version_number(r),
        ),
        reverse=True,
    )

    # Identify the latest available version per set_id.
    latest_by_set = {}

    for record in records:
        set_id = record.get("set_id")
        if set_id and set_id not in latest_by_set:
            latest_by_set[set_id] = record.get("id")

    # Create compact metadata records.
    revisions = []

    for record in records:
        openfda = record.get("openfda") or {}

        matched_enzymes = set()

        for title, field in SECTIONS.items():
            value = record.get(field) or []

            section_text = (
                "\n".join(map(str, value))
                if isinstance(value, list)
                else str(value)
            )

            for enzyme in targets:
                if enzyme.casefold() in section_text.casefold():
                    matched_enzymes.add(enzyme)

        effective_date = record.get("effective_time") or ""

        if len(effective_date) == 8 and effective_date.isdigit():
            date_readable = (
                f"{effective_date[:4]}-"
                f"{effective_date[4:6]}-"
                f"{effective_date[6:]}"
            )
        else:
            date_readable = effective_date

        revisions.append({
            "id": record.get("id"),
            "set_id": record.get("set_id"),
            "version": record.get("version"),
            "date": effective_date,
            "date_readable": date_readable,
            "brand_name": ", ".join(
                openfda.get("brand_name") or []
            ),
            "manufacturer": ", ".join(
                openfda.get("manufacturer_name") or []
            ),
            "has_cyp_matches": bool(matched_enzymes),
            "enzymes": sorted(matched_enzymes),
            "is_current": (
                record.get("id")
                == latest_by_set.get(record.get("set_id"))
            ),
        })

    # Return metadata only.
    return {
        "success": True,
        "drug_name": drug_name,
        "total_matches": total_matches,
        "total_revisions": len(revisions),
        "total_label_sets": len(latest_by_set),
        "revisions_with_cyp": sum(
            r["has_cyp_matches"] for r in revisions
        ),
        "revisions": revisions,
        "truncated": len(records) < total_matches,
    }




def register(mcp: FastMCP) -> None:
    @mcp.tool(name="get_drug_cyp_label_history", annotations={
        "title": "Historical CYP Drug Label Timeline", "readOnlyHint": True,
        "destructiveHint": False, "idempotentHint": True, "openWorldHint": True})
    async def get_drug_cyp_label_history(drug_name: str, cyp_targets: list[str]) -> dict:
        """Retrieve available label revisions and passages mentioning target CYP enzymes."""
        return await build_history(drug_name, cyp_targets)
