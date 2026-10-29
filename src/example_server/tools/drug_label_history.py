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

async def build_history(drug_name: str, cyp_targets: list[str]) -> dict:
    drug_name = drug_name.strip()
    targets = list(dict.fromkeys(t.strip().upper() for t in cyp_targets if t.strip()))
    if not drug_name or not targets:
        return {"success": False, "error": "Provide drug name and CYP targets."}
    if len(targets) > 20 or any(len(t) > 40 for t in targets):
        return {"success": False, "error": "Invalid enzyme list."}
    safe = drug_name.replace("\\", "\\\\").replace('"', '\\"')
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            first = await client.get(URL, params={"search": f'openfda.brand_name:"{safe}"', "limit": 1})
            if first.status_code == 404:
                return {"success": False, "error": "Drug not found."}
            first.raise_for_status()
            matches = first.json().get("results", [])
            set_id = matches[0].get("set_id") if matches else None
            if not set_id:
                return {"success": False, "error": "No set_id available."}
            records = []
            for skip in range(0, 1000, 100):
                res = await client.get(URL, params={"search": f'set_id:"{set_id}"', "limit": 100, "skip": skip})
                if res.status_code == 404:
                    break
                res.raise_for_status()
                batch = res.json().get("results", [])
                records.extend(batch)
                if len(batch) < 100:
                    break
    except httpx.TimeoutException:
        return {"success": False, "error": "openFDA timed out."}
    except httpx.HTTPStatusError as exc:
        return {"success": False, "error": f"openFDA HTTP {exc.response.status_code}."}
    except (httpx.RequestError, ValueError) as exc:
        return {"success": False, "error": f"openFDA retrieval failed: {type(exc).__name__}."}

    records.sort(key=lambda r: (r.get("effective_time") or "", str(r.get("version") or "")))
    revisions = []
    md = [f"# Historical CYP Label Timeline: {drug_name}", f"**Enzymes:** {', '.join(targets)}",
          f"**Revisions retrieved:** {len(records)}", "Source: openFDA (history may be incomplete)", "", "---"]
    for idx, record in enumerate(records, 1):
        date = record.get("effective_time", "")
        generic = record.get("openfda", {}).get("generic_name") or ["N/A"]
        sections = []
        md.extend(["", f"## Revision {idx}: {formatted_date(date)}",
                   f"- Effective date: {date}", f"- Version: {record.get('version', 'N/A')}",
                   f"- Generic name: {generic[0]}", ""])
        for title, key in SECTIONS.items():
            value = record.get(key) or []
            body = "\n".join(value) if isinstance(value, list) else str(value)
            hits = [t for t in targets if t.casefold() in body.casefold()]
            if hits:
                sections.append({"title": title, "enzymes": hits, "text": body})
                md.extend([f"### {title}", f"**Enzymes:** {', '.join(hits)}", "", body, ""])
        if not sections:
            md.append("*No target-enzyme mentions in selected sections.*")
        md.append("---")
        revisions.append({"date": date, "date_readable": formatted_date(date),
                          "version": record.get("version", "N/A"), "sections": sections})
    return {"success": True, "drug_name": drug_name, "set_id": set_id, "enzymes": targets,
            "total_revisions": len(records), "revisions": revisions,
            "markdown": "\n".join(md), "truncated": len(records) >= 1000}

def register(mcp: FastMCP) -> None:
    @mcp.tool(name="get_drug_cyp_label_history", annotations={
        "title": "Historical CYP Drug Label Timeline", "readOnlyHint": True,
        "destructiveHint": False, "idempotentHint": True, "openWorldHint": True})
    async def get_drug_cyp_label_history(drug_name: str, cyp_targets: list[str]) -> dict:
        """Retrieve available label revisions and passages mentioning target CYP enzymes."""
        return await build_history(drug_name, cyp_targets)
