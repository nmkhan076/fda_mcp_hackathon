
import re
import httpx

OPENFDA_URL = "https://api.fda.gov/drug/label.json"

LABEL_FIELDS = (
    "boxed_warning",
    "warnings",
    "warnings_and_precautions",
    "adverse_reactions",
    "drug_interactions",
    "clinical_pharmacology",
    "pharmacokinetics",
)

ER_PATTERN = re.compile(
    r"\b(?:extended.release|controlled.release|"
    r"sustained.release|prolonged.release|"
    r"long.acting|ER|XR|SR|CR)\b",
    re.IGNORECASE,
)


def classify_formulation(record):
    meta = record.get("openfda", {})
    names = meta.get("brand_name", [])

    text = " ".join(
        names
        + record.get("description", [])
        + record.get("dosage_forms_and_strengths", [])
        + record.get("spl_product_data_elements", [])
    )

    if ER_PATTERN.search(text):
        return "Extended Release (ER)"

    # Absence of ER wording does not prove IR.
    return "Unclassified / Other"



async def fetch_and_normalize_drug(
    search_term: str,
    limit: int = 100,
    exclude_er: bool = False,
):
    search_term = search_term.strip()

    if not search_term:
        raise ValueError("Drug name is required")

    safe_term = (
        search_term
        .replace("\\", "\\\\")
        .replace('"', '\\"')
    )

    queries = [
        f'openfda.brand_name.exact:"{safe_term}"',
        f'openfda.brand_name:"{safe_term}"',
        f'openfda.generic_name:"{safe_term}"',
    ]

    records = []
    last_query = None

    async with httpx.AsyncClient(timeout=30) as client:
        for query in queries:
            response = await client.get(
                OPENFDA_URL,
                params={
                    "search": query,
                    "limit": min(limit, 1000),
                },
            )

            if response.status_code == 404:
                continue

            response.raise_for_status()

            candidates = response.json().get("results", [])

            if candidates:
                records = candidates
                last_query = query
                break

    normalized = []

    for record in records:
        meta = record.get("openfda", {})
        formulation = classify_formulation(record)

        if exclude_er and formulation == "Extended Release (ER)":
            continue

        evidence = {
            field: "\n".join(record.get(field, []))
            for field in LABEL_FIELDS
            if record.get(field)
        }

        normalized.append({
            "record_id": record.get("id"),
            "set_id": record.get("set_id"),
            "version": record.get("version"),
            "effective_date": record.get("effective_time"),
            "brands": meta.get("brand_name", []),
            "ingredients": meta.get("generic_name", []),
            "manufacturer": meta.get("manufacturer_name", []),
            "formulation_type": formulation,
            "evidence": evidence,
            "search_query": last_query,
        })

    return normalized
