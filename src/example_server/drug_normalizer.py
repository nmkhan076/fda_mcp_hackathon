"""Normalize openFDA labeling records and retrieve matching recent versions."""
import re
import httpx

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
LABEL_FIELDS = (
    "boxed_warning", "warnings", "warnings_and_precautions", "adverse_reactions",
    "drug_interactions", "clinical_pharmacology", "pharmacokinetics",
)
ER_PATTERN = re.compile(r"\b(?:extended[ -]release|controlled[ -]release|sustained[ -]release|long[ -]acting|ER|XR)\b", re.I)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _matches_search(name, search_term):
    """Match a whole ingredient/brand name or a leading generic ingredient token.

    E.g. tramadol matches TRAMADOL HYDROCHLORIDE, but not tramadol-like text.
    """
    candidate, target = _norm(name), _norm(search_term)
    return candidate == target or candidate.startswith(target + " ")


def classify_formulation(record):
    meta = record.get("openfda") or {}
    values = sum((_as_list(meta.get("brand_name")), _as_list(record.get("dosage_forms_and_strengths")),
                  _as_list(record.get("spl_product_data_elements"))), [])
    return "Extended Release (ER)" if ER_PATTERN.search(" ".join(values)) else "Unclassified / Other"


def _normalize_record(record, query):
    meta = record.get("openfda") or {}
    evidence = {field: "\n".join(_as_list(record.get(field))) for field in LABEL_FIELDS if record.get(field)}
    return {
        "record_id": record.get("id"), "set_id": record.get("set_id"),
        "version": record.get("version"), "effective_date": record.get("effective_time"),
        "brands": _as_list(meta.get("brand_name")), "ingredients": _as_list(meta.get("generic_name")),
        "manufacturer": _as_list(meta.get("manufacturer_name")),
        "formulation_type": classify_formulation(record), "evidence": evidence,
        "search_query": query, "source_provider": "openFDA",
    }


async def fetch_and_normalize_drug(search_term: str, limit: int = 1000,
                                   exclude_er: bool = False,
                                   client: httpx.AsyncClient | None = None):
    """Find matching brand OR generic labels, sort by effective date, dedupe by set_id.

    Searches exact fields first, then analyzed fields. A result must actually contain
    the searched brand or generic name (avoids unrelated partial matches).
    """
    search_term = search_term.strip()
    if not search_term:
        raise ValueError("Drug name is required")
    safe = search_term.replace("\\", "\\\\").replace('"', '\\"')
    queries = [
        f'openfda.brand_name.exact:"{safe}"',
        f'openfda.generic_name.exact:"{safe}"',
        f'openfda.brand_name:"{safe}"',
        f'openfda.generic_name:"{safe}"',
    ]
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30)
    found = {}
    try:
        for query in queries:
            # Request multiple results; API order is not guaranteed to be newest-first.
            response = await client.get(OPENFDA_URL, params={"search": query, "limit": min(limit, 1000)})
            if response.status_code == 404:
                continue
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError(f"openFDA JSON decoding failed for {response.url}: {exc}") from exc
            for record in payload.get("results", []):
                meta = record.get("openfda") or {}
                names = _as_list(meta.get("brand_name")) + _as_list(meta.get("generic_name"))
                if not any(_matches_search(name, search_term) for name in names):
                    continue
                item = _normalize_record(record, query)
                if exclude_er and item["formulation_type"] == "Extended Release (ER)":
                    continue
                key = item["set_id"] or item["record_id"] or repr(item)
                previous = found.get(key)
                if previous is None or (str(item["effective_date"] or ""), int(item["version"] or 0)) > (str(previous["effective_date"] or ""), int(previous["version"] or 0)):
                    found[key] = item
            # Continue across brand and generic queries: newer records may be
            # returned under a different field, even after an earlier hit.
    finally:
        if own_client:
            await client.aclose()
    return sorted(found.values(), key=lambda r: (str(r["effective_date"] or ""), int(r["version"] or 0)), reverse=True)
