"""DailyMed SPL fallback for exact brand searches missing from openFDA."""
from datetime import datetime
import re
import xml.etree.ElementTree as ET

import httpx

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
NS = "{urn:hl7-org:v3}"
SECTION_TITLES = {
    "boxed_warning": ("boxed warning",),
    "warnings": ("warnings and precautions", "warnings"),
    "adverse_reactions": ("adverse reactions",),
    "drug_interactions": ("drug interactions",),
    "clinical_pharmacology": ("clinical pharmacology",),
    "pharmacokinetics": ("pharmacokinetics",),
}


def _date(value):
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except (ValueError, TypeError):
            pass
    return ""


def _brand_matches(title, name):
    """Match the requested brand without confusing ULTRAM with ULTRAM ER."""
    def normalize(value):
        return re.sub(r"\s+", " ", re.sub(r"[®™]", "", str(value or "")).strip().casefold())

    title, name = normalize(title), normalize(name)
    if not title or not name:
        return False
    if not re.match(rf"^{re.escape(name)}(?=$|[\s(\-])", title):
        return False
    # A suffix after the requested brand denotes a different formulation.
    # Do not exclude ER when the user explicitly requested ULTRAM ER.
    suffix = title[len(name):].lstrip(" -(")
    if not re.search(r"\b(?:er|xr|extended[ -]release|extended release|long[ -]acting)\b", name):
        if re.match(r"^(?:er|xr|extended[ -]release|long[ -]acting)\b", suffix):
            return False
    return True


def _spl_header(xml_content):
    """Read the document-level effectiveTime and versionNumber, not section dates."""
    if b"<!DOCTYPE" in xml_content.upper() or b"<!ENTITY" in xml_content.upper():
        raise ValueError("Unexpected XML document declaration")
    root = ET.fromstring(xml_content)
    effective = root.find(NS + "effectiveTime")
    version = root.find(NS + "versionNumber")
    title = root.find(NS + "title")
    return {
        "effective_time": (effective.get("value", "") if effective is not None else "")[:8],
        "version": (version.get("value", "") if version is not None else ""),
        "title": " ".join(title.itertext()).strip() if title is not None else "",
    }


def _version(value):
    try:
        return int(value or 0)
    except (ValueError, TypeError):
        return 0


def _section_text(section):
    texts = []
    for element in section.iter():
        if element.tag in (NS + "text", NS + "paragraph", NS + "item"):
            if element.tag == NS + "text":
                text = " ".join(" ".join(element.itertext()).split())
                if text:
                    texts.append(text)
    return "\n".join(dict.fromkeys(texts))


def parse_spl(xml_content):
    if b"<!DOCTYPE" in xml_content.upper() or b"<!ENTITY" in xml_content.upper():
        raise ValueError("Unexpected XML document declaration")
    root = ET.fromstring(xml_content)
    sections = {}
    for section in root.iter(NS + "section"):
        title_el = section.find(NS + "title")
        title = " ".join(title_el.itertext()).strip().casefold() if title_el is not None else ""
        if not title:
            continue
        for field, aliases in SECTION_TITLES.items():
            if title in aliases and field not in sections:
                value = _section_text(section)
                if value:
                    sections[field] = [value[:40000]]
    return sections


def extract_spl_metadata(xml_content):
    """Extract generic name and manufacturer from DailyMed SPL XML."""

    root = ET.fromstring(xml_content)

    metadata = {
        "generic_name": [],
        "manufacturer_name": [],
    }

    # Extract generic ingredient names from product ingredient entries.
    ingredient_names = []

    for ingredient in root.iter(NS + "ingredient"):
        name_element = ingredient.find(
            ".//" + NS + "name"
        )

        if name_element is not None and name_element.text:
            name = name_element.text.strip()

            if name and name not in ingredient_names:
                ingredient_names.append(name)

    metadata["generic_name"] = ingredient_names

    # Extract organization names from document author information.
    for author in root.findall(".//" + NS + "author"):
        for org in author.findall(".//" + NS + "representedOrganization"):
            name_element = org.find(NS + "name")

            if name_element is not None and name_element.text:
                name = name_element.text.strip()

                if name not in metadata["manufacturer_name"]:
                    metadata["manufacturer_name"].append(name)

    return metadata


async def find_dailymed_brand(drug_name: str, client: httpx.AsyncClient):
    """Find newest exact-brand SPL using the actual XML effectiveTime."""
    candidates = {}
    page = 1
    while True:
        response = await client.get(
            f"{BASE}/spls.json",
            params={"drug_name": drug_name, "name_type": "b", "pagesize": 100, "page": page},
        )
        if response.status_code == 404:
            break
        response.raise_for_status()
        payload = response.json()
        columns = payload.get("COLUMNS", payload.get("columns", []))
        rows = payload.get("DATA", payload.get("data", []))
        if isinstance(rows, dict):
            rows = rows.get("spl", rows.get("spls", []))
        if columns and rows and isinstance(rows[0], list):
            items = [dict(zip(columns, row)) for row in rows]
        else:
            items = rows if isinstance(rows, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            record = {str(k).lower(): v for k, v in item.items()}
            set_id = record.get("setid") or record.get("set_id")
            if set_id and _brand_matches(record.get("title", ""), drug_name):
                candidates[str(set_id)] = record
        meta = payload.get("METADATA", payload.get("metadata", {})) or {}
        total_pages = _version(meta.get("total_pages", meta.get("TOTAL_PAGES", 1))) or 1
        if page >= total_pages:
            break
        page += 1


    if not candidates:
        return None

    # Compare XML metadata for every candidate. Search-list published_date
    # is not necessarily the label's effectiveTime.
    ranked = []
    for set_id, record in candidates.items():
        try:
            xml_response = await client.get(f"{BASE}/spls/{set_id}.xml", timeout=45)
            if xml_response.status_code == 404:
                print(f"DailyMed SPL XML not found: {set_id}")
                continue
            xml_response.raise_for_status()
            xml_content = xml_response.content
            header = _spl_header(xml_content)
            # The SPL document title need not equal the marketed brand name.
            # Candidate brand matching was already performed on the search results.
            effective = header["effective_time"] or _date(record.get("published_date"))
            version = _version(header["version"] or record.get("spl_version"))
            ranked.append((effective, version, set_id, record, xml_content))
        except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
            print(f"Skipping DailyMed SPL {set_id}: {exc}")

    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for effective, version, set_id, record, xml_content in ranked:
        sections = parse_spl(xml_content)
        if not sections:
            continue
        metadata = extract_spl_metadata(xml_content)
        return {
            "id": None,
            "set_id": set_id,
            "version": version,
            "effective_time": effective,
            "published_date": record.get("published_date"),
            "openfda": {
                "brand_name": [drug_name],
                "generic_name": metadata["generic_name"],
                "manufacturer_name": metadata["manufacturer_name"],
            },
            "source_provider": "DailyMed",
            "source": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}",
            **sections,
        }
    return None
