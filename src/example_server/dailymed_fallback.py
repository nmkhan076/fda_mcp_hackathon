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
    """Match brand name at the beginning of the SPL title."""
    normalize = lambda s: re.sub(
        r"\s+", " ", s.strip().casefold()
    )

    title = normalize(title)
    name = normalize(name)

    return bool(
        re.match(
            rf"^{re.escape(name)}(?=$|[\s(\-])",
            title,
        )
    )


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


async def find_dailymed_brand( drug_name: str, client: httpx.AsyncClient, ):
        """Find a branded DailyMed SPL without generic substitution."""

        candidates = []

        for page in range(1, 4):
            response = await client.get(
                f"{BASE}/spls.json",
                params={
                    "drug_name": drug_name,
                    "name_type": "b",
                    "pagesize": 100,
                    "page": page,
                },
            )

            print(
                f"DailyMed search: {drug_name}, "
                f"page={page}, status={response.status_code}"
            )

            if response.status_code == 404:
                break

            response.raise_for_status()
            payload = response.json()
            print("Response keys:", list(payload.keys()))
            print("Metadata:", payload.get("metadata"))
            print("Raw response:", str(payload)[:3000])
            # Support DailyMed's column-based JSON format.
            columns = payload.get("COLUMNS", payload.get("columns", []))
            rows = payload.get("DATA", payload.get("data", []))

            if isinstance(rows, dict):
                rows = rows.get("spl", rows.get("spls", []))

            if columns and rows and isinstance(rows[0], list):
                items = [dict(zip(columns, row)) for row in rows]
            elif isinstance(rows, list):
                items = rows
            else:
                items = []

            print("Number of candidates:", len(items))
            print("First candidates:", items[:3])

            for item in items:
                normalized_item = {
                    str(k).lower(): v
                    for k, v in item.items()
                }

                title = normalized_item.get("title", "")

                print("DailyMed candidate:", title)

                if _brand_matches(title, drug_name):
                    candidates.append(normalized_item)

            metadata = payload.get(
                "METADATA",
                payload.get("metadata", {})
            )

            total_pages = int(
                metadata.get(
                    "total_pages",
                    metadata.get("TOTAL_PAGES", 1)
                )
            )

            if page >= total_pages:
                break

        if not candidates:
            print(
                f"No current DailyMed branded label for {drug_name}. "
                "Historical SPL lookup required."
            )
            return None

        candidates.sort(
            key=lambda r: (
                _date(r.get("published_date")),
                int(r.get("spl_version") or 0),
            ),
            reverse=True,
        )

        chosen = candidates[0]

        set_id = chosen.get("setid")

        if not set_id:
            raise ValueError(
                "DailyMed returned a record without setid"
            )

        xml_response = await client.get(
            f"{BASE}/spls/{set_id}.xml",
            timeout=45,
        )

        xml_response.raise_for_status()

        sections = parse_spl(xml_response.content)
        metadata = extract_spl_metadata(xml_response.content)
        if not sections:
            raise ValueError(
                "DailyMed SPL found but no labeling "
                "sections were extracted"
            )

        return {
            "id": None,
            "set_id": set_id,
            "version": chosen.get("spl_version"),
            "effective_time": _date(
                chosen.get("published_date")
            ),
            "published_date": chosen.get("published_date"),
            "openfda": {
                "brand_name": [drug_name],
                "generic_name": metadata["generic_name"],
                "manufacturer_name": metadata["manufacturer_name"],
            },
            "source_provider": "DailyMed",
            "source": (
                "https://dailymed.nlm.nih.gov/dailymed/"
                f"drugInfo.cfm?setid={set_id}"
            ),
            **sections,
        }

