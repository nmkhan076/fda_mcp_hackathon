
import os
import re
import uuid
from example_server.tools.drug_label_history import summarize_record, evidence_for_record

import httpx

URL = "https://api.fda.gov/drug/label.json"

SECTIONS = {
    "Clinical Pharmacology": "clinical_pharmacology",
    "Pharmacokinetics": "pharmacokinetics",
    "Drug Interactions": "drug_interactions",
    "Warnings and Precautions": "warnings_and_precautions",
    "Boxed Warning": "boxed_warning",
}


import re


def extract_evidence(
    text: str,
    targets: list[str],
    context_sentences: int = 1,
    maximum: int = 5,
) -> list[str]:
    """
    Extract complete sentences containing CYP references.

    - Preserve complete words and sentences.
    - Include surrounding context.
    - Merge overlapping passages.
    - Remove duplicate excerpts.
    """

    if not text or not targets:
        return []

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Split at sentence boundaries.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    pattern = re.compile(
        "|".join(
            re.escape(target)
            for target in sorted(targets, key=len, reverse=True)
        ),
        re.IGNORECASE,
    )

    # Find sentences mentioning CYP enzymes.
    matched_indices = [
        index
        for index, sentence in enumerate(sentences)
        if pattern.search(sentence)
    ]

    if not matched_indices:
        return []

    # Create sentence ranges with context.
    ranges = []

    for index in matched_indices:
        start = max(0, index - context_sentences)
        end = min(len(sentences), index + context_sentences + 1)

        ranges.append((start, end))

    # Merge overlapping or adjacent ranges.
    merged_ranges = []

    for start, end in ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            previous_start, previous_end = merged_ranges[-1]
            merged_ranges[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged_ranges.append((start, end))

    # Build final non-overlapping excerpts.
    excerpts = []

    for start, end in merged_ranges:
        excerpt = " ".join(sentences[start:end]).strip()

        if excerpt and excerpt not in excerpts:
            excerpts.append(excerpt)

        if len(excerpts) >= maximum:
            break

    return excerpts

# def extract_evidence(text, targets, context=220, maximum=5):
#     if not text or not targets:
#         return []
#
#     pattern = re.compile(
#         "|".join(re.escape(t) for t in targets),
#         re.IGNORECASE,
#     )
#
#     excerpts = []
#
#     for match in pattern.finditer(text):
#         start = max(0, match.start() - context)
#         end = min(len(text), match.end() + context)
#         excerpt = " ".join(text[start:end].split())
#
#         if excerpt not in excerpts:
#             excerpts.append(excerpt)
#
#         if len(excerpts) >= maximum:
#             break
#
#     return excerpts


async def fetch_label_details(record_id, cyp_targets):
    try:
        record_id = str(uuid.UUID(record_id))
    except (ValueError, TypeError, AttributeError):
        return {"success": False, "error": "Invalid record ID."}

    targets = list(dict.fromkeys(
        str(t).strip().upper()
        for t in cyp_targets
        if str(t).strip()
    ))

    if not targets or len(targets) > 20:
        return {"success": False, "error": "Invalid CYP targets."}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                URL,
                params={
                    "search": f'id:"{record_id}"',
                    "limit": 1,
                },
            )

            if response.status_code == 404:
                return {"success": False, "error": "Record not found."}

            response.raise_for_status()
            results = response.json().get("results", [])

    except httpx.HTTPError as exc:
        return {
            "success": False,
            "error": f"openFDA request failed: {type(exc).__name__}",
        }

    if not results:
        return {"success": False, "error": "Record not found."}

    record = results[0]

    if record.get("id") != record_id:
        return {"success": False, "error": "Record ID mismatch."}


    sections = []
    seen_excerpts = set()

    # Process specific sections before broader sections.
    section_order = [
        "Pharmacokinetics",
        "Drug Interactions",
        "Warnings and Precautions",
        "Boxed Warning",
        "Clinical Pharmacology",
    ]

    for title in section_order:
        field = SECTIONS[title]

        value = record.get(field) or []
        text = (
            "\n".join(map(str, value))
            if isinstance(value, list)
            else str(value)
        )

        matched = [
            target
            for target in targets
            if target.casefold() in text.casefold()
        ]

        if not matched:
            continue

        excerpts = extract_evidence(
            text,
            matched,
            context_sentences=1,
            maximum=5,
        )

        unique_excerpts = []

        for excerpt in excerpts:
            normalized = re.sub(
                r"\s+",
                " ",
                excerpt,
            ).strip().casefold()

            if normalized not in seen_excerpts:
                unique_excerpts.append(excerpt)
                seen_excerpts.add(normalized)

        if unique_excerpts:
            sections.append({
                "title": title,
                "enzymes": matched,
                "excerpts": unique_excerpts,
            })

    openfda = record.get("openfda") or {}

    llm_result = await summarize_record(record, targets)
    return {
        **llm_result,
        "source_evidence": evidence_for_record(record, targets),
        "success": True,
        "id": record_id,
        "set_id": record.get("set_id"),
        "version": record.get("version"),
        "effective_date": record.get("effective_time"),
        "brand_name": openfda.get("brand_name") or [],
        "manufacturer": openfda.get("manufacturer_name") or [],
        "sections": sections,
    }


def register(mcp):
    @mcp.tool()
    async def get_drug_cyp_label_details(
        record_id: str,
        cyp_targets: list[str],
    ) -> dict:
        """Retrieve CYP evidence for one FDA labeling record."""
        return await fetch_label_details(record_id, cyp_targets)
