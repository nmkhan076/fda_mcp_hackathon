from functools import lru_cache
from pathlib import Path
import re

import pymupdf as fitz
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PDF_URL = "https://www.fda.gov/media/167056/download"

WEB_URL = (
    "https://www.fda.gov/drugs/"
    "food-and-drug-administration-overdose-prevention-framework/"
    "opioid-medications"
)
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

PDF_PATH = DATA_DIR / "opioid_label_updates.pdf"


def chunk_text(text, max_words=180, overlap=35):
    words = text.split()
    step = max_words - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            yield chunk


def _clean_text(value):
    """Normalize spacing without joining words or rewriting FDA wording."""
    value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _table_regions(page):
    """Exclude detected tables from prose extraction to avoid mixed columns."""
    if not hasattr(page, "find_tables"):
        return []
    try:
        return [fitz.Rect(t.bbox) for t in page.find_tables().tables]
    except (ValueError, AttributeError, RuntimeError):
        return []


def _is_table_like(text):
    """Reject flattened table rows and table headings from prose results."""
    stripped = text.strip()
    if re.match(r"^(?:Table\s+\d+\s*[:.]|Former\s*\nNew\b)", stripped, re.I):
        return True
    if re.search(r"\bFormer\s+New\b", stripped, re.I):
        return True
    if len(re.findall(r"\bTable\s+\d+\b", stripped, re.I)) >= 2:
        return True
    if len(stripped.split()) < 28 and 'table' in stripped.casefold():
        return True
    return False


def _append_passage(chunks, text, page_num):
    if len(text.split()) >= 25 and not _is_table_like(text):
        chunks.append({"text": text, "page": page_num, "source": PDF_URL,
                       "source_type": "pdf", "title": "FDA Opioid Label Updates"})


def download_pdf():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"FDA PDF not found at: {PDF_PATH}")
    chunks = []
    with fitz.open(str(PDF_PATH)) as document:
        for page_num, page in enumerate(document, 1):
            table_regions = _table_regions(page)
            blocks = []
            for block in page.get_text("blocks", sort=True):
                if len(block) < 7 or block[6] != 0:
                    continue
                rect = fitz.Rect(block[:4])
                if any((rect & region).get_area() > rect.get_area() * .15
                       for region in table_regions):
                    continue
                text = _clean_text(block[4])
                if len(text) >= 35:
                    blocks.append(text)
            # Keep individual coherent blocks; never join an introduction to
            # a table header or concatenate unrelated table columns.
            for block in blocks:
                for paragraph in re.split(r"\n\s*\n", block):
                    paragraph = paragraph.strip()
                    if _is_table_like(paragraph):
                        continue
                    if len(paragraph.split()) > 135:
                        # Long block: use sentence boundaries, not word slicing.
                        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
                        group = []
                        for sentence in sentences:
                            if sum(len(x.split()) for x in group) + len(sentence.split()) > 135 and group:
                                _append_passage(chunks, " ".join(group), page_num)
                                group = []
                            group.append(sentence)
                        if group:
                            _append_passage(chunks, " ".join(group), page_num)
                    else:
                        _append_passage(chunks, paragraph, page_num)
    return chunks


def extract_pdf_tables():
    """Return table locations; display the original PDF, not corrupted cells."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"FDA PDF not found at: {PDF_PATH}")
    tables = []
    with fitz.open(str(PDF_PATH)) as document:
        for page_num, page in enumerate(document, 1):
            if not hasattr(page, "find_tables"):
                continue
            for table_num, table in enumerate(page.find_tables().tables, 1):
                rows = table.extract()
                content = " ".join(str(cell or "") for row in rows for cell in row)
                content = _clean_text(content)
                if len(content) < 60:
                    continue
                # Only small metadata crosses MCP. Flask renders images separately.
                bbox = [round(float(v), 2) for v in table.bbox]
                tables.append({"id": f"T{page_num}-{table_num}", "type": "table",
                               "title": "FDA Opioid Label Updates", "page": page_num,
                               "table_number": table_num, "bbox": bbox,
                               "source": PDF_URL, "text": content[:650],
                               "display_mode": "original_pdf",
                               "table_note": "Original table preview served separately by Flask."})
    return tables


def download_webpage():
    """Scrape the FDA Opioid Medications webpage."""

    headers = {"User-Agent": "FDA-Research-RAG/1.0", "Accept": "text/html"}

    response = requests.get(WEB_URL, headers=headers, timeout=30)

    print("Requested URL:", WEB_URL)
    print("Final URL:", response.url)
    print("HTTP status:", response.status_code)

    response.raise_for_status()

    # Detect FDA website protection redirects
    if (
        "abuse-detection" in response.url.lower()
        or "apology_objects" in response.url.lower()
    ):
        raise RuntimeError(
            "FDA blocked automated webpage access. Cannot safely scrape this page."
        )

    soup = BeautifulSoup(response.text, "html.parser")

    main = soup.find("main") or soup.find("article")

    if main is None:
        raise ValueError("FDA article content could not be located.")

    # Remove non-article content
    for element in main.select("script, style, nav, footer, header, form, aside"):
        element.decompose()

    # Extract paragraphs and headings
    elements = main.find_all(["h1", "h2", "h3", "p", "li"])

    paragraphs = [
        element.get_text(" ", strip=True)
        for element in elements
        if element.get_text(" ", strip=True)
    ]

    text = "\n".join(paragraphs)

    if len(text) < 200:
        raise ValueError("Insufficient FDA article content extracted.")

    chunks = []

    for chunk in chunk_text(text):
        chunks.append(
            {
                "text": chunk,
                "page": None,
                "source": WEB_URL,
                "source_type": "web",
                "title": "FDA Opioid Medications",
            }
        )

    print("Webpage chunks created:", len(chunks))

    return chunks


@lru_cache(maxsize=2)
def build_indexes(source_type):

    if source_type == "pdf":
        chunks = download_pdf()

    elif source_type == "web":
        chunks = download_webpage()

    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    if not chunks:
        raise ValueError(f"No content extracted from {source_type}")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    return chunks, vectorizer, matrix


def _search_query(query):
    """Conservative query expansion for safety-instruction terminology."""
    q = query.strip()
    lower = q.casefold()
    if "opioid" in lower and any(word in lower for word in ("safety", "safe use", "instruction", "counsel")):
        q += " patient counseling medication guide safe use storage disposal overdose respiratory depression accidental ingestion"
    return q


def search_fda_source(query, source_type, top_k=3):
    if not query.strip():
        return {"success": False, "error": "Search query cannot be empty."}
    if source_type not in ("pdf", "web"):
        return {"success": False, "error": "Invalid source type."}
    try:
        chunks, vectorizer, matrix = build_indexes(source_type)
        query_vector = vectorizer.transform([_search_query(query)])
        scores = cosine_similarity(query_vector, matrix)[0]
        # A relevant coherent paragraph is preferred to a corrupted table.
        matches = []
        for index in scores.argsort()[::-1]:
            score = float(scores[index])
            if score <= 0:
                break
            chunk = chunks[index]
            if _is_table_like(chunk["text"]) or ("following tables provide" in chunk["text"].casefold()):
                continue
            if any(chunk["page"] == m["page"] and
                   chunk["text"][:90] == m["text"][:90] for m in matches):
                continue
            matches.append({**chunk, "score": round(score, 4), "type": "passage"})
            if len(matches) >= 1:
                break
        evidence = []
        selected_tables = []
        if source_type == "pdf":
            try:
                tables = extract_pdf_tables()
                if tables:
                    table_scores = cosine_similarity(
                        query_vector, vectorizer.transform([t["text"] for t in tables]))[0]
                    # Table is an additional evidence section, not competing for
                    # the top_k passage slots. Avoid an arbitrary 0.15 cutoff.
                    for idx in table_scores.argsort()[::-1]:
                        score = float(table_scores[idx])
                        if score <= 0:
                            break
                        item = dict(tables[idx])
                        item["score"] = round(score, 4)
                        selected_tables.append(item)
                        if len(selected_tables) >= 3:
                            break
            except (AttributeError, ValueError, RuntimeError) as exc:
                print(f"PDF table extraction unavailable: {exc}")
        evidence = evidence + selected_tables
        for n, item in enumerate(evidence, 1):
            item["id"] = f"E{n}"
            item["source_url"] = (f"{item['source']}#page={item['page']}"
                                  if item.get("page") else item["source"])
        return {"success": True, "query": query,
                "matches": selected_tables,
                "evidence": evidence,
                "tables": selected_tables,
                "best_evidence": evidence[0] if evidence else None,
                "message": None if evidence else "No relevant evidence found."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
