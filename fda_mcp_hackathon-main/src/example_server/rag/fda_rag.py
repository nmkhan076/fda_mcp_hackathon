from functools import lru_cache
from pathlib import Path

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


def download_pdf():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"FDA PDF not found at: {PDF_PATH}")

    document = fitz.open(str(PDF_PATH))
    chunks = []

    try:
        for page_num, page in enumerate(document, start=1):
            text = page.get_text("text")

            for chunk in chunk_text(text):
                chunks.append(
                    {
                        "text": chunk,
                        "page": page_num,
                        "source": PDF_URL,
                        "source_type": "pdf",
                        "title": "FDA Opioid Label Updates",
                    }
                )

    finally:
        document.close()

    return chunks


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


def search_fda_source(query, source_type, top_k=3):
    if not query.strip():
        return {"success": False, "error": "Search query cannot be empty."}

    if source_type not in ("pdf", "web"):
        return {"success": False, "error": "Invalid source type."}

    try:
        chunks, vectorizer, matrix = build_indexes(source_type)

        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix)[0]

        ranked = scores.argsort()[::-1]

        matches = []

        for index in ranked:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = chunks[index]

            matches.append(
                {
                    "text": chunk["text"],
                    "score": round(score, 4),
                    "source": chunk["source"],
                    "source_type": chunk["source_type"],
                    "title": chunk["title"],
                    "page": chunk["page"],
                }
            )

            if len(matches) >= max(1, min(top_k, 10)):
                break

        return {
            "success": True,
            "query": query,
            "matches": matches,
            "message": ("No relevant passages found." if not matches else None),
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}
