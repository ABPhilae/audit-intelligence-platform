"""
Document Preprocessing Pipeline.

This is the QUALITY step between loading and chunking.
Raw document text often contains noise that hurts retrieval quality:
- Repeated headers and footers on every page
- Page numbers ("Page 3 of 25")
- Extra whitespace and formatting artifacts
- Inconsistent encoding

This pipeline cleans all of that before the text reaches the embedding model.

Analogy: Imagine you are preparing ingredients for a chef.
Loading = buying the vegetables (getting raw text from files).
Preprocessing = washing, peeling, and trimming (cleaning the text).
Chunking = chopping to the right size.
Embedding = cooking (creating vector representations).
Skipping preprocessing is like cooking unwashed vegetables — the meal
still works, but quality suffers.
"""
import re
import logging

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """
    Clean and normalise document text.

    Steps:
    1. Remove common headers/footers patterns
    2. Remove page numbers
    3. Normalise whitespace
    4. Normalise unicode characters

    Args:
        text: Raw text from a document loader

    Returns:
        Cleaned text ready for chunking
    """
    if not text:
        return ""

    # Step 1: Remove common page number patterns
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"- \d+ -", "", text)
    text = re.sub(r"\n\d+\n", "\n", text)

    # Step 2: Remove repeated header/footer patterns
    # (lines that appear on every page tend to be headers/footers)
    lines = text.split("\n")
    if len(lines) > 10:
        line_counts: dict = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100:  # Headers are usually short
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        # If a short line appears more than 3 times, it is likely a header/footer
        repeated = {line for line, count in line_counts.items() if count > 3}
        if repeated:
            lines = [line for line in lines if line.strip() not in repeated]
            text = "\n".join(lines)

    # Step 3: Normalise whitespace
    text = re.sub(r" +", " ", text)  # Multiple spaces → single space
    text = re.sub(r"\n{3,}", "\n\n", text)  # Multiple blank lines → double
    text = re.sub(r"[ \t]+\n", "\n", text)  # Trailing whitespace

    # Step 4: Normalise common unicode issues
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # Smart quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # Dashes
    text = text.replace("\xa0", " ")  # Non-breaking space

    return text.strip()


def enrich_metadata(metadata: dict, text: str) -> dict:
    """
    Add computed metadata to a document chunk.

    Enriched metadata enables better filtering at query time.
    """
    enriched = metadata.copy()

    # Detect if the text mentions specific regions
    text_lower = text.lower()
    regions = []
    region_keywords = {
        "hong_kong": ["hong kong", "hk", "hkma"],
        "singapore": ["singapore", "sg", "mas"],
        "tokyo": ["tokyo", "japan", "jfsa"],
        "sydney": ["sydney", "australia", "apra"],
    }
    for region, keywords in region_keywords.items():
        if any(kw in text_lower for kw in keywords):
            regions.append(region)
    enriched["detected_regions"] = regions

    # Detect content categories
    categories = []
    category_keywords = {
        "audit_finding": ["finding", "observation", "weakness", "deficiency"],
        "compliance": ["compliance", "regulatory", "regulation", "policy"],
        "risk": ["risk", "threat", "vulnerability", "exposure"],
        "remediation": ["remediation", "action item", "deadline", "corrective"],
        "financial": ["budget", "cost", "revenue", "financial", "hkd", "usd"],
    }
    for category, keywords in category_keywords.items():
        if any(kw in text_lower for kw in keywords):
            categories.append(category)
    enriched["detected_categories"] = categories

    # Word count (useful for chunk quality assessment)
    enriched["word_count"] = len(text.split())

    return enriched
