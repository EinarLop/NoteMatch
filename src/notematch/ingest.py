"""Ingestion pipeline (design.md section 4): catalog -> index/.

Run: uv run python -m notematch.ingest
"""

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymupdf
from pydantic import BaseModel, TypeAdapter

from notematch.config import Config, load_config
from notematch.llm import complete_json, embed
from notematch.schemas import IndexEntry, Product, ProductRecord, TocEntry

SUMMARY_PROMPT = """You are describing a study-notes product for a catalog.
Using ONLY the information below, write 2-4 sentences on what the notes cover.
Do not mention anything not present below.

Title: {title}
Subject: {subject}
Table of contents:
{toc}
Author's summary:
{page1_summary}"""


class _Summary(BaseModel):
    summary: str


def load_products(catalog: Path) -> list[Product]:
    """Read metadata.csv. Fails loudly: a skipped product would only surface later
    as a mysteriously bad recommendation."""
    with (catalog / "metadata.csv").open(newline="") as f:
        products = [Product.model_validate(row) for row in csv.DictReader(f)]
    ids = [p.product_id for p in products]
    if dupes := {i for i in ids if ids.count(i) > 1}:
        raise ValueError(f"Duplicate product_id in metadata.csv: {sorted(dupes)}")
    if missing := [i for i in ids if not (catalog / "pdfs" / f"{i}.pdf").exists()]:
        raise FileNotFoundError(f"No PDF for: {missing}")
    return products


def extract(pdf_path: Path) -> tuple[list[list], str]:
    """Return (raw outline, page-1 summary). FR7: reads the outline metadata and
    page 1 only; pages 2+ (the paid content) are never touched."""
    with pymupdf.open(pdf_path) as doc:
        outline = doc.get_toc()                   # [[depth, title, page], ...]
        page1 = doc[0].get_text().splitlines()
    if not outline:
        raise ValueError(f"{pdf_path.name}: no embedded outline (LLM fallback not built, design 4.2)")
    lines = [line.strip() for line in page1]
    try:
        block = lines[lines.index("Summary") + 1 : lines.index("Contents")]
    except ValueError:
        raise ValueError(f"{pdf_path.name}: page 1 has no 'Summary' ... 'Contents' block") from None
    text = "\n".join(block)
    # ponytail: joins every line-end hyphen ("en-\ncoding" -> "encoding"), so a real
    # hyphenated word split across lines loses its hyphen. Harmless for a summarizer input.
    text = text.replace("-\n", "")
    return outline, re.sub(r"\n(?!•)", " ", text)  # unwrap lines, keep one bullet per line


def toc_entries(outline: list[list], title: str, stoplist: list[str]) -> list[tuple[TocEntry, str]]:
    """Outline -> [(TocEntry, text_to_embed)], dropping generic entries.
    The embedded text carries its context: "Title › Parent › Entry"."""
    stop = {s.lower() for s in stoplist}
    path: list[str] = []                          # titles of the current ancestors
    out = []
    for depth, text, page in outline:
        del path[depth - 1 :]                     # climb back up to this entry's parent
        path.append(text)
        if text.strip().lower() not in stop:
            out.append((TocEntry(text=text, depth=depth, page=page), " › ".join([title, *path])))
    return out


def build_index(cfg: Config) -> None:
    products = load_products(cfg.paths.catalog)
    records: list[ProductRecord] = []
    entries: list[IndexEntry] = []
    texts: list[str] = []                         # texts[i] is embedded into vectors[i]

    for n, p in enumerate(products, 1):
        outline, page1_summary = extract(cfg.paths.catalog / "pdfs" / f"{p.product_id}.pdf")
        toc = toc_entries(outline, p.title, cfg.ingestion.toc_stoplist)
        toc_text = "\n".join("  " * (e.depth - 1) + e.text for e, _ in toc)
        prompt = SUMMARY_PROMPT.format(title=p.title, subject=p.subject, toc=toc_text,
                                       page1_summary=page1_summary)
        summary = complete_json(prompt, _Summary, cfg.llm)[0].summary
        print(f"[{n}/{len(products)}] {p.product_id}: {len(toc)} TOC entries")

        records.append(ProductRecord(**p.model_dump(), summary=summary, toc=[e for e, _ in toc]))
        for i, (e, context) in enumerate(toc):
            entries.append(IndexEntry(entry_id=f"{p.product_id}:toc:{i}", product_id=p.product_id,
                                      kind="toc", text=e.text))
            texts.append(context)
        entries.append(IndexEntry(entry_id=f"{p.product_id}:summary", product_id=p.product_id,
                                  kind="summary", text=summary))
        texts.append(f"{p.title}. {summary}")

    vectors = embed(texts, cfg.embedding, "document")

    out = cfg.paths.index
    out.mkdir(parents=True, exist_ok=True)
    (out / "products.json").write_bytes(TypeAdapter(list[ProductRecord]).dump_json(records, indent=2))
    (out / "entries.json").write_bytes(TypeAdapter(list[IndexEntry]).dump_json(entries, indent=2))
    np.save(out / "vectors.npy", vectors)
    (out / "manifest.json").write_text(json.dumps({
        "embedding_model": cfg.embedding.model,
        "dim": int(vectors.shape[1]),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_products": len(records),
        "n_entries": len(entries),
    }, indent=2))
    print(f"Index written to {out}/: {len(records)} products, {len(entries)} entries, dim {vectors.shape[1]}")


if __name__ == "__main__":
    build_index(load_config())
