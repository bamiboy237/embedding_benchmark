"""Chunk an xlsx into chunk records (no embedding)."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import sys
from typing import Any
import re
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from normalise import load_and_flatten_data


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


_NUM_SPLIT_RE = re.compile(r"\d+")


def _dotted_number_key(value: Any) -> tuple[int, ...]:
    """Return a tuple key that sorts dotted numeric strings naturally.

    Examples:
    - "2" -> (2,)
    - "206.1" -> (206, 1)
    - ""/None -> ()
    """
    if value is None:
        return ()
    s = str(value).strip()
    if not s:
        return ()
    parts = _NUM_SPLIT_RE.findall(s)
    if not parts:
        return ()
    return tuple(int(p) for p in parts)


def _sort_flattened_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sort flattened rows so parents come before children in numeric order."""
    # Map parent section id -> chapter so subsections inherit a chapter for sorting.
    parent_chapter: dict[str, str] = {}
    if "id" in df.columns and "chapter" in df.columns and "level" in df.columns:
        parents = df[df["level"].astype(int) == 0]
        parent_chapter = dict(zip(parents["id"].astype(str), parents["chapter"].fillna("").astype(str)))

    def effective_chapter(row: pd.Series) -> str:
        chap = str(row.get("chapter", "") or "").strip()
        if chap:
            return chap
        parent_id = row.get("parent_section_id")
        if pd.notna(parent_id):
            return str(parent_chapter.get(str(parent_id), ""))
        return ""

    # Group rows by top-level section id so each subsection stays next to its parent.
    def group_id(row: pd.Series) -> str:
        parent_id = row.get("parent_section_id")
        if pd.notna(parent_id) and str(parent_id).strip():
            return str(parent_id)
        return str(row.get("id", ""))

    df2 = df.copy()
    df2["__chapter_key"] = df2.apply(lambda r: _dotted_number_key(effective_chapter(r)), axis=1)
    df2["__group_key"] = df2.apply(lambda r: _dotted_number_key(group_id(r)), axis=1)
    df2["__level_key"] = df2["level"].astype(int) if "level" in df2.columns else 0
    df2["__section_num_key"] = df2["section_num"].apply(_dotted_number_key) if "section_num" in df2.columns else df2.apply(lambda _: (), axis=1)
    df2["__id_key"] = df2["id"].apply(_dotted_number_key) if "id" in df2.columns else df2.apply(lambda _: (), axis=1)

    # Stable sort so relative order is preserved when keys tie.
    df2 = df2.sort_values(
        by=["__chapter_key", "__group_key", "__level_key", "__section_num_key", "__id_key"],
        kind="mergesort",
    )

    return df2.drop(
        columns=["__chapter_key", "__group_key", "__level_key", "__section_num_key", "__id_key"],
        errors="ignore",
    )


def write_jsonl(rows: list[dict], output_path: Path) -> None:
    """Write list of dicts as JSONL (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(
    input_file: Path,
    output_file: Path,
    save_flattened: Path | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    threshold: int = 2000,
) -> None:
    """Load Excel file, flatten, chunk text, and save with metadata."""
    logger.info(f"Chunking rows from {input_file}")

    # Validate input file
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    if input_file.suffix != ".xlsx":
        raise ValueError(f"Input file {input_file} is not a valid Excel file (.xlsx).")

    # Load and optionally save flattened data
    logger.info("Loading and flattening data...")
    df: pd.DataFrame = load_and_flatten_data(input_file, output_file=save_flattened)
    if df is None or df.empty:
        raise ValueError(f"No data found in {input_file}.")

    # Sort rows into a consistent section/subsection order (independent of Excel row order).
    df = _sort_flattened_rows(df)

    # Initialize text splitter with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunk_rows: list[dict] = []
    docs_kept_whole = 0
    docs_chunked = 0

    # Process each row: chunk if long, keep whole if short
    for _, row in df.iterrows():
        # Build metadata dict for this row
        base = {
            "id": row["id"],
            "section_num": str(row["section_num"]),
            "title": row["title"],
            "level": int(row["level"]),
            "parent_section_id": str(row["parent_section_id"]) if pd.notna(row["parent_section_id"]) else None,
            "chapter": row["chapter"],
            "is_subsection": bool(row["is_subsection"]),
            "source": input_file.name,
        }

        text = row["text"]
        # Keep short text whole, split long text into overlapping chunks
        if len(text) <= threshold:
            chunk_rows.append({**base, "chunk_index": 0, "chunk_total": 1, "text": text})
            docs_kept_whole += 1
        else:
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_rows.append({**base, "chunk_index": i, "chunk_total": len(chunks), "text": chunk})
            docs_chunked += 1

    logger.info(f"Documents kept whole: {docs_kept_whole}")
    logger.info(f"Documents chunked: {docs_chunked}")
    logger.info(f"Total chunks: {len(chunk_rows)}")

    # Write output in requested format
    if output_file.suffix.lower() == ".csv":
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(chunk_rows).to_csv(output_file, index=False)
    elif output_file.suffix.lower() in [".jsonl", ".ndjson"]:
        write_jsonl(chunk_rows, output_file)
    else:
        raise ValueError("Output file must end with .csv, .jsonl, or .ndjson")

    logger.info(f"Wrote chunk output to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk legal text from Excel into chunk records (no embedding).")
    parser.add_argument("--input", type=Path, required=True, help="Path to input Excel file (.xlsx)")
    parser.add_argument("--output", type=Path, required=True, help="Path to output (.csv or .jsonl)")
    parser.add_argument("--save-flattened", type=Path, default=None, help="Optional: save flattened CSV for debugging")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--threshold", type=int, default=2000, help="If text length <= threshold, keep unchunked")

    args = parser.parse_args()
    main(
        input_file=args.input,
        output_file=args.output,
        save_flattened=args.save_flattened,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        threshold=args.threshold,
    )