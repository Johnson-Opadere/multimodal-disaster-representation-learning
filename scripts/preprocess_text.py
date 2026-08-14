#!/usr/bin/env python3
"""
preprocess_text.py
==================

Project 2A — Raw Text Aggregation Pipeline
------------------------------------------

Builds event-level text files from raw disaster reports.

This script intentionally performs ONLY lightweight preprocessing.

Rationale
---------
Project 2A's semantic supervision pipeline is:

    reports/
        ↓
    preprocess_text.py
        ↓
    raw_text/
        ↓
    filter_text.py
        ↓
    text_filtered/fragments.jsonl
        ↓
    build_text_tokens.py
        ↓
    tokens.pt

The REAL semantic filtering happens later inside:
    filter_text.py

Therefore:
    preprocess_text.py should remain lightweight,
    stable, deterministic, and reproducible.

Behavior
--------
- concatenates report text files
- preserves paragraph structure
- preserves article boundaries
- normalizes unicode
- normalizes whitespace INSIDE paragraphs
- does NOT perform semantic filtering
- metadata used only for dataset consistency

Output Structure
----------------
data/raw_text/

Each file:
    <event_id>.txt

Example
-------
PYTHONPATH=. python3 2A/scripts/preprocess_text.py \
    --reports_root data/reports \
    --metadata_root data/metadata \
    --output_dir data/raw_text
"""

import os
import re
import json
import argparse
import unicodedata


# ============================================================
# Text Cleaning
# ============================================================
def clean_text(text: str) -> str:
    """
    Lightweight normalization suitable for transformer pipelines.

    Operations:
        - unicode normalization
        - line-ending normalization
        - tab cleanup
        - paragraph preservation
        - whitespace normalization inside paragraphs

    Important:
        This stage intentionally avoids:
            - semantic filtering
            - aggressive regex cleaning
            - sentence filtering
            - keyword filtering
            - chunking heuristics
    """

    if text is None:
        return ""

    # --------------------------------------------------------
    # Unicode normalization
    # --------------------------------------------------------
    text = unicodedata.normalize("NFKC", text)

    # --------------------------------------------------------
    # Normalize line endings
    # --------------------------------------------------------
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # --------------------------------------------------------
    # Replace tabs
    # --------------------------------------------------------
    text = text.replace("\t", " ")

    # --------------------------------------------------------
    # Split into paragraphs
    # Blank lines define paragraph boundaries
    # --------------------------------------------------------
    paragraphs = re.split(r"\n\s*\n+", text)

    cleaned = []

    for p in paragraphs:

        # Normalize whitespace INSIDE paragraph only
        p = re.sub(r"\s+", " ", p)

        p = p.strip()

        if len(p) > 0:
            cleaned.append(p)

    # Preserve paragraph spacing
    return "\n\n".join(cleaned)


# ============================================================
# Metadata Loader
# ============================================================
def load_metadata(metadata_root: str):
    """
    Lightweight metadata validation.

    Ensures metadata files exist and are readable.
    """

    metadata = {}

    for fname in os.listdir(metadata_root):

        if not fname.endswith(".json"):
            continue

        event_id = fname.replace(".json", "")

        path = os.path.join(metadata_root, fname)

        with open(path, "r") as f:
            obj = json.load(f)

        metadata[event_id] = obj

    return metadata


# ============================================================
# Build Event-Level Text
# ============================================================
def build_text(
    reports_root: str,
    metadata_root: str,
    output_dir: str,
):
    """
    Build event-level raw text files.
    """

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------
    metadata = load_metadata(metadata_root)

    # --------------------------------------------------------
    # Process events
    # --------------------------------------------------------
    events = sorted(os.listdir(reports_root))

    for event_id in events:

        event_dir = os.path.join(reports_root, event_id)

        if not os.path.isdir(event_dir):
            continue

        # Skip events without metadata
        if event_id not in metadata:
            print(f" Skipping {event_id} (missing metadata)")
            continue

        texts = []

        txt_files = sorted(
            f for f in os.listdir(event_dir)
            if f.endswith(".txt")
        )

        for fname in txt_files:

            path = os.path.join(event_dir, fname)

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:

                raw = f.read()

            cleaned = clean_text(raw)

            if len(cleaned) > 0:
                texts.append(cleaned)

        # ----------------------------------------------------
        # Preserve article boundaries
        # ----------------------------------------------------
        final_text = "\n\n".join(texts)

        out_path = os.path.join(
            output_dir,
            f"{event_id}.txt",
        )

        with open(
            out_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(final_text)

        print(
            f" Saved {out_path} "
            f"({len(final_text)} chars)"
        )


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reports_root",
        required=True,
        help="Path to data/reports/",
    )

    parser.add_argument(
        "--metadata_root",
        required=True,
        help="Path to data/metadata/",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to output raw_text/",
    )

    args = parser.parse_args()

    build_text(
        reports_root=args.reports_root,
        metadata_root=args.metadata_root,
        output_dir=args.output_dir,
    )