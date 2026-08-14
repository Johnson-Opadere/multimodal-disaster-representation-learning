#!/usr/bin/env python3
"""
clean_whisper_transcripts.py
============================

Project 2A — Whisper Transcript Cleaning Pipeline
-------------------------------------------------

Cleans merged Whisper transcripts before semantic filtering.

Pipeline Role
-------------
event_whisper/
    ↓
merge_whisper_transcripts.py
    ↓
event_whisper_merged/
    ↓
clean_whisper_transcripts.py
    ↓
raw_whisper/
    ↓
filter_whisper.py
    ↓
whisper_filtered/fragments.jsonl

Purpose
-------
Whisper transcripts often contain:
    - filler words
    - speech artifacts
    - ASR punctuation inconsistencies
    - repeated punctuation
    - stray formatting noise

This script performs lightweight cleaning while preserving:
    - disaster semantics
    - spoken descriptions
    - narrative structure

Cleaning Operations
-------------------
- remove filler words
- normalize punctuation
- collapse repeated spaces
- remove repeated punctuation
- remove stray line-edge dashes
- trim whitespace

Input
-----
event_whisper_merged/
    <event>.txt

Output
------
raw_whisper/
    <event>.txt

Behavior
--------
- loads merged event transcripts
- performs lightweight transcript cleaning
- preserves semantic content
- saves cleaned event-level transcripts

Example
-------
Input:
    raw Whisper transcript with:
        "uh"
        "um"
        repeated punctuation
        strange quotes
        speech artifacts

Output:
    cleaner normalized transcript suitable for:
        - semantic filtering
        - tokenization
        - multimodal retrieval training

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/clean_whisper_transcripts.py

Custom Paths
------------
PYTHONPATH=. python3 2A/scripts/clean_whisper_transcripts.py \
    --in_root event_whisper_merged \
    --out_root raw_whisper
"""

import os
import re
from pathlib import Path
import argparse


# ============================================================
# Filler Words
# ============================================================
# Expandable list of common speech fillers.
#
# These frequently appear in:
#   - interviews
#   - spoken narration
#   - ASR transcripts
# ============================================================

FILLER_WORDS = [

    r"\buh\b",
    r"\bum\b",
    r"\bumm\b",
    r"\buhh\b",

    r"\bya know\b",
    r"\byou know\b",

    r"\blike\b",

    r"\bokay\b",

    # Mild filler words often unnecessary
    # at sentence starts
    r"\bso\b",

    r"\bi mean\b",

    r"\bwell\b",
]

# ------------------------------------------------------------
# Combined filler-word regex
# ------------------------------------------------------------
FILLER_RE = re.compile(
    "(" + "|".join(FILLER_WORDS) + ")",
    flags=re.I
)


# ============================================================
# Smart Punctuation Normalization
# ============================================================
PUNCT_REPLACEMENTS = {

    "“": '"',
    "”": '"',

    "‘": "'",
    "’": "'",

    "–": "-",
    "—": "-",
    "−": "-",
}


# ============================================================
# Normalize Punctuation
# ============================================================
def normalize_punctuation(text):
    """
    Normalize unicode punctuation into standard ASCII forms.
    """

    for bad, good in PUNCT_REPLACEMENTS.items():

        text = text.replace(bad, good)

    return text


# ============================================================
# Main Cleaning Function
# ============================================================
def clean_text_block(text):
    """
    Perform lightweight Whisper transcript cleaning.

    Important:
        Cleaning is intentionally conservative.

    Goal:
        preserve semantic disaster content while removing:
            - filler speech
            - punctuation noise
            - formatting artifacts
    """

    # --------------------------------------------------------
    # Remove filler words
    # --------------------------------------------------------
    text = FILLER_RE.sub("", text)

    # --------------------------------------------------------
    # Normalize punctuation
    # --------------------------------------------------------
    text = normalize_punctuation(text)

    # --------------------------------------------------------
    # Remove repeated punctuation
    #
    # Example:
    #   "!!!" -> "!"
    # --------------------------------------------------------
    text = re.sub(
        r"[,.!?]{2,}",
        lambda m: m.group(0)[0],
        text
    )

    # --------------------------------------------------------
    # Remove stray dashes at line starts
    # --------------------------------------------------------
    text = re.sub(
        r"^\s*-\s*",
        "",
        text,
        flags=re.M
    )

    # --------------------------------------------------------
    # Collapse repeated spaces
    # --------------------------------------------------------
    text = re.sub(
        r"\s{2,}",
        " ",
        text
    )

    # --------------------------------------------------------
    # Trim leading/trailing whitespace
    # --------------------------------------------------------
    text = text.strip()

    return text


# ============================================================
# Clean Single Event
# ============================================================
def clean_event(event_file, out_dir):
    """
    Clean one merged event transcript.
    """

    event = event_file.stem

    text = event_file.read_text(
        encoding="utf-8"
    )

    print(f"🧹 Cleaning {event_file.name} ...")

    cleaned = clean_text_block(text)

    out_path = out_dir / f"{event}.txt"

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save cleaned transcript
    # --------------------------------------------------------
    out_path.write_text(
        cleaned,
        encoding="utf-8"
    )

    print(f"   ✔ Saved cleaned → {out_path}")

    print(
        f"   📏 Length: "
        f"{len(cleaned):,} chars\n"
    )


# ============================================================
# Main Pipeline
# ============================================================
def main(
    in_root="event_whisper_merged",
    out_root="raw_whisper"
):
    """
    Clean all merged Whisper transcripts.
    """

    in_root = Path(in_root)

    out_root = Path(out_root)

    # --------------------------------------------------------
    # Validate input directory
    # --------------------------------------------------------
    if not in_root.exists():

        print(
            f"❌ ERROR: "
            f"Folder not found: {in_root}"
        )

        return

    # --------------------------------------------------------
    # Discover transcript files
    # --------------------------------------------------------
    files = sorted(
        in_root.glob("*.txt")
    )

    print(
        f"🎧 Found {len(files)} "
        f"merged transcripts\n"
    )

    # --------------------------------------------------------
    # Clean each event transcript
    # --------------------------------------------------------
    for f in files:

        clean_event(
            f,
            out_root
        )

    print(
        "🎉 ALL DONE — "
        "Clean Whisper transcripts ready!\n"
    )


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_root",
        default="event_whisper_merged",
        help="Directory containing merged Whisper transcripts."
    )

    parser.add_argument(
        "--out_root",
        default="raw_whisper",
        help="Directory to save cleaned Whisper transcripts."
    )

    args = parser.parse_args()

    main(
        args.in_root,
        args.out_root
    )