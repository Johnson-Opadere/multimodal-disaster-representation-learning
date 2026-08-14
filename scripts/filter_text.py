#!/usr/bin/env python3
"""
filter_text.py
==============

Project 2A — Semantic Text Filtering Pipeline
---------------------------------------------

Filters raw disaster-report text into semantically meaningful
damage-related fragments for multimodal retrieval training.

Pipeline Role
-------------
raw_text/
    ↓
filter_text.py
    ↓
text_filtered/fragments.jsonl
    ↓
build_text_tokens.py
    ↓
tokens.pt

Behavior
--------
- deterministic sentence splitting
- lightweight normalization
- removes attribution-heavy journalism fragments
- keeps only damage-semantic sentences
- assigns semantic damage buckets
- preserves source event metadata

Output
------
data/text_filtered/fragments.jsonl

Each line:
{
    "text": "...",
    "damage_bucket": "...",
    "source_event": "..."
}

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/filter_text.py
"""

import os
import re
import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================
RAW_TEXT_DIR = "data/raw_text"

OUT_DIR = "data/text_filtered"

OUT_FILE = os.path.join(
    OUT_DIR,
    "fragments.jsonl"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Damage Semantics
# ============================================================
DAMAGE_OUTCOME_KEYWORDS = {
    "destroyed",
    "damaged",
    "collapsed",
    "buried",
    "burned",
    "submerged",
    "inundated",
    "washed",
    "blocked",
    "covered",
    "engulfed",
    "ruined",
    "uninhabitable",
    "rubble",
}

AFFECTED_OBJECTS = {
    "homes",
    "houses",
    "buildings",
    "villages",
    "roads",
    "bridges",
    "infrastructure",
    "schools",
    "farms",
    "crops",
    "communities",
    "apartments",
    "offices",
}


# ============================================================
# Attribution / Journalism Noise
# ============================================================
ATTRIBUTION_PHRASES = {
    "associated press",
    "ap news",
    "reuters",
    "said",
    "told",
    "according to",
    "reported",
    "news network",
}


# ============================================================
# Semantic Bucket Refinement
# ============================================================
EVENT_REFINERS = {

    # --------------------------------------------------------
    # Volcanic damage
    # --------------------------------------------------------
    "volcanic_damage": {
        "ash",
        "lava",
        "pyroclastic",
        "lahar",
    },

    # --------------------------------------------------------
    # Flooding / inundation
    # --------------------------------------------------------
    "flooding": {
        "flood",
        "flooded",
        "inundation",
        "rainfall",
    },

    # --------------------------------------------------------
    # Wildfire damage
    # --------------------------------------------------------
    "wildfire": {
        "wildfire",
        "burn",
        "smoke",
        "charred",
    },

    # --------------------------------------------------------
    # Tsunami damage
    # --------------------------------------------------------
    "tsunami_inundation": {
        "tsunami",
        "wave",
        "coastal",
    },

    # --------------------------------------------------------
    # Cross-event structural damage
    # IMPORTANT:
    # Unified bucket for:
    #   earthquake collapse
    #   rubble
    #   uninhabitable structures
    # --------------------------------------------------------
    "structural_damage": {
        "earthquake",
        "seismic",
        "collapsed",
        "rubble",
        "uninhabitable",
    },
}


# ============================================================
# Deterministic Sentence Splitter
# (NLTK intentionally removed)
# ============================================================
def sent_tokenize(text):
    """
    Lightweight deterministic sentence splitter.
    """

    return re.split(
        r"(?<=[.!?])\s+",
        text,
    )


# ============================================================
# Lightweight Normalization
# ============================================================
def normalize(text):
    """
    Normalize sentence text.

    Operations:
        - lowercase
        - remove numbers
        - collapse whitespace
    """

    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Attribution Filtering
# ============================================================
def is_attribution_heavy(text):
    """
    Remove journalism-heavy fragments.
    """

    return any(
        p in text
        for p in ATTRIBUTION_PHRASES
    )


# ============================================================
# Damage Semantic Filtering
# ============================================================
def has_damage_semantics(text):
    """
    Keep only:
        damage outcome
        +
        affected object

    Example:
        "homes were destroyed"
    """

    return (
        any(k in text for k in DAMAGE_OUTCOME_KEYWORDS)
        and
        any(o in text for o in AFFECTED_OBJECTS)
    )


# ============================================================
# Semantic Bucket Assignment
# ============================================================
def assign_bucket(text):
    """
    Assign semantic damage bucket.
    """

    for bucket, kws in EVENT_REFINERS.items():

        if any(k in text for k in kws):
            return bucket

    return "generic_damage"


# ============================================================
# Main Filtering Loop
# ============================================================
with open(OUT_FILE, "w") as out_f:

    for fname in os.listdir(RAW_TEXT_DIR):

        if not fname.endswith(".txt"):
            continue

        event = fname.replace(".txt", "")

        path = os.path.join(
            RAW_TEXT_DIR,
            fname,
        )

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            raw = f.read()

        # ----------------------------------------------------
        # Sentence splitting
        # ----------------------------------------------------
        sentences = sent_tokenize(raw)

        for sent in sentences:

            text = normalize(sent)

            # ------------------------------------------------
            # Remove attribution-heavy fragments
            # ------------------------------------------------
            if is_attribution_heavy(text):
                continue

            # ------------------------------------------------
            # Keep only damage-semantic fragments
            # ------------------------------------------------
            if not has_damage_semantics(text):
                continue

            # ------------------------------------------------
            # Assign semantic bucket
            # ------------------------------------------------
            bucket = assign_bucket(text)

            record = {
                "text": text,
                "damage_bucket": bucket,
                "source_event": event,
            }

            out_f.write(
                json.dumps(record) + "\n"
            )


# ============================================================
# Completion Logs
# ============================================================
print("Filtering complete.")
print(f"Saved to {OUT_FILE}")