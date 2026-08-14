#!/usr/bin/env python3
"""
filter_whisper.py
=================

Project 2A — Whisper Semantic Filtering Pipeline
------------------------------------------------

Filters Whisper / ASR disaster transcripts into
damage-semantic fragments for multimodal retrieval training.

Pipeline Role
-------------
raw_whisper/
    ↓
filter_whisper.py
    ↓
whisper_filtered/fragments.jsonl
    ↓
build_whisper_tokens.py
    ↓
tokens.pt
    ↓
multimodal contrastive training

Purpose
-------
Whisper / ASR transcripts are significantly noisier than
written reports and often contain:

    - journalism narration
    - attribution-heavy speech
    - interview artifacts
    - conversational filler
    - incomplete ASR fragments
    - speaker transitions
    - non-semantic broadcast language

This script converts noisy spoken disaster transcripts into:

    structured semantic supervision signals

used for:
    - multimodal alignment
    - contrastive retrieval training
    - semantic embedding learning
    - cross-event retrieval evaluation

Behavior
--------
- deterministic sentence splitting
- lightweight normalization
- aggressive ASR / journalism noise filtering
- keeps only damage-semantic fragments
- assigns semantic damage buckets
- preserves source event metadata

Output
------
data/whisper_filtered/fragments.jsonl

Each line:
{
    "text": "...",
    "damage_bucket": "...",
    "source_event": "..."
}

Important Design Notes
----------------------
- filtering is intentionally deterministic
- ASR transcripts require stronger filtering than reports
- semantic grounding requires:
      damage outcome
      +
      affected physical object
- structural_damage is intentionally cross-event
- metadata alignment preserved for downstream retrieval

Run Command
-----------
PYTHONPATH=. python3 2A_v2/scripts/filter_whisper.py
"""

import os
import re
import json

from pathlib import Path


# ============================================================
# Paths
# ============================================================

RAW_WHISPER_DIR = "data/raw_whisper"

OUT_DIR = "data/whisper_filtered"

OUT_FILE = os.path.join(
    OUT_DIR,
    "fragments.jsonl"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# Damage Semantics
# ============================================================
#
# Outcome/destruction language.
#
# These keywords capture:
#   - physical destruction
#   - structural collapse
#   - inundation
#   - fire damage
#
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


# ============================================================
# Grounded Physical Objects
# ============================================================
#
# Ensures semantic grounding.
#
# Example:
#   GOOD:
#       "homes were destroyed"
#
#   BAD:
#       "everything was destroyed"
#
# ============================================================

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

    "power",

    "water",

    "utilities",
}


# ============================================================
# Semantic Bucket Refinement
# ============================================================
#
# Weak semantic supervision categories.
#
# Important:
#   structural_damage is intentionally cross-event.
#
# Goal:
#   encourage semantic generalization
#   instead of event memorization.
#
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
    #
    # Unified bucket for:
    #   - earthquake collapse
    #   - rubble
    #   - structural destruction
    # --------------------------------------------------------
    "structural_damage": {

        "earthquake",

        "seismic",

        "collapsed",

        "rubble",

        "uninhabitable",
    },

    # --------------------------------------------------------
    # Infrastructure damage
    # --------------------------------------------------------
    "infrastructure_damage": {

        "roads",

        "bridges",

        "power",

        "water",

        "utilities",
    },
}


# ============================================================
# Whisper / ASR Noise Phrases
# ============================================================
#
# Removes:
#   - journalism narration
#   - interview structure
#   - attribution-heavy speech
#   - broadcast-style chatter
#
# Helps prevent:
#   semantic shortcut learning.
#
# ============================================================

WHISPER_NOISE_PHRASES = {

    "for npr news",

    "reporting from",

    "this is",

    "said",

    "told",

    "according to",

    "we spoke to",

    "i spoke with",

    "officials say",

    "authorities say",

    "the government",

    "president",

    "minister",

    "reporter",

    "news",

    "interview",

    "host",

    "anchor",
}


# ============================================================
# Sentence Splitter
# ============================================================
def sent_tokenize(text):
    """
    Lightweight deterministic sentence splitter.

    Uses regex instead of external NLP dependencies
    to preserve:
        - reproducibility
        - simplicity
        - deterministic behavior
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

    Goal:
        reduce lexical noise
        while preserving semantic content.
    """

    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Whisper Noise Filtering
# ============================================================
def is_noise(text):
    """
    Remove journalism-heavy / ASR-heavy fragments.

    Examples:
        - "officials said"
        - "reporting from"
        - "according to"
    """

    return any(
        p in text
        for p in WHISPER_NOISE_PHRASES
    )


# ============================================================
# Damage Semantic Filtering
# ============================================================
def has_damage_semantics(text):
    """
    Keep only semantically grounded damage fragments.

    Requirement:
        damage outcome
        +
        affected physical object

    Helps improve:
        - retrieval grounding
        - semantic density
        - multimodal alignment quality
    """

    return (

        any(
            k in text
            for k in DAMAGE_OUTCOME_KEYWORDS
        )

        and

        any(
            o in text
            for o in AFFECTED_OBJECTS
        )
    )


# ============================================================
# Semantic Bucket Assignment
# ============================================================
def assign_bucket(text):
    """
    Assign semantic damage category.

    Returns:
        bucket name
        OR
        generic_damage
    """

    for bucket, kws in EVENT_REFINERS.items():

        if any(k in text for k in kws):

            return bucket

    return "generic_damage"


# ============================================================
# Short Fragment Removal
# ============================================================
def too_short(text, min_words=5):
    """
    Remove extremely short ASR fragments.

    Helps prevent:
        - weak supervision
        - unstable contrastive anchors
        - low-information embeddings
    """

    return len(text.split()) < min_words


# ============================================================
# Main Filtering Loop
# ============================================================

with open(OUT_FILE, "w") as out_f:

    # --------------------------------------------------------
    # Iterate through raw Whisper transcripts
    # --------------------------------------------------------
    for fname in os.listdir(RAW_WHISPER_DIR):

        if not fname.endswith(".txt"):
            continue

        event = fname.replace(".txt", "")

        path = os.path.join(
            RAW_WHISPER_DIR,
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

        # ----------------------------------------------------
        # Process each sentence fragment
        # ----------------------------------------------------
        for sent in sentences:

            text = normalize(sent)

            # ------------------------------------------------
            # Remove very short fragments
            # ------------------------------------------------
            if too_short(text):
                continue

            # ------------------------------------------------
            # Remove ASR / journalism-heavy noise
            # ------------------------------------------------
            if is_noise(text):
                continue

            # ------------------------------------------------
            # Keep only grounded damage semantics
            # ------------------------------------------------
            if not has_damage_semantics(text):
                continue

            # ------------------------------------------------
            # Assign semantic bucket
            # ------------------------------------------------
            bucket = assign_bucket(text)

            # ------------------------------------------------
            # Structured semantic supervision record
            # ------------------------------------------------
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

print("Whisper filtering complete.")

print(f"Saved to {OUT_FILE}")