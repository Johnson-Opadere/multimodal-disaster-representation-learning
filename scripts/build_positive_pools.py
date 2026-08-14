#!/usr/bin/env python3
"""
build_positive_pools.py
=======================

Project 2A — Multi-Positive Supervision Builder
------------------------------------------------

Builds semantic positive pools used for multimodal
contrastive retrieval training.

Pipeline Role
-------------
text_filtered/
    ↓
build_text_tokens.py
    ↓
text_tokens/index.json

whisper_filtered/
    ↓
build_whisper_tokens.py
    ↓
whisper_tokens/index.json

text_tokens + whisper_tokens
    ↓
build_positive_pools.py
    ↓
positive_pools/pools.json
    ↓
train_encoder.py

Purpose
-------
This script constructs semantic positive pools for
multi-positive contrastive learning.

Instead of training with:
    one positive pair only

the system builds:
    semantic positive groups

organized by:
    damage_bucket

Example
-------
All fragments belonging to:
    flooding

become mutual semantic positives,
even if they originate from:
    - different disaster events
    - different modalities
    - different wording styles

This encourages:
    - cross-event semantic alignment
    - modality generalization
    - retrieval robustness

Input
-----
data/text_tokens/index.json

data/whisper_tokens/index.json

Each record contains:
{
    "idx": ...,
    "damage_bucket": "...",
    "source_event": "...",
    "text": "..."
}

Output
------
data/positive_pools/

    pools.json
        semantic positive pools

    stats.json
        bucket statistics

Output Example
--------------
{
    "flooding": {
        "text": [0, 3, 7, ...],
        "whisper": [1, 5, 9, ...]
    }
}

Important Design Notes
----------------------
- pools are bucket-based
- positives are cross-event
- text and Whisper remain modality-aware
- sparse modalities are allowed
- deterministic pool construction

Training Impact
---------------
These pools enable:
    multi-positive contrastive learning

instead of:
    single-pair supervision

This significantly improves:
    semantic retrieval generalization.

Run Command
-----------
PYTHONPATH=2A python3 2A/scripts/build_positive_pools.py
"""

import os
import json

from collections import defaultdict


# ============================================================
# Paths
# ============================================================

TEXT_INDEX = "data/text_tokens/index.json"

WHISPER_INDEX = "data/whisper_tokens/index.json"

OUT_DIR = "data/positive_pools"

POOLS_OUT = os.path.join(
    OUT_DIR,
    "pools.json"
)

STATS_OUT = os.path.join(
    OUT_DIR,
    "stats.json"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# Load Token Metadata Indices
# ============================================================
#
# These indices preserve alignment between:
#   - token tensors
#   - semantic metadata
#
# ============================================================

with open(TEXT_INDEX, "r") as f:

    text_index = json.load(f)

with open(WHISPER_INDEX, "r") as f:

    whisper_index = json.load(f)


# ============================================================
# Build Semantic Positive Pools
# ============================================================
#
# Structure:
#
# pools[bucket]["text"]
# pools[bucket]["whisper"]
#
# Each bucket groups semantically related fragments
# across:
#   - events
#   - modalities
#   - phrasing styles
#
# ============================================================

pools = defaultdict(

    lambda: {
        "text": [],
        "whisper": []
    }
)


# ============================================================
# Add Text Supervision
# ============================================================

for rec in text_index:

    bucket = rec["damage_bucket"]

    pools[bucket]["text"].append(
        rec["idx"]
    )


# ============================================================
# Add Whisper Supervision
# ============================================================

for rec in whisper_index:

    bucket = rec["damage_bucket"]

    pools[bucket]["whisper"].append(
        rec["idx"]
    )


# ============================================================
# Filter Weak / Empty Buckets
# ============================================================
#
# A semantic bucket is retained if:
#   at least one modality contains positives.
#
# This supports:
#   sparse multi-positive supervision.
#
# ============================================================

final_pools = {}

stats = {}

for bucket, entries in pools.items():

    t = len(entries["text"])

    w = len(entries["whisper"])

    # --------------------------------------------------------
    # Skip fully empty buckets
    # --------------------------------------------------------
    if t == 0 and w == 0:
        continue

    final_pools[bucket] = entries

    # --------------------------------------------------------
    # Bucket statistics
    # --------------------------------------------------------
    stats[bucket] = {

        "num_text": t,

        "num_whisper": w,

        "total": t + w
    }


# ============================================================
# Save Positive Pools
# ============================================================

with open(POOLS_OUT, "w") as f:

    json.dump(
        final_pools,
        f,
        indent=2
    )


# ============================================================
# Save Pool Statistics
# ============================================================

summary = {

    "num_buckets": len(final_pools),

    "buckets": stats
}

with open(STATS_OUT, "w") as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# Logging
# ============================================================

print("Positive pool construction complete.")

print(f"Saved pools → {POOLS_OUT}")

print(f"Saved stats → {STATS_OUT}")