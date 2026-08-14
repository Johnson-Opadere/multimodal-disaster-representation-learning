#!/usr/bin/env python3
"""
build_whisper_tokens.py
=======================

Project 2A — Whisper Tokenization Pipeline
------------------------------------------

Tokenizes semantically filtered Whisper / ASR disaster fragments
using DistilBERT and stores transformer-ready tensors for
multimodal contrastive retrieval training.

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
whisper_tokens/
    ↓
train_encoder.py

Purpose
-------
This script converts spoken disaster-semantic fragments into
fixed-length transformer token tensors suitable for:

    - spoken-language embedding generation
    - multimodal semantic alignment
    - contrastive retrieval learning
    - cross-modal supervision

The resulting tensors become:
    the Whisper-language supervision inputs for Project 2A.

Input
-----
data/whisper_filtered/fragments.jsonl

Expected JSONL format:

{
    "text": "...",
    "damage_bucket": "...",
    "source_event": "..."
}

Output
------
data/whisper_tokens/

    tokens.pt
        PyTorch tensors:
            input_ids
            attention_mask

    index.json
        metadata aligned with token tensors

    stats.json
        tokenization statistics

Important Design Notes
----------------------
- tokenizer locked to:
      distilbert-base-uncased

- fixed token length:
      MAX_LEN = 128

- degenerate ASR fragments removed

- token tensors remain aligned with:
      index.json metadata

- deterministic ordering preserved

- Whisper supervision intentionally separated
  from report-text supervision

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/build_whisper_tokens.py

Dependencies
------------
pip install transformers torch
"""

import os
import json
import torch

from pathlib import Path
from transformers import DistilBertTokenizerFast


# ============================================================
# Paths
# ============================================================

IN_FILE = "data/whisper_filtered/fragments.jsonl"

OUT_DIR = "data/whisper_tokens"

TOKENS_OUT = os.path.join(
    OUT_DIR,
    "tokens.pt"
)

INDEX_OUT = os.path.join(
    OUT_DIR,
    "index.json"
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
# Tokenizer Configuration
# ============================================================
#
# v10 canonical tokenizer setup.
#
# DistilBERT selected for:
#   - lightweight transformer inference
#   - efficient semantic encoding
#   - strong pretrained language representations
#
# ============================================================

TOKENIZER_NAME = "distilbert-base-uncased"

MAX_LEN = 128

MIN_NONPAD_TOKENS = 3

tokenizer = DistilBertTokenizerFast.from_pretrained(
    TOKENIZER_NAME
)


# ============================================================
# Storage
# ============================================================

# ------------------------------------------------------------
# Transformer token tensors
# ------------------------------------------------------------
input_ids_list = []

attention_mask_list = []

# ------------------------------------------------------------
# Metadata aligned with tensor rows
# ------------------------------------------------------------
index = []

# ------------------------------------------------------------
# Semantic bucket distribution
# ------------------------------------------------------------
bucket_counts = {}

# ------------------------------------------------------------
# Degenerate fragment counter
# ------------------------------------------------------------
dropped = 0


# ============================================================
# Main Tokenization Loop
# ============================================================

with open(IN_FILE, "r") as f:

    for line in f:

        rec = json.loads(line)

        text = rec["text"]

        bucket = rec["damage_bucket"]

        source_event = rec["source_event"]

        # ----------------------------------------------------
        # DistilBERT tokenization
        # ----------------------------------------------------
        enc = tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=MAX_LEN,

            return_attention_mask=True,

            return_tensors="pt"
        )

        # ----------------------------------------------------
        # Remove batch dimension
        #
        # Shape:
        #   (1, L)
        #       →
        #   (L,)
        # ----------------------------------------------------
        input_ids = enc["input_ids"].squeeze(0)

        attention_mask = enc["attention_mask"].squeeze(0)

        # ----------------------------------------------------
        # Remove semantically weak fragments
        #
        # Important for:
        #   - stable embeddings
        #   - contrastive learning quality
        #   - semantic supervision density
        # ----------------------------------------------------
        if attention_mask.sum().item() < MIN_NONPAD_TOKENS:

            dropped += 1

            continue

        # ----------------------------------------------------
        # Stable tensor row index
        # ----------------------------------------------------
        idx = len(input_ids_list)

        # ----------------------------------------------------
        # Store token tensors
        # ----------------------------------------------------
        input_ids_list.append(input_ids)

        attention_mask_list.append(attention_mask)

        # ----------------------------------------------------
        # Store aligned metadata
        # ----------------------------------------------------
        index.append({

            "idx": idx,

            "damage_bucket": bucket,

            "source_event": source_event,

            "text": text
        })

        # ----------------------------------------------------
        # Semantic bucket statistics
        # ----------------------------------------------------
        bucket_counts[bucket] = (
            bucket_counts.get(bucket, 0) + 1
        )


# ============================================================
# Stack Token Tensors
# ============================================================
#
# Final tensor shapes:
#
#   input_ids       -> (N, L)
#   attention_mask  -> (N, L)
#
# ============================================================

input_ids_tensor = torch.stack(
    input_ids_list
)

attention_mask_tensor = torch.stack(
    attention_mask_list
)


# ============================================================
# Save Token Tensors
# ============================================================

torch.save(
    {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor
    },
    TOKENS_OUT
)


# ============================================================
# Save Metadata Index
# ============================================================

with open(INDEX_OUT, "w") as f:

    json.dump(
        index,
        f,
        indent=2
    )


# ============================================================
# Save Tokenization Statistics
# ============================================================

stats = {

    "num_tokens": len(index),

    "dropped_fragments": dropped,

    "bucket_distribution": bucket_counts,

    "max_len": MAX_LEN,

    "tokenizer": TOKENIZER_NAME,

    "framework": "pytorch"
}

with open(STATS_OUT, "w") as f:

    json.dump(
        stats,
        f,
        indent=2
    )


# ============================================================
# Logging
# ============================================================

print("Whisper token building (PyTorch) complete.")

print(f"Saved tokens → {TOKENS_OUT}")

print(f"Saved index  → {INDEX_OUT}")

print(f"Dropped fragments: {dropped}")