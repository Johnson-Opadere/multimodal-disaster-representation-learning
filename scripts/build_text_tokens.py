#!/usr/bin/env python3
"""
build_text_tokens.py
====================

Project 2A — Text Tokenization Pipeline
---------------------------------------

Tokenizes semantically filtered disaster-report fragments using
DistilBERT and stores token tensors for multimodal contrastive
training.

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
text_tokens/
    ↓
train_encoder.py

Purpose
-------
This script converts semantic disaster text fragments into
fixed-length transformer token tensors suitable for:

    - text embedding generation
    - multimodal alignment
    - contrastive learning
    - retrieval supervision

Each fragment is tokenized using:
    DistilBERT tokenizer

The resulting tensors become:
    the language-side inputs for Project 2A.

Input
-----
data/text_filtered/fragments.jsonl

Expected JSONL format:

{
    "text": "...",
    "damage_bucket": "...",
    "source_event": "..."
}

Output
------
data/text_tokens/

    tokens.pt
        PyTorch tensors:
            input_ids
            attention_mask

    index.json
        metadata aligned with token tensor rows

    stats.json
        dataset/tokenization statistics

Important Design Notes
----------------------
- tokenizer is frozen to:
      distilbert-base-uncased

- all sequences padded/truncated to:
      MAX_LEN = 128

- degenerate fragments are removed

- token tensors remain perfectly aligned with:
      index.json metadata

- deterministic ordering preserved

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/build_text_tokens.py

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

IN_FILE = "data/text_filtered/fragments.jsonl"

OUT_DIR = "data/text_tokens"

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
# v10 canonical configuration
#
# DistilBERT chosen for:
#   - lightweight inference
#   - strong semantic embeddings
#   - stable transformer baseline
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
# Token tensors
# ------------------------------------------------------------
input_ids_list = []

attention_mask_list = []

# ------------------------------------------------------------
# Metadata aligned with token tensors
# ------------------------------------------------------------
index = []

# ------------------------------------------------------------
# Semantic bucket statistics
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
        # Drop degenerate fragments
        #
        # Example:
        #   nearly empty sentences
        #   tokenization artifacts
        #
        # Helps stabilize:
        #   embedding quality
        #   contrastive learning
        # ----------------------------------------------------
        if attention_mask.sum().item() < MIN_NONPAD_TOKENS:

            dropped += 1

            continue

        # ----------------------------------------------------
        # Tensor row index
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
        # Bucket distribution stats
        # ----------------------------------------------------
        bucket_counts[bucket] = (
            bucket_counts.get(bucket, 0) + 1
        )


# ============================================================
# Stack Token Tensors
# ============================================================
#
# Final shapes:
#
#   input_ids        -> (N, L)
#   attention_mask   -> (N, L)
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
# Save Dataset Statistics
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

print("Text token building (PyTorch) complete.")

print(f"Saved tokens → {TOKENS_OUT}")

print(f"Saved index  → {INDEX_OUT}")

print(f"Dropped fragments: {dropped}")