#!/usr/bin/env python3
"""
build_positive_pools_single.py
==============================

Project 2A — Single-Positive Supervision Builder
------------------------------------------------

Builds intentionally constrained single-positive supervision
pools for Ablation B experiments.

Pipeline Role
-------------
positive_pools/pools.json
    ↓
build_positive_pools_single.py
    ↓
positive_pools/pools_single.json
    ↓
single-positive contrastive training
    ↓
Ablation B evaluation

Purpose
-------
This script converts multi-positive semantic supervision pools
into intentionally weak single-positive supervision pools.

Unlike:
    build_positive_pools.py

which builds:
    rich semantic neighborhoods

this script keeps:
    only ONE positive sample per semantic bucket.

Goal
----
The purpose is diagnostic:
    evaluate the importance of multi-positive supervision.

This creates a deliberately brittle supervision setup
for controlled ablation experiments.

Single-Positive Policy
----------------------
For each semantic bucket:

1. Prefer ONE text positive if available

2. Else fallback to ONE Whisper positive

3. Else drop the bucket entirely

This intentionally reduces:
    semantic diversity
    supervision density
    cross-modal richness

Training Impact
---------------
Compared to multi-positive supervision:

single-positive supervision:
    - weakens semantic neighborhoods
    - reduces representation diversity
    - increases supervision brittleness
    - tests contrastive learning robustness

Input
-----
data/positive_pools/pools.json

Example:
{
    "flooding": {
        "text": [0, 3, 7],
        "whisper": [2, 5]
    }
}

Output
------
data/positive_pools/pools_single.json

Example:
{
    "flooding": {
        "text": [0],
        "whisper": []
    }
}

Important Design Notes
----------------------
- intentionally weak supervision
- deterministic positive selection
- designed specifically for ablation studies
- preserves modality-awareness
- cross-event semantics still preserved

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/build_positive_pools_single.py
"""

import json

from pathlib import Path


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path(
    "data/positive_pools/pools.json"
)

OUTPUT_PATH = Path(
    "data/positive_pools/pools_single.json"
)


# ============================================================
# Main
# ============================================================
def main():
    """
    Build intentionally constrained single-positive pools.

    Policy:
        1. Prefer one text positive
        2. Else fallback to one Whisper positive
        3. Else drop bucket

    Goal:
        create brittle supervision for diagnostic comparison.
    """

    # --------------------------------------------------------
    # Load multi-positive supervision pools
    # --------------------------------------------------------
    with open(INPUT_PATH, "r") as f:

        pools = json.load(f)

    # --------------------------------------------------------
    # Final single-positive pools
    # --------------------------------------------------------
    single_pools = {}

    # --------------------------------------------------------
    # Buckets with no valid supervision
    # --------------------------------------------------------
    dropped = []

    # --------------------------------------------------------
    # Process semantic buckets
    # --------------------------------------------------------
    for bucket, group in pools.items():

        text_idxs = group.get("text", [])

        whisper_idxs = group.get("whisper", [])

        # ====================================================
        # Policy 1:
        # Prefer ONE text positive
        # ====================================================
        if len(text_idxs) > 0:

            single_pools[bucket] = {

                "text": [text_idxs[0]],

                "whisper": []
            }

            print(
                f"[OK] {bucket}: "
                f"using 1 TEXT positive"
            )

        # ====================================================
        # Policy 2:
        # Fallback to ONE Whisper positive
        # ====================================================
        elif len(whisper_idxs) > 0:

            single_pools[bucket] = {

                "text": [],

                "whisper": [whisper_idxs[0]]
            }

            print(
                f"[OK] {bucket}: "
                f"using 1 WHISPER positive"
            )

        # ====================================================
        # Policy 3:
        # Drop unsupervised bucket
        # ====================================================
        else:

            dropped.append(bucket)

            print(
                f"[DROP] {bucket}: "
                f"no positives available"
            )

    # ========================================================
    # Save Single-Positive Pools
    # ========================================================
    with open(OUTPUT_PATH, "w") as f:

        json.dump(
            single_pools,
            f,
            indent=2
        )

    # ========================================================
    # Summary
    # ========================================================
    print("\n=== Summary ===")

    print(f"Buckets kept:   {len(single_pools)}")

    print(f"Buckets dropped: {len(dropped)}")

    if dropped:

        print("Dropped buckets:", dropped)

    print(
        f"\n Saved single-positive pools → "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":

    main()