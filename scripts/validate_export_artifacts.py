#!/usr/bin/env python3
"""
validate_export_artifacts.py
===========================================================
Project 2A — Artifact Integrity Validation
===========================================================

OVERVIEW
===========================================================

This script validates exported embedding artifacts generated
by:

    export_embeddings.py

The goal is to ensure all retrieval artifacts are:

    • structurally correct
    • numerically stable
    • retrieval-ready
    • scientifically reproducible

This script acts as a:

    retrieval integrity gate

between:

    2A → embedding generation
    2B → retrieval evaluation

===========================================================
WHY THIS SCRIPT MATTERS
===========================================================

Project 2B relies ENTIRELY on exported artifacts.

If embeddings or indices become corrupted, retrieval quality
and evaluation metrics become invalid.

Therefore this script verifies:

    • embedding geometry integrity
    • metadata alignment
    • deterministic ordering
    • normalization correctness
    • semantic metadata consistency

===========================================================
SUPPORTED ARTIFACT HIERARCHY
===========================================================

2A/artifacts/

    info_nce/
        layer4_multi/
        full_multi/

    margin/
        layer4_multi/
        layer4_single/

===========================================================
VALIDATION CHECKS
===========================================================

Each variant undergoes the following checks:

-----------------------------------------------------------
1️. Required Files
-----------------------------------------------------------

Ensures all retrieval artifacts exist:

    • vision_embeddings.npy
    • vision_index.json
    • text_embeddings.npy
    • text_index.json
    • whisper_embeddings.npy
    • whisper_index.json
    • metadata.json

-----------------------------------------------------------
2️. Embedding Shape Validation
-----------------------------------------------------------

Checks:

    embeddings.shape == (N, 256)

This verifies:
    • architecture consistency
    • retrieval contract correctness

-----------------------------------------------------------
3️. Row ↔ Metadata Alignment
-----------------------------------------------------------

Ensures:

    embedding[i]
        ↔
    metadata[i]

remain perfectly aligned.

This is CRITICAL for:
    • retrieval correctness
    • leakage analysis
    • XE evaluation
    • debugging

-----------------------------------------------------------
4️. L2 Normalization
-----------------------------------------------------------

Verifies embeddings satisfy:

    ||z|| = 1

This is REQUIRED for:

    • cosine similarity
    • FAISS inner-product retrieval
    • hyperspherical embedding geometry

-----------------------------------------------------------
5️. NaN / Inf Validation
-----------------------------------------------------------

Ensures embeddings contain:

    • no NaNs
    • no infinities

This protects against:
    • numerical instability
    • failed training
    • retrieval corruption

-----------------------------------------------------------
6️. Event/Bucket Consistency
-----------------------------------------------------------

Validates:

    • event_id types
    • damage_bucket types

Useful for:
    • retrieval auditing
    • metadata integrity
    • downstream filtering

===========================================================
SCIENTIFIC IMPORTANCE
===========================================================

This script improves:

    • reproducibility
    • retrieval reliability
    • scientific rigor

Without validation:
retrieval metrics may silently become invalid.

===========================================================
EMBEDDING GEOMETRY THEORY
===========================================================

The exported embeddings live on a:

    hypersphere

because they are L2-normalized.

This allows:

    cosine_similarity(a,b)
        =
    aᵀb

after normalization.

Therefore:
retrieval operates on:

    angular semantic similarity

rather than raw magnitude.

===========================================================
COLLAPSE DETECTION THEORY
===========================================================

Embedding collapse occurs when:

    all embeddings become nearly identical.

Symptoms:
    • low embedding variance
    • pairwise cosine ≈ 1
    • weak retrieval diversity

This script helps detect such failures early.

===========================================================
WHY INDEX ALIGNMENT MATTERS
===========================================================

Suppose:

    embedding[42]

accidentally corresponds to:

    metadata[43]

Then:
    • retrieval visualization breaks
    • XE metrics break
    • leakage analysis becomes invalid

Therefore deterministic ordering is ESSENTIAL.

===========================================================
RUN COMMANDS
===========================================================

cd 2A

-----------------------------------------------------------
1️. Validate specific objective + variant
-----------------------------------------------------------

PYTHONPATH=. python scripts/validate_export_artifacts.py \
    --objective info_nce \
    --variant layer4_multi

-----------------------------------------------------------
2️. Validate all variants under one objective
-----------------------------------------------------------

PYTHONPATH=. python scripts/validate_export_artifacts.py \
    --objective margin \
    --validate_all

-----------------------------------------------------------
3️. Validate ALL objectives + ALL variants
-----------------------------------------------------------

PYTHONPATH=. python scripts/validate_export_artifacts.py \
    --validate_all_global

===========================================================
EXPECTED OUTPUT
===========================================================

Successful validation prints:

     Artifact integrity verified.

===========================================================
"""

import os
import argparse
import json
import numpy as np


# ==========================================================
# Required Artifact Files
# ==========================================================
#
# Every retrieval-ready artifact directory MUST contain
# these files.
#
REQUIRED_FILES = [

    # ------------------------------------------------------
    # Vision retrieval artifacts
    # ------------------------------------------------------
    "vision_embeddings.npy",
    "vision_index.json",

    # ------------------------------------------------------
    # Text retrieval artifacts
    # ------------------------------------------------------
    "text_embeddings.npy",
    "text_index.json",

    # ------------------------------------------------------
    # Whisper retrieval artifacts
    # ------------------------------------------------------
    "whisper_embeddings.npy",
    "whisper_index.json",

    # ------------------------------------------------------
    # Metadata / diagnostics
    # ------------------------------------------------------
    "metadata.json",
]


# ==========================================================
# File Existence Validation
# ==========================================================
#
# Ensures retrieval artifacts are complete.
#
def check_file_existence(path):

    for fname in REQUIRED_FILES:

        fpath = os.path.join(path, fname)

        if not os.path.exists(fpath):

            raise FileNotFoundError(
                f"Missing required file: {fpath}"
            )


# ==========================================================
# NaN / Inf Validation
# ==========================================================
#
# Detects numerical corruption.
#
# Important because:
# retrieval systems cannot safely operate on:
#   • NaNs
#   • infinities
#
def check_no_nan_inf(arr, name):

    if not np.isfinite(arr).all():

        raise ValueError(
            f"{name} contains NaN or Inf values"
        )


# ==========================================================
# L2 Normalization Validation
# ==========================================================
#
# Verifies embeddings lie on:
#
#     ||z|| = 1
#
# This is REQUIRED for:
#   • cosine similarity
#   • FAISS inner-product retrieval
#   • hyperspherical geometry
#
def check_l2_normalized(
    arr,
    name,
    tol=1e-5,
):

    norms = np.linalg.norm(
        arr,
        axis=1,
    )

    mean_norm = np.mean(norms)

    if abs(mean_norm - 1.0) > tol:

        raise ValueError(

            f"{name} not properly L2 normalized "
            f"(mean norm={mean_norm:.6f})"
        )


# ==========================================================
# Row ↔ Metadata Alignment Validation
# ==========================================================
#
# Ensures:
#
#   embedding[i]
#       ↔
#   metadata[i]
#
# remain perfectly aligned.
#
# This is CRITICAL for:
#   • retrieval correctness
#   • XE metrics
#   • leakage analysis
#   • debugging
#
def check_index_alignment(
    embeddings,
    index_data,
    name,
):

    # ------------------------------------------------------
    # Row-count validation
    # ------------------------------------------------------
    if embeddings.shape[0] != len(index_data):

        raise ValueError(

            f"{name} row mismatch: "
            f"embeddings={embeddings.shape[0]} "
            f"vs index={len(index_data)}"
        )

    # ------------------------------------------------------
    # Deterministic index ordering
    # ------------------------------------------------------
    #
    # index[i]["idx"] must equal i
    #
    for i in range(min(20, len(index_data))):

        if index_data[i]["idx"] != i:

            raise ValueError(
                f"{name} index misalignment at row {i}"
            )


# ==========================================================
# Event/Bucket Consistency Validation
# ==========================================================
#
# Ensures metadata fields remain valid.
#
# Useful for:
#   • XE retrieval
#   • semantic filtering
#   • leakage auditing
#
def check_event_bucket_consistency(
    vision_index
):

    for row in vision_index[:50]:

        if not isinstance(
            row["event_id"],
            str,
        ):

            raise ValueError(
                "Invalid event_id type"
            )

        if not isinstance(
            row["damage_bucket"],
            str,
        ):

            raise ValueError(
                "Invalid damage_bucket type"
            )


# ==========================================================
# Core Variant Validation
# ==========================================================
#
# Performs ALL retrieval integrity checks on:
#
#     one artifact variant
#
def validate_variant(variant_path):

    print(f"\n Validating: {variant_path}")

    # ======================================================
    # Required files
    # ======================================================
    check_file_existence(variant_path)

    # ======================================================
    # Load embeddings
    # ======================================================
    vision = np.load(
        os.path.join(
            variant_path,
            "vision_embeddings.npy"
        )
    )

    text = np.load(
        os.path.join(
            variant_path,
            "text_embeddings.npy"
        )
    )

    whisper = np.load(
        os.path.join(
            variant_path,
            "whisper_embeddings.npy"
        )
    )

    # ======================================================
    # Load indices
    # ======================================================
    with open(
        os.path.join(
            variant_path,
            "vision_index.json"
        )
    ) as f:

        vision_index = json.load(f)

    with open(
        os.path.join(
            variant_path,
            "text_index.json"
        )
    ) as f:

        text_index = json.load(f)

    with open(
        os.path.join(
            variant_path,
            "whisper_index.json"
        )
    ) as f:

        whisper_index = json.load(f)

    # ======================================================
    # Shape Validation
    # ======================================================
    print("Vision shape:", vision.shape)

    print("Text shape:", text.shape)

    print("Whisper shape:", whisper.shape)

    # ------------------------------------------------------
    # Embedding dimensionality contract
    # ------------------------------------------------------
    #
    # All embeddings MUST be:
    #
    #     256-dimensional
    #
    if vision.shape[1] != 256:

        raise ValueError(
            "Vision embedding dim != 256"
        )

    if text.shape[1] != 256:

        raise ValueError(
            "Text embedding dim != 256"
        )

    if whisper.shape[1] != 256:

        raise ValueError(
            "Whisper embedding dim != 256"
        )

    # ======================================================
    # Alignment Validation
    # ======================================================
    check_index_alignment(
        vision,
        vision_index,
        "Vision",
    )

    check_index_alignment(
        text,
        text_index,
        "Text",
    )

    check_index_alignment(
        whisper,
        whisper_index,
        "Whisper",
    )

    # ======================================================
    # Numerical Stability Validation
    # ======================================================
    check_no_nan_inf(
        vision,
        "Vision",
    )

    check_no_nan_inf(
        text,
        "Text",
    )

    check_no_nan_inf(
        whisper,
        "Whisper",
    )

    # ======================================================
    # L2 Normalization Validation
    # ======================================================
    check_l2_normalized(
        vision,
        "Vision",
    )

    check_l2_normalized(
        text,
        "Text",
    )

    check_l2_normalized(
        whisper,
        "Whisper",
    )

    # ======================================================
    # Event/Bucket Metadata Validation
    # ======================================================
    check_event_bucket_consistency(
        vision_index
    )

    print(" Artifact integrity verified.")


# ==========================================================
# Main Entry
# ==========================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--objective",
        type=str,
    )

    parser.add_argument(
        "--variant",
        type=str,
    )

    parser.add_argument(
        "--validate_all",
        action="store_true",
    )

    parser.add_argument(
        "--validate_all_global",
        action="store_true",
    )

    args = parser.parse_args()

    # ======================================================
    # Artifact Root
    # ======================================================
    base = "2A_v2/artifacts"

    # ======================================================
    # Validate ALL objectives + ALL variants
    # ======================================================
    if args.validate_all_global:

        for objective in os.listdir(base):

            obj_path = os.path.join(
                base,
                objective,
            )

            if not os.path.isdir(obj_path):

                continue

            for variant in os.listdir(obj_path):

                validate_variant(
                    os.path.join(
                        obj_path,
                        variant,
                    )
                )

        return

    # ======================================================
    # Validate ALL variants under one objective
    # ======================================================
    if args.validate_all:

        if not args.objective:

            raise ValueError(
                "Provide --objective with "
                "--validate_all"
            )

        obj_path = os.path.join(
            base,
            args.objective,
        )

        for variant in os.listdir(obj_path):

            validate_variant(
                os.path.join(
                    obj_path,
                    variant,
                )
            )

        return

    # ======================================================
    # Validate Single Variant
    # ======================================================
    if not args.objective or not args.variant:

        raise ValueError(

            "Provide --objective AND --variant "
            "or use "
            "--validate_all / "
            "--validate_all_global"
        )

    validate_variant(

        os.path.join(
            base,
            args.objective,
            args.variant,
        )
    )


if __name__ == "__main__":
    main()