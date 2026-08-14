#!/usr/bin/env python3
"""
diagnose_embedding_geometry.py
===========================================================
Project 2A — Embedding Geometry Diagnostics
===========================================================

OVERVIEW
===========================================================

This script analyzes the geometric structure of exported
vision embeddings produced by:

    export_embeddings.py

The goal is to diagnose whether the learned representation
space exhibits healthy semantic structure suitable for:

    • multimodal retrieval
    • semantic search
    • FAISS indexing
    • nearest-neighbor ranking
    • cross-event generalization

This script is part of the project's:

    representation diagnostics pipeline

===========================================================
WHY EMBEDDING GEOMETRY MATTERS
===========================================================

Contrastive learning does NOT merely optimize loss values.

It shapes:

    semantic embedding topology.

A low training loss does NOT guarantee:

    • semantic separation
    • retrieval quality
    • non-collapsed embeddings
    • meaningful multimodal geometry

Therefore:
embedding diagnostics are critical.

===========================================================
WHAT THIS SCRIPT ANALYZES
===========================================================

1️. Embedding Variance
-----------------------------------------------------------

Measures feature-space diversity.

Low variance may indicate:

    embedding collapse.

-----------------------------------------------------------

2️. Mean Cosine Similarity
-----------------------------------------------------------

Measures how similar embeddings are globally.

Very high mean cosine similarity suggests:

    embeddings becoming too similar.

-----------------------------------------------------------

3️. Intra-event vs Inter-event Similarity
-----------------------------------------------------------

Measures:

    similarity within same event
        vs
    similarity across different events.

Healthy embeddings should show:

    intra-event > inter-event

while still preserving:

    cross-event semantic generalization.

-----------------------------------------------------------

4️. Bucket Centroid Separation
-----------------------------------------------------------

Computes similarity between semantic bucket centroids.

Examples:
    • flooding
    • wildfire
    • structural_damage

If centroids are too similar:
semantic separation may be weak.

-----------------------------------------------------------

5️. Similarity Distribution Spread
-----------------------------------------------------------

Analyzes the spread of pairwise cosine similarities.

Healthy embedding spaces typically exhibit:

    meaningful similarity variance

rather than:
    uniformly similar embeddings.

===========================================================
SCIENTIFIC PURPOSE
===========================================================

This script helps diagnose:

    • embedding collapse
    • weak semantic structure
    • poor bucket separation
    • retrieval instability
    • over-clustered geometry

It improves:

    • scientific rigor
    • retrieval reliability
    • representation interpretability

===========================================================
IMPORTANT EMBEDDING THEORY
===========================================================

All exported embeddings are:

    L2-normalized

Therefore:
embeddings lie on a:

    hypersphere

This allows:

    cosine_similarity(a,b)
        =
    aᵀb

because:

    ||a|| = ||b|| = 1

===========================================================
CONTRASTIVE LEARNING THEORY
===========================================================

Contrastive learning aims to:

    pull positives together
    push negatives apart

This creates:

    semantic clusters
inside embedding space.

This script evaluates whether those clusters actually formed.

===========================================================
INTRA vs INTER EVENT THEORY
===========================================================

Intra-event similarity:
-----------------------------------------------------------

Measures similarity between samples from:

    SAME disaster event.

Inter-event similarity:
-----------------------------------------------------------

Measures similarity between samples from:

    DIFFERENT disaster events.

===========================================================
IMPORTANT PROJECT GOAL
===========================================================

This project aims for:

    semantic cross-event retrieval

NOT:
    event memorization.

Therefore:
some inter-event similarity is EXPECTED
for semantically related disasters.

Example:

    Harvey flooding
        ↔
    Midwest flooding

should remain semantically related.

===========================================================
CENTROID THEORY
===========================================================

A bucket centroid represents:

    the mean semantic direction
of a damage category.

Example:

    flooding centroid

represents:
average flooding semantics.

High centroid similarity between buckets may indicate:

    weak semantic separation.

===========================================================
EMBEDDING COLLAPSE THEORY
===========================================================

Embedding collapse occurs when:

    all embeddings become nearly identical.

Symptoms:
    • low variance
    • high mean cosine similarity
    • weak separation
    • poor retrieval diversity

This script helps detect collapse early.

===========================================================
RUN COMMANDS
===========================================================

cd 2A

-----------------------------------------------------------
1️. Analyze one variant
-----------------------------------------------------------

PYTHONPATH=. python scripts/diagnose_embedding_geometry.py \
    --variant layer4_multi

-----------------------------------------------------------
2️. Analyze all exported variants
-----------------------------------------------------------

PYTHONPATH=. python scripts/diagnose_embedding_geometry.py \
    --analyze_all

===========================================================
EXPECTED OUTPUT
===========================================================

Example metrics:

    Embedding variance
    Mean cosine similarity
    Intra-event similarity
    Inter-event similarity
    Centroid similarity

Interpretation warnings may also appear for:

    • collapse risk
    • weak separation
    • overly similar embeddings

===========================================================
"""

import os
import argparse
import json
import numpy as np


# ==========================================================
# Artifact Root
# ==========================================================
#
# Contains exported embedding variants:
#
# 2A/artifacts/
#     layer4_multi/
#     frozen_multi/
#     full_multi/
#
ARTIFACT_ROOT = "2A/artifacts"


# ==========================================================
# Load One Variant
# ==========================================================
#
# Loads:
#   • vision embeddings
#   • aligned metadata index
#
# IMPORTANT:
# embedding[i]
#     ↔
# index[i]
#
# must remain aligned.
#
def load_variant(path):

    V = np.load(
        os.path.join(
            path,
            "vision_embeddings.npy"
        )
    )

    with open(
        os.path.join(
            path,
            "vision_index.json"
        )
    ) as f:

        index = json.load(f)

    return V, index


# ==========================================================
# Cosine Similarity Matrix
# ==========================================================
#
# Since embeddings are L2-normalized:
#
#     cosine(a,b) = aᵀb
#
# Matrix multiplication computes all pairwise similarities.
#
def cosine_matrix(V):

    return V @ V.T


# ==========================================================
# Embedding Variance
# ==========================================================
#
# Computes average standard deviation across dimensions.
#
# Low variance may indicate:
#   • embedding collapse
#   • weak semantic diversity
#
def compute_variance(V):

    return np.mean(
        np.std(V, axis=0)
    )


# ==========================================================
# Global Cosine Similarity Statistics
# ==========================================================
#
# Measures:
#   • mean pairwise similarity
#   • similarity spread
#
# Excludes diagonal self-similarity.
#
def compute_mean_cosine(V):

    cos = cosine_matrix(V)

    n = cos.shape[0]

    # ------------------------------------------------------
    # Remove diagonal:
    # cosine(x,x) = 1
    # ------------------------------------------------------
    off_diag = cos[
        ~np.eye(n, dtype=bool)
    ]

    return (
        np.mean(off_diag),
        np.std(off_diag),
    )


# ==========================================================
# Intra-event vs Inter-event Similarity
# ==========================================================
#
# Measures:
#
#   SAME event similarity
#       vs
#   DIFFERENT event similarity
#
# Healthy embeddings should typically satisfy:
#
#     intra > inter
#
# while still allowing:
#   cross-event semantic generalization.
#
def intra_inter_similarity(
    V,
    index,
):

    cos = cosine_matrix(V)

    n = len(index)

    intra = []

    inter = []

    for i in range(n):

        for j in range(i + 1, n):

            # --------------------------------------------------
            # SAME event
            # --------------------------------------------------
            if (
                index[i]["event_id"]
                ==
                index[j]["event_id"]
            ):

                intra.append(
                    cos[i, j]
                )

            # --------------------------------------------------
            # DIFFERENT event
            # --------------------------------------------------
            else:

                inter.append(
                    cos[i, j]
                )

    return (

        np.mean(intra)
        if intra else 0.0,

        np.mean(inter)
        if inter else 0.0,
    )


# ==========================================================
# Semantic Bucket Centroid Separation
# ==========================================================
#
# Computes semantic bucket centroids:
#
#     centroid =
#     average embedding direction
#
# Then measures:
#   centroid-to-centroid cosine similarity.
#
# High centroid similarity may indicate:
#   weak semantic separation.
#
def centroid_separation(
    V,
    index,
):

    buckets = {}

    # ======================================================
    # Group embeddings by semantic bucket
    # ======================================================
    for i, row in enumerate(index):

        b = row["damage_bucket"]

        if b not in buckets:

            buckets[b] = []

        buckets[b].append(V[i])

    # ======================================================
    # Compute centroids
    # ======================================================
    centroids = {

        b: np.mean(
            np.vstack(vecs),
            axis=0
        )

        for b, vecs in buckets.items()
    }

    names = list(centroids.keys())

    sep = []

    # ======================================================
    # Pairwise centroid similarities
    # ======================================================
    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            c1 = centroids[names[i]]

            c2 = centroids[names[j]]

            #
            # Since embeddings are normalized:
            # dot product ≈ cosine similarity
            #
            sep.append(
                np.dot(c1, c2)
            )

    return (
        np.mean(sep)
        if sep else 0.0
    )


# ==========================================================
# Variant Analysis
# ==========================================================
#
# Runs all embedding geometry diagnostics on:
#
#     one exported artifact variant
#
def analyze_variant(path):

    print("\n====================================================")
    print(f"Analyzing: {path}")
    print("====================================================")

    # ======================================================
    # Load embeddings + metadata
    # ======================================================
    V, index = load_variant(path)

    # ======================================================
    # 1️. Embedding Variance
    # ======================================================
    var = compute_variance(V)

    print(
        f"Embedding variance "
        f"(mean std per dim): {var:.6f}"
    )

    # ======================================================
    # 2️. Cosine Distribution
    # ======================================================
    mean_cos, std_cos = compute_mean_cosine(V)

    print(
        f"Mean cosine similarity "
        f"(off-diagonal): {mean_cos:.6f}"
    )

    print(
        f"Cosine similarity std: {std_cos:.6f}"
    )

    # ======================================================
    # 3️. Intra vs Inter Event Similarity
    # ======================================================
    intra, inter = intra_inter_similarity(
        V,
        index,
    )

    print(
        f"Intra-event similarity: {intra:.6f}"
    )

    print(
        f"Inter-event similarity: {inter:.6f}"
    )

    print(
        f"Gap (intra - inter): "
        f"{intra - inter:.6f}"
    )

    # ======================================================
    # 4️. Bucket Centroid Separation
    # ======================================================
    centroid_sim = centroid_separation(
        V,
        index,
    )

    print(
        f"Mean centroid cosine similarity: "
        f"{centroid_sim:.6f}"
    )

    # ======================================================
    # Interpretation Hints
    # ======================================================
    #
    # These are heuristic diagnostics,
    # NOT strict pass/fail rules.
    #
    print("\n--- Interpretation Hints ---")

    # ------------------------------------------------------
    # Possible embedding collapse
    # ------------------------------------------------------
    if var < 0.01:

        print(
            " Very low variance → "
            "possible embedding collapse"
        )

    # ------------------------------------------------------
    # Excessively similar embeddings
    # ------------------------------------------------------
    if mean_cos > 0.5:

        print(
            " High mean cosine → "
            "embeddings too similar "
            "(collapse risk)"
        )

    # ------------------------------------------------------
    # Weak event separation
    # ------------------------------------------------------
    if (intra - inter) < 0.01:

        print(
            " Weak event separation"
        )

    # ------------------------------------------------------
    # Weak bucket separation
    # ------------------------------------------------------
    if centroid_sim > 0.7:

        print(
            " Buckets not well separated"
        )

    print("====================================================\n")


# ==========================================================
# Main Entry
# ==========================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        type=str,
    )

    parser.add_argument(
        "--analyze_all",
        action="store_true",
    )

    args = parser.parse_args()

    # ======================================================
    # Analyze ALL variants
    # ======================================================
    if args.analyze_all:

        for v in os.listdir(ARTIFACT_ROOT):

            analyze_variant(

                os.path.join(
                    ARTIFACT_ROOT,
                    v,
                )
            )

    # ======================================================
    # Analyze ONE variant
    # ======================================================
    else:

        if not args.variant:

            raise ValueError(
                "Provide --variant "
                "or --analyze_all"
            )

        analyze_variant(

            os.path.join(
                ARTIFACT_ROOT,
                args.variant,
            )
        )


if __name__ == "__main__":
    main()