#!/usr/bin/env python3
"""
visualize_cross_event_retrieval.py
=====================================================================

Project 2A_v2 — Cross-Event Vision Retrieval Visualization

Purpose
---------------------------------------------------------------------
This script generates qualitative cross-event retrieval examples using
the learned vision embedding space produced by Project 2A.

A query disaster image is embedded using the trained vision encoder,
and nearest-neighbor retrieval is performed directly in embedding
space using cosine similarity.

Unlike standard nearest-neighbor retrieval, this script explicitly
enforces cross-event retrieval by excluding all images originating
from the same disaster event as the query.

This allows visualization of whether the learned representation
captures disaster semantics that generalize across independent events.

Examples:

    hurricane-harvey (flooding)
        →
    hurricane-florence (flooding)
        →
    midwest-flooding (flooding)

    santa-rosa-wildfire
        →
    socal-fire
        →
    other wildfire events

The resulting visualizations provide qualitative evidence that
semantically related disasters are grouped together in the learned
embedding space despite differences in geographic location,
collection conditions, and event identity.

---------------------------------------------------------------------
Methodology
---------------------------------------------------------------------

1. Load exported vision embeddings

       vision_embeddings.npy

2. Load retrieval metadata

       vision_index.json

3. Select valid query images

       flooding
       wildfire
       structural_damage

4. L2-normalize all embeddings

5. Compute cosine similarity

       similarity = z_query · z_candidate

6. Exclude:

       • self-match
       • same-event samples

7. Retrieve top-K nearest neighbors

8. Render qualitative retrieval grids

       Query Image
             +
       Top-K Cross-Event Neighbors

9. Save visualization figures to disk

---------------------------------------------------------------------
Input Artifacts
---------------------------------------------------------------------

2A_v2/artifacts/layer4_multi/

    vision_embeddings.npy
    vision_index.json

Image Directory

    data/images/hold/post_disaster/

---------------------------------------------------------------------
Output
---------------------------------------------------------------------

2A_v2/visualization/cross_event_retrieval/

    cross_event_00.png
    cross_event_01.png
    cross_event_02.png
    ...

Each figure contains:

    • Query image
    • Top-K retrieved neighbors
    • Event identifiers
    • Semantic buckets
    • Cosine similarity scores

---------------------------------------------------------------------
Why This Matters
---------------------------------------------------------------------

Project 2A is designed to learn a shared semantic representation
space rather than perform closed-set classification.

A successful embedding space should retrieve semantically related
disaster imagery from entirely different events.

This visualization serves as qualitative validation that:

    • flooding clusters with flooding
    • wildfire clusters with wildfire
    • structural damage clusters with structural damage

even when retrieved samples originate from previously unseen events.

The resulting retrieval behavior demonstrates cross-event
generalization and forms the foundation for the retrieval
infrastructure developed in Project 2B.

---------------------------------------------------------------------
Run Command
---------------------------------------------------------------------

PYTHONPATH=2A python3 \
2A/scripts/visualize_cross_event_retrieval.py

---------------------------------------------------------------------
Expected Console Output
---------------------------------------------------------------------

[INFO] Loading embeddings...
[INFO] Loading metadata...
[INFO] Valid query candidates: XXX

[INFO] Processing query 1/5
[OK] Saved:
2A_v2/visualization/cross_event_retrieval/cross_event_00.png

...

[DONE] Cross-event retrieval visualization complete.

=====================================================================
"""

import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

ARTIFACT_DIR = "2A/artifacts/layer4_multi"

VISION_EMBED_PATH = os.path.join(
    ARTIFACT_DIR,
    "vision_embeddings.npy"
)

VISION_INDEX_PATH = os.path.join(
    ARTIFACT_DIR,
    "vision_index.json"
)

IMAGE_ROOT = "data/images/hold/post_disaster"

OUTPUT_DIR = (
    "2A_v2/visualization/cross_event_retrieval"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPK = 5

NUM_QUERIES = 5

TARGET_BUCKETS = {
    "flooding",
    "wildfire",
    "structural_damage",
}

FIGSIZE = (20, 6)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# STEP 1: LOAD EMBEDDINGS AND METADATA
# ============================================================

print("[INFO] Loading embeddings...")

vision_embeddings = np.load(VISION_EMBED_PATH)

print("[INFO] Loading metadata...")

with open(VISION_INDEX_PATH, "r") as f:
    vision_index = json.load(f)

print("[INFO] Loaded:")
print(f"  vision embeddings : {vision_embeddings.shape}")


# ============================================================
# STEP 2: NORMALIZE EMBEDDINGS
# ============================================================

def l2_normalize(x, axis=1, eps=1e-8):
    """
    Apply L2 normalization to embedding vectors.

    Ensures cosine similarity can be computed using
    a simple dot product.
    """

    norm = np.linalg.norm(
        x,
        axis=axis,
        keepdims=True
    )

    return x / np.clip(norm, eps, None)


vision_embeddings = l2_normalize(
    vision_embeddings
)


# ============================================================
# STEP 3: SELECT VALID QUERY IMAGES
# ============================================================

def build_image_path(meta):
    """
    Construct the post-disaster image path associated
    with a metadata entry.

    Expected filename format:

        <patch_id>_post_disaster.png

    Example:

        hurricane-harvey_00000067_post_disaster.png
    """

    patch_id = meta["patch_id"]

    filename = (
        f"{patch_id}_post_disaster.png"
    )

    return os.path.join(
        IMAGE_ROOT,
        filename
    )


# ============================================================
# STEP 4: PERFORM CROSS-EVENT RETRIEVAL
# ============================================================

def retrieve_cross_event_neighbors(
    query_idx,
    topk=5
):
    """
    Retrieve the top-K most similar vision embeddings
    while enforcing cross-event retrieval.

    Excludes:

        • the query image itself
        • samples originating from the same event

    Similarity is computed using cosine similarity
    between normalized embeddings.

    Returns:
        List of nearest-neighbor metadata records.
    """

    query_embedding = vision_embeddings[query_idx]

    query_meta = vision_index[query_idx]

    query_event = query_meta["event_id"]

    sims = vision_embeddings @ query_embedding

    sorted_idx = np.argsort(-sims)

    neighbors = []

    for idx in sorted_idx:

        # skip self
        if idx == query_idx:
            continue

        candidate_meta = vision_index[idx]

        candidate_event = candidate_meta["event_id"]

        # enforce cross-event retrieval
        if candidate_event == query_event:
            continue

        neighbors.append({

            "idx":
                idx,

            "similarity":
                float(sims[idx]),

            "event_id":
                candidate_meta["event_id"],

            "damage_bucket":
                candidate_meta["damage_bucket"],

            "meta":
                candidate_meta
        })

        if len(neighbors) >= topk:
            break

    return neighbors


# ============================================================
# FILTER VALID QUERIES
# ============================================================

candidate_indices = []

for i, meta in enumerate(vision_index):

    bucket = meta.get(
        "damage_bucket",
        ""
    )

    if bucket not in TARGET_BUCKETS:
        continue

    image_path = build_image_path(meta)

    if not os.path.exists(image_path):
        continue

    candidate_indices.append(i)

print(f"[INFO] Valid query candidates: {len(candidate_indices)}")

random.shuffle(candidate_indices)

query_indices = candidate_indices[:NUM_QUERIES]


# ============================================================
# STEP 5: RENDER QUALITATIVE RETRIEVAL VISUALIZATIONS
# ============================================================

for qnum, qidx in enumerate(query_indices):

    print(
        f"\n[INFO] Processing query {qnum+1}/{NUM_QUERIES}"
    )

    query_meta = vision_index[qidx]

    query_image_path = build_image_path(
        query_meta
    )

    query_image = Image.open(
        query_image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # RETRIEVE NEIGHBORS
    # --------------------------------------------------------

    neighbors = retrieve_cross_event_neighbors(
        qidx,
        topk=TOPK
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        TOPK + 1,
        figsize=FIGSIZE
    )

    # ========================================================
    # QUERY IMAGE
    # ========================================================

    axes[0].imshow(query_image)

    axes[0].set_title(
        (
            "QUERY\n\n"
            f"{query_meta['event_id']}\n"
            f"{query_meta['damage_bucket']}"
        ),
        fontsize=11,
        weight="bold"
    )

    axes[0].axis("off")

    # ========================================================
    # RETRIEVED NEIGHBORS
    # ========================================================

    for k, item in enumerate(neighbors):

        neighbor_meta = item["meta"]

        neighbor_path = build_image_path(
            neighbor_meta
        )

        if not os.path.exists(neighbor_path):

            print(
                f"[WARNING] Missing image: "
                f"{neighbor_path}"
            )

            axes[k + 1].axis("off")
            continue

        neighbor_image = Image.open(
            neighbor_path
        ).convert("RGB")

        axes[k + 1].imshow(neighbor_image)

        axes[k + 1].set_title(
            (
                f"Rank #{k+1}\n\n"
                f"{item['event_id']}\n"
                f"{item['damage_bucket']}\n"
                f"sim={item['similarity']:.3f}"
            ),
            fontsize=10
        )

        axes[k + 1].axis("off")

    # ========================================================
    # GLOBAL TITLE
    # ========================================================

    fig.suptitle(
        (
            "Cross-Event Vision Retrieval\n"
            "Project 2A"
        ),
        fontsize=18,
        weight="bold"
    )

    plt.tight_layout()

    # ============================================================
	# STEP 6: SAVE FIGURES TO DISK
	# ============================================================

    out_path = os.path.join(
        OUTPUT_DIR,
        f"cross_event_{qnum:02d}.png"
    )

    plt.savefig(
        out_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[OK] Saved: {out_path}")

print("\n[DONE] Cross-event retrieval visualization complete.")