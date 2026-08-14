#!/usr/bin/env python3
"""
visualize_embedding_umap.py
=====================================================================

Project 2A — Embedding Geometry Visualization

Purpose
---------------------------------------------------------------------
This script visualizes the semantic structure learned by the Project
2A multimodal retrieval model.

The exported embedding vectors are projected from the original
256-dimensional embedding space into two dimensions using:

    • UMAP
    • t-SNE

The resulting visualizations provide qualitative evidence that the
contrastive training objective has organized semantically related
disaster samples into coherent regions of the embedding space.

Examples:

    flooding
        → flooding

    wildfire
        → wildfire

    structural_damage
        → structural_damage

even when samples originate from different disaster events.

The script additionally visualizes all modalities together
(vision, text, and whisper) to inspect whether a shared semantic
space has emerged across modalities.

---------------------------------------------------------------------
Generated Visualizations
---------------------------------------------------------------------

1. UMAP — Vision Only

    Visualizes vision embeddings only.

    Goal:

        Determine whether disaster categories
        form coherent semantic clusters.

2. t-SNE — Vision Only

    Alternative nonlinear projection.

    Goal:

        Validate cluster structure using a
        second manifold-learning technique.

3. UMAP — Multimodal

    Projects:

        • vision embeddings
        • text embeddings
        • whisper embeddings

    into a common 2D space.

    Goal:

        Inspect cross-modal alignment and
        shared semantic organization.

---------------------------------------------------------------------
Input Artifacts
---------------------------------------------------------------------

2A/artifacts/layer4_multi/

    vision_embeddings.npy
    text_embeddings.npy
    whisper_embeddings.npy

    vision_index.json
    text_index.json
    whisper_index.json

Expected Shapes

    vision_embeddings.npy
        (694, 256)

    text_embeddings.npy
        (70, 256)

    whisper_embeddings.npy
        (11, 256)

All embeddings are expected to be L2-normalized
representation vectors exported by the Project 2A
training pipeline.

---------------------------------------------------------------------
Methodology
---------------------------------------------------------------------

STEP 1

    Load exported embeddings and metadata.

STEP 2

    Apply L2 normalization.

STEP 3

    Compute UMAP projections.

STEP 4

    Compute t-SNE projections.

STEP 5

    Generate vision-only semantic cluster plots.

STEP 6

    Generate multimodal semantic-space plots.

STEP 7

    Save publication-quality figures.

---------------------------------------------------------------------
Output Directory
---------------------------------------------------------------------

2A_v2/visualization/embedding_clusters/

Generated Files

    umap_vision.png
    tsne_vision.png
    umap_multimodal.png

---------------------------------------------------------------------
Interpretation
---------------------------------------------------------------------

A successful embedding space should exhibit:

    • flooding clusters
    • wildfire clusters
    • structural damage clusters

rather than grouping samples by event identity.

Strong separation between semantic categories
suggests that the model has learned meaningful
disaster representations.

The multimodal projection further allows
inspection of whether vision, text, and
spoken-language embeddings occupy a common
semantic manifold.

---------------------------------------------------------------------
Why This Matters
---------------------------------------------------------------------

Project 2A focuses on representation learning.

The quality of retrieval depends directly on
the geometry of the learned embedding space.

These visualizations serve as qualitative
evidence that:

    • semantic structure has emerged
    • cross-event generalization exists
    • multimodal alignment is feasible

and therefore support the retrieval experiments
presented elsewhere in Project 2A.

---------------------------------------------------------------------
Dependencies
---------------------------------------------------------------------

pip install umap-learn scikit-learn

---------------------------------------------------------------------
Run Command
---------------------------------------------------------------------

PYTHONPATH=2A python3 \
2A/scripts/visualize_embedding_umap.py

---------------------------------------------------------------------
Expected Console Output
---------------------------------------------------------------------

[INFO] Loading embeddings...
[INFO] Loading metadata...

[INFO] Computing UMAP (vision)...
[OK] Saved: umap_vision.png

[INFO] Computing t-SNE (vision)...
[OK] Saved: tsne_vision.png

[INFO] Computing multimodal UMAP...
[OK] Saved: umap_multimodal.png

[DONE] Embedding visualization complete.

=====================================================================
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import umap.umap_ as umap


# ============================================================
# CONFIG
# ============================================================

ARTIFACT_DIR = "2A_v2/artifacts/layer4_multi"

VISION_EMBED_PATH = os.path.join(
    ARTIFACT_DIR,
    "vision_embeddings.npy"
)

TEXT_EMBED_PATH = os.path.join(
    ARTIFACT_DIR,
    "text_embeddings.npy"
)

WHISPER_EMBED_PATH = os.path.join(
    ARTIFACT_DIR,
    "whisper_embeddings.npy"
)

VISION_INDEX_PATH = os.path.join(
    ARTIFACT_DIR,
    "vision_index.json"
)

TEXT_INDEX_PATH = os.path.join(
    ARTIFACT_DIR,
    "text_index.json"
)

WHISPER_INDEX_PATH = os.path.join(
    ARTIFACT_DIR,
    "whisper_index.json"
)

OUTPUT_DIR = (
    "2A_v2/visualization/embedding_clusters"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

FIGSIZE = (12, 10)


# ============================================================
# COLOR MAP
# ============================================================

BUCKET_COLORS = {

    "flooding":
        "tab:blue",

    "wildfire":
        "tab:red",

    "structural_damage":
        "tab:green",

    "volcanic_damage":
        "tab:orange",

    "tsunami_inundation":
        "tab:purple",

    "generic_damage":
        "tab:gray",
}


# ============================================================
# MARKER MAP
# ============================================================

MODALITY_MARKERS = {

    "vision":
        "o",

    "text":
        "^",

    "whisper":
        "s",
}


# ============================================================
# STEP 1: LOAD EMBEDDINGS AND METADATA
# ============================================================

print("[INFO] Loading embeddings...")

vision_embeddings = np.load(
    VISION_EMBED_PATH
)

text_embeddings = np.load(
    TEXT_EMBED_PATH
)

whisper_embeddings = np.load(
    WHISPER_EMBED_PATH
)

print("[INFO] Loading metadata...")

with open(VISION_INDEX_PATH, "r") as f:
    vision_index = json.load(f)

with open(TEXT_INDEX_PATH, "r") as f:
    text_index = json.load(f)

with open(WHISPER_INDEX_PATH, "r") as f:
    whisper_index = json.load(f)

print("[INFO] Loaded:")
print(f"  vision  : {vision_embeddings.shape}")
print(f"  text    : {text_embeddings.shape}")
print(f"  whisper : {whisper_embeddings.shape}")


# ============================================================
# STEP 2: NORMALIZE EMBEDDINGS
# ============================================================

def l2_normalize(x, axis=1, eps=1e-8):
    """
    Apply L2 normalization to embedding vectors.

    Normalization ensures that cosine similarity
    can be computed directly via dot products.

    Args:
        x:
            Embedding matrix.

        axis:
            Normalization axis.

        eps:
            Numerical stability constant.

    Returns:
        L2-normalized embeddings.
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

text_embeddings = l2_normalize(
    text_embeddings
)

whisper_embeddings = l2_normalize(
    whisper_embeddings
)


# ============================================================
# STEP 3: DEFINE MANIFOLD PROJECTION HELPERS
# ============================================================

def compute_umap(embeddings):
    """
    Project embeddings into two dimensions using UMAP.

    UMAP preserves local neighborhood structure and
    is commonly used for visualizing semantic manifolds.

    Args:
        embeddings:
            NxD embedding matrix.

    Returns:
        Nx2 projected coordinates.
    """

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    return reducer.fit_transform(
        embeddings
    )


# ============================================================
# t-SNE HELPER
# ============================================================

def compute_tsne(embeddings):
    """
    Project embeddings into two dimensions using t-SNE.

    t-SNE emphasizes local cluster structure and
    provides an alternative view of embedding geometry.

    Args:
        embeddings:
            NxD embedding matrix.

    Returns:
        Nx2 projected coordinates.
    """

    reducer = TSNE(
        n_components=2,
        perplexity=30,
        metric="cosine",
        random_state=RANDOM_STATE,
        init="random",
    )

    return reducer.fit_transform(
        embeddings
    )


# ============================================================
# STEP 4: DEFINE VISUALIZATION HELPERS
# ============================================================

def plot_embeddings(
    coords,
    metadata,
    title,
    save_path,
    modality=None,
):
    """
    Visualize semantic embedding clusters.

    Samples are colored according to damage category.

    Optional modality-specific markers may be used
    to distinguish:

        • vision
        • text
        • whisper

    embeddings.

    Args:
        coords:
            2D projected coordinates.

        metadata:
            Sample metadata records.

        title:
            Figure title.

        save_path:
            Output image path.

        modality:
            Optional modality identifier.

    Returns:
        None.
    """

    plt.figure(figsize=FIGSIZE)

    used_labels = set()

    for i, meta in enumerate(metadata):

        bucket = meta.get(
            "damage_bucket",
            "generic_damage"
        )

        color = BUCKET_COLORS.get(
            bucket,
            "black"
        )

        marker = "o"

        if modality is not None:
            marker = MODALITY_MARKERS[
                modality
            ]

        label = bucket

        if label in used_labels:
            label = None
        else:
            used_labels.add(label)

        plt.scatter(
            coords[i, 0],
            coords[i, 1],
            c=color,
            marker=marker,
            s=50,
            alpha=0.8,
            label=label
        )

    plt.title(
        title,
        fontsize=18,
        weight="bold"
    )

    plt.xticks([])
    plt.yticks([])

    plt.legend(
        fontsize=10,
        loc="best"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[OK] Saved: {save_path}")


# ============================================================
# STEP 5: GENERATE UMAP PROJECTION (VISION)
# ============================================================

print("\n[INFO] Computing UMAP (vision)...")

vision_umap = compute_umap(
    vision_embeddings
)

plot_embeddings(
    coords=vision_umap,
    metadata=vision_index,
    title=(
        "UMAP Semantic Clusters (Vision)\n"
        "Project 2A"
    ),
    save_path=os.path.join(
        OUTPUT_DIR,
        "umap_vision.png"
    ),
    modality="vision",
)


# ============================================================
# STEP 6: GENERATE t-SNE PROJECTION (VISION)
# ============================================================

print("\n[INFO] Computing t-SNE (vision)...")

vision_tsne = compute_tsne(
    vision_embeddings
)

plot_embeddings(
    coords=vision_tsne,
    metadata=vision_index,
    title=(
        "t-SNE Semantic Clusters (Vision)\n"
        "Project 2A"
    ),
    save_path=os.path.join(
        OUTPUT_DIR,
        "tsne_vision.png"
    ),
    modality="vision",
)


# ============================================================
# STEP 7: GENERATE MULTIMODAL UMAP PROJECTION
# ============================================================

print("\n[INFO] Computing multimodal UMAP...")

all_embeddings = np.concatenate([

    vision_embeddings,
    text_embeddings,
    whisper_embeddings

], axis=0)

all_metadata = (
    vision_index
    + text_index
    + whisper_index
)

modalities = (

    ["vision"] * len(vision_embeddings)

    +

    ["text"] * len(text_embeddings)

    +

    ["whisper"] * len(whisper_embeddings)
)

multimodal_umap = compute_umap(
    all_embeddings
)

# ============================================================
# STEP 8: SAVE VISUALIZATION FIGURES
# ============================================================

plt.figure(figsize=(14, 12))

used_labels = set()

for i, meta in enumerate(all_metadata):

    bucket = meta.get(
        "damage_bucket",
        "generic_damage"
    )

    modality = modalities[i]

    color = BUCKET_COLORS.get(
        bucket,
        "black"
    )

    marker = MODALITY_MARKERS[
        modality
    ]

    label = f"{bucket}"

    if label in used_labels:
        label = None
    else:
        used_labels.add(label)

    plt.scatter(
        multimodal_umap[i, 0],
        multimodal_umap[i, 1],
        c=color,
        marker=marker,
        s=60,
        alpha=0.75,
        label=label
    )

# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------

plt.title(
    (
        "Shared Multimodal Semantic Space\n"
        "UMAP Projection — Project 2A"
    ),
    fontsize=18,
    weight="bold"
)

plt.xticks([])
plt.yticks([])

plt.legend(
    fontsize=10,
    loc="best"
)

# ------------------------------------------------------------
# MODALITY LEGEND
# ------------------------------------------------------------

from matplotlib.lines import Line2D

modality_handles = [

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="vision",
        markerfacecolor="black",
        markersize=10
    ),

    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        label="text",
        markerfacecolor="black",
        markersize=10
    ),

    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="whisper",
        markerfacecolor="black",
        markersize=10
    ),
]

plt.legend(
    handles=modality_handles,
    loc="lower right",
    title="Modality"
)

plt.tight_layout()

save_path = os.path.join(
    OUTPUT_DIR,
    "umap_multimodal.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"[OK] Saved: {save_path}")

print("\n[DONE] Embedding visualization complete.")