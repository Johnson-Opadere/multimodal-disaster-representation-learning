#!/usr/bin/env python3
"""
export_embeddings.py
===========================================================
Project 2A — Embedding Materialization
===========================================================

OVERVIEW
===========================================================

This script exports frozen multimodal embeddings from a
trained VisionEncoder checkpoint and produces all metadata
required for:

    Project 2B retrieval evaluation.

This script marks the architectural boundary between:

    2A → representation learning
    2B → retrieval evaluation

===========================================================
HIGH-LEVEL OBJECTIVE
===========================================================

Project 2A learns:

    semantic embedding geometry

Project 2B later evaluates:

    retrieval quality

using:
    • FAISS indexing
    • nearest-neighbor retrieval
    • Recall@K
    • MRR
    • XE retrieval metrics
    • leakage analysis

IMPORTANT:
-----------------------------------------------------------

2B MUST NOT load model checkpoints.

2B consumes ONLY:

    • frozen embeddings
    • deterministic indices
    • metadata

===========================================================
REPRESENTATION LEARNING CONTRACT
===========================================================

All exported embeddings satisfy:

    • float32 dtype
    • deterministic ordering
    • 256-dimensional vectors
    • L2 normalization

This is REQUIRED for:

    • cosine similarity
    • FAISS inner-product retrieval
    • stable ranking
    • reproducible evaluation

===========================================================
EXPORTED ARTIFACTS
===========================================================

2A/artifacts/{finetune}_{positive}/

    vision_embeddings.npy
    vision_index.json

    text_embeddings.npy
    text_index.json

    whisper_embeddings.npy
    whisper_index.json

    metadata.json

===========================================================
ARCHITECTURE RESPONSIBILITIES
===========================================================

2A responsibilities:
-----------------------------------------------------------

    • train embedding geometry
    • export frozen embeddings
    • export deterministic indices

2B responsibilities:
-----------------------------------------------------------

    • similarity search
    • retrieval ranking
    • retrieval metrics
    • cross-event evaluation
    • leakage auditing

===========================================================
IMPORTANT SCIENTIFIC IDEAS
===========================================================

1️. Deterministic Export
-----------------------------------------------------------

DataLoader uses:

    shuffle=False

This guarantees:

    embedding[i]
        ↔
    metadata[i]

remain permanently aligned.

This is CRITICAL for retrieval evaluation.

-----------------------------------------------------------

2️. Embedding Collapse Monitoring
-----------------------------------------------------------

The script computes geometry diagnostics for:

    • embedding collapse
    • diversity health
    • cosine similarity spread

These metrics help diagnose:

    • failed contrastive learning
    • unstable geometry
    • retrieval degradation

-----------------------------------------------------------

3️. Cross-Event Evaluation
-----------------------------------------------------------

This project evaluates:

    semantic retrieval across events

NOT:
    same-event memorization.

Therefore:
event metadata is intentionally exported for:

    • XE retrieval metrics
    • same-event filtering
    • leakage analysis
    • retrieval auditing

===========================================================
EMBEDDING GEOMETRY THEORY
===========================================================

The exported embeddings represent:

    semantic disaster representations

Examples:

    flood imagery
        ↔
    flood text
        ↔
    flood whisper narration

All embeddings are projected into a shared:

    z ∈ R^256

semantic space.

===========================================================
RUN COMMANDS
===========================================================

cd 2A_v2

-----------------------------------------------------------
1️. Recommended final model 
-----------------------------------------------------------

PYTHONPATH=. python scripts/export_embeddings.py \
    --finetune layer4 \
    --positive multi \
    --epoch 10 \
    --split hold

-----------------------------------------------------------
2️. Frozen baseline
-----------------------------------------------------------

PYTHONPATH=. python scripts/export_embeddings.py \
    --finetune frozen \
    --positive multi \
    --epoch 0 \
    --split hold

-----------------------------------------------------------
3️. Full vision finetuning
-----------------------------------------------------------

PYTHONPATH=. python scripts/export_embeddings.py \
    --finetune full \
    --positive multi \
    --epoch 4 \
    --split hold

-----------------------------------------------------------
4️. Supervision ablation
-----------------------------------------------------------

PYTHONPATH=. python scripts/export_embeddings.py \
    --finetune layer4 \
    --positive single \
    --epoch 11 \
    --split hold

===========================================================
OUTPUT DIRECTORY
===========================================================

2A/artifacts/{finetune}_{positive}/

Example:

    2A/artifacts/layer4_multi/

===========================================================
"""

import argparse
import json
import os
import shutil
from collections import Counter

import numpy as np
import torch

from torch.utils.data import DataLoader

from tqdm import tqdm

from src.data.multimodal_damage_dataset import (
    MultimodalDamageDataset,
)

from src.model.text_encoder import TextEncoder

from src.model.vision_encoder import VisionEncoder

from src.model.whisper_encoder import WhisperEncoder


# ==========================================================
# Stable Event → Bucket Mapping
# ==========================================================
#
# IMPORTANT:
# Event identity is NOT used as supervision.
#
# Multiple disaster events may share the SAME
# semantic damage bucket.
#
# This enables:
#   • cross-event retrieval
#   • semantic generalization
#   • retrieval robustness
#
EVENT_TO_BUCKET = {

    # ------------------------------------------------------
    # Structural
    # ------------------------------------------------------
    "mexico-earthquake": "structural_damage",

    # ------------------------------------------------------
    # Volcanic
    # ------------------------------------------------------
    "guatemala-volcano": "volcanic_damage",

    # ------------------------------------------------------
    # Flooding
    # ------------------------------------------------------
    "hurricane-harvey": "flooding",
    "hurricane-florence": "flooding",
    "hurricane-matthew": "flooding",
    "hurricane-michael": "flooding",
    "midwest-flooding": "flooding",

    # ------------------------------------------------------
    # Tsunami
    # ------------------------------------------------------
    "palu-tsunami": "tsunami_inundation",

    # ------------------------------------------------------
    # Wildfire
    # ------------------------------------------------------
    "santa-rosa-wildfire": "wildfire",
    "socal-fire": "wildfire",
}


# ==========================================================
# L2 Normalization
# ==========================================================
#
# Converts embeddings onto a hypersphere:
#
#     ||z|| = 1
#
# This allows:
#   • cosine similarity
#   • FAISS inner-product retrieval
#   • stable semantic geometry
#
def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    Row-wise L2 normalization.

    Input:
        (N, D)

    Output:
        L2-normalized embeddings
    """

    return x / (
        np.linalg.norm(
            x,
            axis=1,
            keepdims=True,
        ) + 1e-12
    )


# ==========================================================
# Embedding Diagnostics
# ==========================================================
#
# Computes geometry-health diagnostics for:
#   • collapse detection
#   • semantic diversity
#   • retrieval stability
#
def compute_embedding_diagnostics(
    embeddings: np.ndarray,
):
    """
    Compute embedding geometry diagnostics.

    Metrics:
        • norm_mean
        • norm_std
        • embedding_std
        • pairwise cosine mean/std

    These metrics help detect:
        • embedding collapse
        • low diversity
        • unstable contrastive training
    """

    # ------------------------------------------------------
    # Norms
    # ------------------------------------------------------
    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    # ------------------------------------------------------
    # Pairwise cosine similarities
    # ------------------------------------------------------
    similarity = embeddings @ embeddings.T

    #
    # Remove diagonal self-similarity
    #
    mask = ~np.eye(
        similarity.shape[0],
        dtype=bool,
    )

    pairwise = similarity[mask]

    diagnostics = {

        # ----------------------------------------------
        # Norm health
        # ----------------------------------------------
        "norm_mean":
            float(norms.mean()),

        "norm_std":
            float(norms.std()),

        # ----------------------------------------------
        # Geometry diversity
        # ----------------------------------------------
        "embedding_std":
            float(embeddings.std()),

        "pairwise_cosine_mean":
            float(pairwise.mean()),

        "pairwise_cosine_std":
            float(pairwise.std()),
    }

    return diagnostics


# ==========================================================
# Main Export Entry
# ==========================================================
def main():

    # ======================================================
    # CLI
    # ======================================================
    parser = argparse.ArgumentParser(
        description=(
            "Project 2A_v2 — Export embeddings "
            "for retrieval evaluation"
        )
    )

    parser.add_argument(
        "--finetune",
        required=True,
    )

    parser.add_argument(
        "--positive",
        required=True,
    )

    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--split",
        default="hold",
    )

    args = parser.parse_args()

    # ======================================================
    # Device
    # ======================================================
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n===================================================")
    print("Project 2A_v2 — Export Embeddings")
    print("===================================================\n")

    print(f"Device     : {DEVICE}")
    print(f"Finetune   : {args.finetune}")
    print(f"Positive   : {args.positive}")
    print(f"Epoch      : {args.epoch}")
    print(f"Split      : {args.split}")

    print("\n===================================================\n")

    # ======================================================
    # Checkpoint Path
    # ======================================================
    ckpt_path = os.path.join(
        "2A",
        "checkpoints",
        f"{args.finetune}_{args.positive}",
        f"vision_encoder_epoch{args.epoch}.pt",
    )

    if not os.path.exists(ckpt_path):

        raise FileNotFoundError(
            f"Checkpoint not found:\n{ckpt_path}"
        )

    print(f"Loading checkpoint:\n{ckpt_path}\n")

    # ======================================================
    # Vision Encoder
    # ======================================================
    #
    # Loads trained semantic geometry.
    #
    vision_encoder = VisionEncoder(
        rgb_channels=3,
        sar_channels=8,
        embed_dim=256,
    ).to(DEVICE)

    vision_encoder.load_state_dict(
        torch.load(
            ckpt_path,
            map_location=DEVICE,
        )
    )

    vision_encoder.eval()

    # ======================================================
    # Dataset
    # ======================================================
    dataset = MultimodalDamageDataset(
        root_dir="data/normalized_data",
        split=args.split,
        event_to_bucket=EVENT_TO_BUCKET,
    )

    # ======================================================
    # Leakage / Split Diagnostics
    # ======================================================
    #
    # Useful for:
    #   • auditing
    #   • XE retrieval
    #   • split verification
    #
    print("Split Diagnostics")
    print("---------------------------------------------------")

    events = []

    buckets = []

    for sid in dataset.ids:

        event = sid.rsplit("_", 1)[0]

        bucket = EVENT_TO_BUCKET[event]

        events.append(event)

        buckets.append(bucket)

    unique_events = sorted(
        set(events)
    )

    bucket_counts = Counter(buckets)

    print(f"Samples        : {len(dataset)}")

    print(f"Unique events  : {len(unique_events)}")

    print(f"Events         : {unique_events}")

    print("\nBucket Distribution")
    print("---------------------------------------------------")

    for bucket, count in bucket_counts.items():

        print(f"{bucket:<25} {count}")

    print("\n===================================================\n")

    # ======================================================
    # Deterministic DataLoader
    # ======================================================
    #
    # IMPORTANT:
    #
    # shuffle=False guarantees:
    #
    #   embedding[i]
    #       ↔
    #   index[i]
    #
    # remain permanently aligned.
    #
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    # ======================================================
    # Vision Embedding Extraction
    # ======================================================
    vision_embeddings = []

    vision_index = []

    print("Extracting vision embeddings...")
    print("---------------------------------------------------")

    with torch.no_grad():

        for idx, batch in enumerate(
            tqdm(loader)
        ):

            # --------------------------------------------------
            # Vision embedding
            # --------------------------------------------------
            #
            # Produces:
            #   z_patch ∈ R^256
            #
            z_patch = vision_encoder(
                rgb_pre=batch["rgb_pre"].to(DEVICE),
                sar_pre=batch["sar_pre"].to(DEVICE),
                rgb_post=batch["rgb_post"].to(DEVICE),
                sar_post=batch["sar_post"].to(DEVICE),
            )

            vision_embeddings.append(
                z_patch.cpu().numpy()
            )

            # --------------------------------------------------
            # Deterministic metadata
            # --------------------------------------------------
            #
            # Used later for:
            #   • retrieval auditing
            #   • XE metrics
            #   • leakage analysis
            #
            sid = dataset.ids[idx]

            event_id = sid.rsplit("_", 1)[0]

            bucket = EVENT_TO_BUCKET[event_id]

            vision_index.append({

                "idx":
                    idx,

                "event_id":
                    event_id,

                "damage_bucket":
                    bucket,

                "patch_id":
                    sid,

                "split":
                    args.split,

                #
                # Useful later for:
                #   • XE filtering
                #   • held-out evaluation
                #   • leakage auditing
                #
                "is_holdout_event":
                    False,
            })

    # ======================================================
    # Stack + Normalize
    # ======================================================
    vision_embeddings = np.vstack(
        vision_embeddings
    ).astype(np.float32)

    vision_embeddings = l2_normalize(
        vision_embeddings
    )

    print(
        f"\nVision embeddings shape : "
        f"{vision_embeddings.shape}"
    )

    # ======================================================
    # Vision Diagnostics
    # ======================================================
    print("\nVision Embedding Diagnostics")
    print("---------------------------------------------------")

    vision_diag = compute_embedding_diagnostics(
        vision_embeddings
    )

    for k, v in vision_diag.items():

        print(f"{k:<30} {v:.6f}")

    # ======================================================
    # Text + Whisper Encoders
    # ======================================================
    #
    # Frozen language encoders used to produce:
    #   • text embeddings
    #   • whisper embeddings
    #
    text_encoder = TextEncoder(
        embed_dim=256,
        freeze_backbone=True,
    ).to(DEVICE).eval()

    whisper_encoder = WhisperEncoder(
        embed_dim=256,
        freeze_backbone=True,
    ).to(DEVICE).eval()

    # ======================================================
    # Token Files
    # ======================================================
    text_tokens = torch.load(
        "data/text_tokens/tokens.pt"
    )

    whisper_tokens = torch.load(
        "data/whisper_tokens/tokens.pt"
    )

    # ======================================================
    # Language Embedding Extraction
    # ======================================================
    print("\nExtracting language embeddings...")
    print("---------------------------------------------------")

    with torch.no_grad():

        z_text = text_encoder(
            text_tokens["input_ids"].to(DEVICE),
            text_tokens["attention_mask"].to(DEVICE),
        )

        z_whisper = whisper_encoder(
            whisper_tokens["input_ids"].to(DEVICE),
            whisper_tokens["attention_mask"].to(DEVICE),
        )

    # ======================================================
    # L2 Normalize Language Embeddings
    # ======================================================
    text_embeddings = l2_normalize(
        z_text.cpu().numpy().astype(np.float32)
    )

    whisper_embeddings = l2_normalize(
        z_whisper.cpu().numpy().astype(np.float32)
    )

    print(
        f"text_embeddings     : "
        f"{text_embeddings.shape}"
    )

    print(
        f"whisper_embeddings  : "
        f"{whisper_embeddings.shape}"
    )

    # ======================================================
    # Language Diagnostics
    # ======================================================
    print("\nText Embedding Diagnostics")
    print("---------------------------------------------------")

    text_diag = compute_embedding_diagnostics(
        text_embeddings
    )

    for k, v in text_diag.items():

        print(f"{k:<30} {v:.6f}")

    print("\nWhisper Embedding Diagnostics")
    print("---------------------------------------------------")

    whisper_diag = compute_embedding_diagnostics(
        whisper_embeddings
    )

    for k, v in whisper_diag.items():

        print(f"{k:<30} {v:.6f}")

    # ======================================================
    # Artifact Directory
    # ======================================================
    artifact_dir = os.path.join(
        "2A_v2",
        "artifacts",
        f"{args.finetune}_{args.positive}",
    )

    os.makedirs(
        artifact_dir,
        exist_ok=True,
    )

    # ======================================================
    # Save Embeddings
    # ======================================================
    #
    # These embeddings become the ONLY retrieval inputs
    # for Project 2B.
    #
    np.save(
        os.path.join(
            artifact_dir,
            "vision_embeddings.npy",
        ),
        vision_embeddings,
    )

    np.save(
        os.path.join(
            artifact_dir,
            "text_embeddings.npy",
        ),
        text_embeddings,
    )

    np.save(
        os.path.join(
            artifact_dir,
            "whisper_embeddings.npy",
        ),
        whisper_embeddings,
    )

    # ======================================================
    # Save Vision Index
    # ======================================================
    with open(
        os.path.join(
            artifact_dir,
            "vision_index.json",
        ),
        "w",
    ) as f:

        json.dump(
            vision_index,
            f,
            indent=2,
        )

    # ======================================================
    # Copy Language Indices
    # ======================================================
    #
    # Preserves deterministic alignment:
    #
    #   embedding[i]
    #       ↔
    #   metadata[i]
    #
    shutil.copy(
        "data/text_tokens/index.json",
        os.path.join(
            artifact_dir,
            "text_index.json",
        ),
    )

    shutil.copy(
        "data/whisper_tokens/index.json",
        os.path.join(
            artifact_dir,
            "whisper_index.json",
        ),
    )

    # ======================================================
    # Export Metadata
    # ======================================================
    metadata = {

        # --------------------------------------------------
        # Export info
        # --------------------------------------------------
        "split":
            args.split,

        "finetune":
            args.finetune,

        "positive":
            args.positive,

        "epoch":
            args.epoch,

        # --------------------------------------------------
        # Embedding contract
        # --------------------------------------------------
        "embed_dim":
            256,

        "l2_normalized":
            True,

        "dtype":
            "float32",

        # --------------------------------------------------
        # Provenance
        # --------------------------------------------------
        "checkpoint":
            ckpt_path,

        # --------------------------------------------------
        # Counts
        # --------------------------------------------------
        "num_patches":
            int(vision_embeddings.shape[0]),

        "num_text":
            int(text_embeddings.shape[0]),

        "num_whisper":
            int(whisper_embeddings.shape[0]),

        # --------------------------------------------------
        # Geometry diagnostics
        # --------------------------------------------------
        "vision_diagnostics":
            vision_diag,

        "text_diagnostics":
            text_diag,

        "whisper_diagnostics":
            whisper_diag,
    }

    with open(
        os.path.join(
            artifact_dir,
            "metadata.json",
        ),
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ======================================================
    # Final Logging
    # ======================================================
    print("\n===================================================")
    print("Export Complete")
    print("===================================================")

    print(f"Artifacts saved to:\n{artifact_dir}")

    print("===================================================\n")


if __name__ == "__main__":
    main()