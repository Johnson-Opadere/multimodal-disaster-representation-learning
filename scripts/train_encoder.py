#!/usr/bin/env python3
"""
train_encoder.py
===========================================================
Project 2A_v2 — Multimodal Representation Learning
===========================================================

OVERVIEW
===========================================================

This script trains the multimodal VisionEncoder to learn a
shared semantic embedding space aligned with:

    vision ↔ text ↔ whisper

using:

    symmetric multi-positive contrastive learning.

The project objective is:

    semantic disaster representation learning

NOT:
    - image classification
    - event memorization
    - caption generation
    - segmentation

The learned embeddings are later exported into Project 2B
for:

    • semantic retrieval
    • FAISS indexing
    • multimodal search
    • cross-event evaluation
    • retrieval ranking analysis

===========================================================
LEARNING OBJECTIVE
===========================================================

The system learns a shared embedding geometry where:

    semantically similar disaster patterns

are pulled together across modalities.

Examples:

    flood imagery
        ↔
    flood text reports
        ↔
    flood eyewitness transcripts

while semantically different disaster types are pushed apart.

===========================================================
REPRESENTATION LEARNING PIPELINE
===========================================================

RGB pre/post
SAR pre/post
        │
        ▼

MultimodalDamageDataset
        │
        ▼

SemanticBatchSampler
        │
        ▼

VisionEncoder
        │
        ▼

z_patch embeddings
        │
        ▼

ContrastiveLoss
        │
        ▼

Shared semantic embedding space

===========================================================
2A RESPONSIBILITIES
===========================================================

Project 2A_v2 focuses on:

    • representation learning
    • multimodal semantic alignment
    • vision encoder training
    • embedding geometry formation

Project 2B_v2 later handles:

    • embedding export
    • FAISS retrieval
    • retrieval ranking
    • retrieval metrics
    • leakage analysis
    • grounded retrieval visualization

===========================================================
FINETUNING MODES
===========================================================

1️. frozen
-----------------------------------------------------------

Completely frozen geometry baseline.

Trainable:
    none

Frozen:
    • vision encoder
    • text encoder
    • whisper encoder

Purpose:
    establish semantic geometry baseline.

Only an epoch0 checkpoint is saved.

-----------------------------------------------------------

2️. layer4
-----------------------------------------------------------

Recommended final configuration.

Trainable:
    • ResNet layer4
    • projection heads

Frozen:
    • early vision backbone
    • text encoder
    • whisper encoder

Purpose:
    controlled semantic refinement.

This was the FINAL recommended configuration.

-----------------------------------------------------------

3️. full
-----------------------------------------------------------

Full vision finetuning.

Trainable:
    • entire vision encoder

Frozen:
    • text encoder
    • whisper encoder

Purpose:
    capacity stress test.

===========================================================
POSITIVE SUPERVISION MODES
===========================================================

1️. multi
-----------------------------------------------------------

Multi-positive semantic supervision.

Each bucket may contain:
    • multiple text positives
    • multiple whisper positives

Example:

    flooding:
        • Harvey flooding
        • Florence flooding
        • Midwest flooding

This is the FINAL recommended setup.

-----------------------------------------------------------

2️. single
-----------------------------------------------------------

Single-positive supervision ablation.

Each bucket uses:
    • ONE text positive OR
    • ONE whisper positive

Purpose:
    evaluate whether multi-positive supervision improves
    retrieval robustness and semantic geometry.

===========================================================
SEMANTIC SUPERVISION
===========================================================

The system uses weak semantic supervision through:

    semantic damage buckets

Examples:
    • flooding
    • wildfire
    • volcanic_damage
    • structural_damage

The model learns:

    semantic alignment across events

NOT:
    event identity memorization.

===========================================================
IMPORTANT REPRESENTATION LEARNING IDEAS
===========================================================

1️. Shared Embedding Space
-----------------------------------------------------------

All modalities are projected into:

    z ∈ R^256

with L2 normalization.

This enables:
    • cosine similarity
    • multimodal retrieval
    • semantic ranking

-----------------------------------------------------------

2️. Contrastive Learning
-----------------------------------------------------------

The loss learns:

    positives → closer
    negatives → farther

using:
    symmetric InfoNCE optimization.

-----------------------------------------------------------

3️. Semantic Batch Sampling
-----------------------------------------------------------

Batch diversity strongly affects:
    • hard negatives
    • semantic separation
    • retrieval geometry

Therefore:
SemanticBatchSampler is used instead of random batching.

===========================================================
CHECKPOINT OUTPUTS
===========================================================

Checkpoints:
-----------------------------------------------------------

2A/checkpoints/{finetune}_{positive}/
    vision_encoder_epoch{N}.pt

Metrics:
-----------------------------------------------------------

metrics.json

===========================================================
SCIENTIFIC NOTES
===========================================================

IMPORTANT:
High retrieval similarity alone is NOT sufficient.

The project specifically evaluates:

    cross-event semantic generalization

NOT:
    same-event memorization.

Therefore:
retrieval behavior must later be validated in 2B_v2.

===========================================================
2B_v2 DESIGN CONTRACT
===========================================================

Project 2B assumes:

    • embed_dim = 256
    • L2-normalized embeddings
    • deterministic export ordering
    • shared embedding space

Breaking these assumptions may invalidate retrieval.

===========================================================
RUN COMMANDS
===========================================================

1️. Frozen baseline
-----------------------------------------------------------

cd 2A_v2

PYTHONPATH=. python scripts/train_encoder.py \
    --finetune frozen \
    --positive multi

===========================================================

2️. Full vision finetuning
-----------------------------------------------------------

PYTHONPATH=. python scripts/train_encoder.py \
    --finetune full \
    --positive multi

===========================================================

3️. Recommended final model (i.e Layer4)
-----------------------------------------------------------

PYTHONPATH=. python scripts/train_encoder.py \
    --finetune layer4 \
    --positive multi

===========================================================

4️. Supervision ablation
-----------------------------------------------------------

PYTHONPATH=. python scripts/train_encoder.py \
    --finetune layer4 \
    --positive single

===========================================================
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.multimodal_damage_dataset import (
    MultimodalDamageDataset,
)

from src.data.semantic_batch_sampler import (
    SemanticBatchSampler,
)

from src.loss.contrastive_loss import ContrastiveLoss

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
# Instead:
# multiple events may map to the SAME semantic bucket.
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
# Seed Everything
# ==========================================================
#
# Ensures deterministic behavior across:
#   • Python RNG
#   • NumPy RNG
#   • PyTorch RNG
#
# Important for:
#   • reproducibility
#   • retrieval consistency
#   • scientific rigor
#
def seed_everything(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# Count Trainable Parameters
# ==========================================================
#
# Used to verify:
#   • frozen baseline
#   • partial finetuning
#   • full finetuning
#
def count_trainable_params(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# Save Metrics
# ==========================================================
#
# Saves training metrics for:
#   • analysis
#   • plotting
#   • experiment comparison
#
def save_metrics(metrics, path):

    with open(path, "w") as f:

        json.dump(metrics, f, indent=4)


# ==========================================================
# Main Training Entry
# ==========================================================
def main():

    # ======================================================
    # CLI
    # ======================================================
    parser = argparse.ArgumentParser(
        description="Project 2A_v2 — Representation Training"
    )

    parser.add_argument(
        "--finetune",
        required=True,
        choices=[
            "frozen",
            "full",
            "layer4",
        ],
    )

    parser.add_argument(
        "--positive",
        required=True,
        choices=[
            "multi",
            "single",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # ======================================================
    # Reproducibility
    # ======================================================
    seed_everything(args.seed)

    # ======================================================
    # Device Configuration
    # ======================================================
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ======================================================
    # Core Training Config
    # ======================================================
    #
    # IMPORTANT:
    # Contrastive learning quality depends heavily on:
    #
    #   • semantic diversity
    #   • hard negatives
    #   • cross-bucket competition
    #
    BATCH_SIZE = 8

    LR = 3e-4

    NUM_EPOCHS = (
        5
        if args.finetune == "full"
        else 12
    )

    ROOT_DIR = "data/normalized_data"

    SPLIT = "train"

    # ======================================================
    # Console Logging
    # ======================================================
    print("\n===================================================")
    print("Project 2A_v2 — Training")
    print("===================================================")

    print(f"Device          : {DEVICE}")
    print(f"Finetune Mode   : {args.finetune}")
    print(f"Positive Mode   : {args.positive}")
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Learning Rate   : {LR}")
    print(f"Epochs          : {NUM_EPOCHS}")
    print(f"Seed            : {args.seed}")

    print("===================================================\n")

    # ======================================================
    # Save Directory
    # ======================================================
    SAVE_DIR = os.path.join(
        "2A_v2",
        "checkpoints",
        f"{args.finetune}_{args.positive}",
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ======================================================
    # Load Positive Pools
    # ======================================================
    #
    # Multi-positive:
    #   richer semantic supervision
    #
    # Single-positive:
    #   ablation supervision
    #
    pools_path = (
        "data/positive_pools/pools.json"
        if args.positive == "multi"
        else "data/positive_pools/pools_single.json"
    )

    with open(pools_path, "r") as f:

        positive_pools = json.load(f)

    print(f"Using positive pools: {pools_path}")

    # ======================================================
    # Dataset
    # ======================================================
    dataset = MultimodalDamageDataset(
        root_dir=ROOT_DIR,
        split=SPLIT,
        event_to_bucket=EVENT_TO_BUCKET,
    )

    # ======================================================
    # Semantic Batch Sampler
    # ======================================================
    #
    # IMPORTANT:
    # Batch composition heavily affects:
    #
    #   • hard negatives
    #   • semantic geometry
    #   • retrieval quality
    #
    # Random batching is substantially weaker.
    #
    sampler = SemanticBatchSampler(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        seed=args.seed,
        rare_bucket_alpha=0.15,
    )

    # ======================================================
    # DataLoader
    # ======================================================
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Training Samples : {len(dataset)}\n")

    # ======================================================
    # Models
    # ======================================================

    # ------------------------------------------------------
    # Vision Encoder
    # ------------------------------------------------------
    #
    # Learns:
    #   multimodal temporal change embeddings
    #
    vision_encoder = VisionEncoder(
        rgb_channels=3,
        sar_channels=8,
        embed_dim=256,
        freeze_rgb_backbone=True,
    ).to(DEVICE)

    # ------------------------------------------------------
    # Text Encoder
    # ------------------------------------------------------
    #
    # Produces:
    #   semantic text embeddings
    #
    text_encoder = TextEncoder(
        embed_dim=256,
        freeze_backbone=True,
    ).to(DEVICE)

    # ------------------------------------------------------
    # Whisper Encoder
    # ------------------------------------------------------
    #
    # Produces:
    #   spoken-language semantic embeddings
    #
    whisper_encoder = WhisperEncoder(
        embed_dim=256,
        freeze_backbone=True,
    ).to(DEVICE)

    # ======================================================
    # Freeze Language Encoders
    # ======================================================
    #
    # IMPORTANT:
    # Only vision encoder is trained.
    #
    # Language encoders remain frozen to:
    #   • stabilize geometry
    #   • preserve language priors
    #   • reduce overfitting
    #
    for p in text_encoder.parameters():
        p.requires_grad = False

    for p in whisper_encoder.parameters():
        p.requires_grad = False

    # ======================================================
    # Finetuning Strategy
    # ======================================================

    # ------------------------------------------------------
    # Frozen baseline
    # ------------------------------------------------------
    if args.finetune == "frozen":

        print("→ Frozen baseline")

        for p in vision_encoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------
    # Full vision finetuning
    # ------------------------------------------------------
    elif args.finetune == "full":

        print("→ Full vision finetuning")

        for p in vision_encoder.parameters():
            p.requires_grad = True

    # ------------------------------------------------------
    # Layer4-only finetuning
    # ------------------------------------------------------
    elif args.finetune == "layer4":

        print("→ Layer4-only finetuning ")

        #
        # Earlier backbone layers remain frozen.
        # Only high-level semantic layers adapt.
        #
        vision_encoder.unfreeze_layer4()

    # ======================================================
    # Count Trainable Parameters
    # ======================================================
    num_trainable = count_trainable_params(
        vision_encoder
    )

    print(
        f"Trainable Vision Params : "
        f"{num_trainable:,}\n"
    )

    # ======================================================
    # Frozen Baseline Logic
    # ======================================================
    #
    # No optimization occurs.
    # Save geometry baseline checkpoint directly.
    #
    if num_trainable == 0:

        print(
            "→ No trainable parameters detected."
        )

        checkpoint_path = os.path.join(
            SAVE_DIR,
            "vision_encoder_epoch0.pt",
        )

        torch.save(
            vision_encoder.state_dict(),
            checkpoint_path,
        )

        print(
            f"✓ Saved frozen baseline checkpoint:\n"
            f"{checkpoint_path}"
        )

        return

    # ======================================================
    # Precompute Language Embeddings
    # ======================================================
    #
    # IMPORTANT:
    # Language encoders are frozen.
    #
    # Therefore:
    # language embeddings can be computed ONCE
    # and reused during training.
    #
    print("Precomputing language embeddings...\n")

    text_encoder.eval()

    whisper_encoder.eval()

    text_tokens = torch.load(
        "data/text_tokens/tokens.pt"
    )

    whisper_tokens = torch.load(
        "data/whisper_tokens/tokens.pt"
    )

    with torch.no_grad():

        z_text = text_encoder(
            text_tokens["input_ids"].to(DEVICE),
            text_tokens["attention_mask"].to(DEVICE),
        )

        z_whisper = whisper_encoder(
            whisper_tokens["input_ids"].to(DEVICE),
            whisper_tokens["attention_mask"].to(DEVICE),
        )

    print(f"z_text shape     : {z_text.shape}")

    print(f"z_whisper shape  : {z_whisper.shape}\n")

    # ======================================================
    # Contrastive Loss
    # ======================================================
    #
    # Uses:
    #   • symmetric InfoNCE
    #   • multi-positive supervision
    #   • vision↔language alignment
    #
    loss_fn = ContrastiveLoss(
        temperature=0.07,
        lambda_v2l=1.0,
        lambda_l2v=1.0,
    )

    # ======================================================
    # Optimizer
    # ======================================================
    trainable_params = list(
        filter(
            lambda p: p.requires_grad,
            vision_encoder.parameters(),
        )
    )

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=LR,
    )

    # ======================================================
    # Metrics Storage
    # ======================================================
    training_metrics = []

    # ======================================================
    # Training Loop
    # ======================================================
    for epoch in range(1, NUM_EPOCHS + 1):

        # --------------------------------------------------
        # Training mode
        # --------------------------------------------------
        vision_encoder.train()

        #
        # IMPORTANT:
        # BatchNorm statistics remain frozen.
        #
        # Helps stabilize partial finetuning.
        #
        vision_encoder.freeze_bn()

        running_loss = 0.0

        running_v2l = 0.0

        running_l2v = 0.0

        # --------------------------------------------------
        # Batch Loop
        # --------------------------------------------------
        for batch in loader:

            # ==============================================
            # Vision Embeddings
            # ==============================================
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

            # ==============================================
            # Contrastive Loss
            # ==============================================
            #
            # Aligns:
            #   vision ↔ language
            #
            loss, metrics = loss_fn(
                z_patch=z_patch,
                patch_buckets=batch["bucket"],
                z_text=z_text,
                z_whisper=z_whisper,
                positive_pools=positive_pools,
            )

            # ==============================================
            # Optimization
            # ==============================================
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # ==============================================
            # Metric Accumulation
            # ==============================================
            running_loss += loss.item()

            running_v2l += (
                metrics["loss_v2l"].item()
            )

            running_l2v += (
                metrics["loss_l2v"].item()
            )

        # --------------------------------------------------
        # Epoch Metrics
        # --------------------------------------------------
        avg_loss = running_loss / len(loader)

        avg_v2l = running_v2l / len(loader)

        avg_l2v = running_l2v / len(loader)

        epoch_metrics = {
            "epoch": epoch,
            "loss_total": avg_loss,
            "loss_v2l": avg_v2l,
            "loss_l2v": avg_l2v,
        }

        training_metrics.append(epoch_metrics)

        # --------------------------------------------------
        # Console Logging
        # --------------------------------------------------
        print(
            f"[Epoch {epoch:02d}/{NUM_EPOCHS}] "
            f"Loss={avg_loss:.6f} | "
            f"V2L={avg_v2l:.6f} | "
            f"L2V={avg_l2v:.6f}"
        )

        # --------------------------------------------------
        # Save Checkpoint
        # --------------------------------------------------
        checkpoint_path = os.path.join(
            SAVE_DIR,
            f"vision_encoder_epoch{epoch}.pt",
        )

        torch.save(
            vision_encoder.state_dict(),
            checkpoint_path,
        )

    # ======================================================
    # Save Metrics
    # ======================================================
    metrics_path = os.path.join(
        SAVE_DIR,
        "metrics.json",
    )

    save_metrics(
        training_metrics,
        metrics_path,
    )

    # ======================================================
    # Final Logging
    # ======================================================
    print("\n===================================================")
    print("Training Complete")
    print("===================================================")

    print(f"Checkpoints : {SAVE_DIR}")

    print(f"Metrics     : {metrics_path}")

    print("===================================================\n")


if __name__ == "__main__":
    main()