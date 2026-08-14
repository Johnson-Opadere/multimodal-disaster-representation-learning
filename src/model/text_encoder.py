#!/usr/bin/env python3
"""
text_encoder.py
================

Project 2A — Text Semantic Encoder
----------------------------------

Defines the language encoder used for:
    written disaster-report semantic embeddings.

The encoder transforms:
    tokenized disaster-language fragments
into:
    dense semantic retrieval embeddings.

Pipeline Role
-------------
text fragments
    ↓
DistilBERT tokenizer
    ↓
input_ids + attention_mask
    ↓
TextEncoder
    ↓
semantic embedding z_text
    ↓
multimodal contrastive alignment

Purpose
-------
This encoder provides:
    retrieval-oriented semantic language representations
for multimodal contrastive learning.

The learned embeddings are aligned with:
    - vision change embeddings
    - Whisper transcript embeddings

inside a shared multimodal embedding space.

Core Design Goals
-----------------
- semantic retrieval
- cross-event generalization
- modality alignment
- stable contrastive supervision
- lightweight transformer inference

Architecture
-------------
Input:
    tokenized text fragments

Backbone:
    DistilBERT transformer encoder

Pooling:
    masked mean pooling

Projection:
    MLP projection head

Output:
    L2-normalized semantic embedding

Important Design Decisions
--------------------------
1. DistilBERT backbone
    lightweight semantic transformer encoder

2. Masked mean pooling
    empirically stronger retrieval behavior than CLS pooling

3. Projection head
    maps transformer semantics into retrieval geometry

4. L2 normalization
    stabilizes contrastive similarity learning

5. Optional frozen backbone
    improves stability for low-resource contrastive training

Embedding Geometry
------------------
The final output:
    z_text ∈ R^d

is optimized so:
    semantically related disaster descriptions
cluster together in embedding space.

Example:
    "roads flooded"
    "bridges submerged"

should become nearby embeddings.

Run Command
-----------
This file is imported by training scripts and is not run directly.

Example training usage:

PYTHONPATH=2A_v2 python 2A_v2/scripts/train_encoder.py \
    --finetune layer4 \
    --positive multi

Dependencies
------------
pip install torch transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Text Semantic Encoder
    ---------------------

    Frozen DistilBERT-based semantic encoder
    for multimodal retrieval learning.

    Role
    ----
    Converts:
        disaster-language fragments

    into:
        dense semantic retrieval embeddings.

    Design Philosophy
    -----------------
    The encoder is designed for:
        - retrieval geometry
        - semantic clustering
        - multimodal alignment
        - contrastive learning

    Retrieval-Oriented Design
    -------------------------
    Unlike standard classification NLP pipelines,
    this encoder optimizes:
        embedding geometry

    rather than:
        class logits.

    Pooling Strategy
    ----------------
    Uses:
        masked mean pooling

    instead of:
        CLS pooling.

    Rationale:
        mean pooling often produces more stable
        semantic retrieval embeddings.

    Output
    ------
    Returns:
        L2-normalized embeddings

    compatible with:
        vision Δf embedding space.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embed_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        """
        Initialize text encoder.

        Args
        ----
        model_name:
            HuggingFace transformer backbone name.

        embed_dim:
            Final retrieval embedding dimension.

        freeze_backbone:
            If True:
                transformer backbone weights remain frozen.

            Useful for:
                stable low-resource contrastive learning.
        """

        super().__init__()

        # ======================================================
        # Transformer Backbone
        # ======================================================
        #
        # DistilBERT provides:
        #   - contextual language representations
        #   - lightweight inference
        #   - strong semantic priors
        #
        # ======================================================
        self.backbone = AutoModel.from_pretrained(
            model_name
        )

        # ------------------------------------------------------
        # Transformer hidden dimension
        # ------------------------------------------------------
        hidden_dim = self.backbone.config.hidden_size

        # ======================================================
        # Optional Backbone Freezing
        # ======================================================
        #
        # Frozen transformer training:
        #   - stabilizes optimization
        #   - reduces overfitting
        #   - lowers GPU memory usage
        #
        # ======================================================
        if freeze_backbone:

            for p in self.backbone.parameters():

                p.requires_grad = False

        # ======================================================
        # Projection Head
        # ======================================================
        #
        # Maps transformer semantic features
        # into:
        #   retrieval-oriented embedding geometry.
        #
        # Architecture:
        #   Linear → ReLU → Linear
        #
        # ======================================================
        self.proj = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, embed_dim),
        )

    # ==========================================================
    # Masked Mean Pooling
    # ==========================================================
    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform masked mean pooling.

        Purpose
        -------
        Aggregates token-level transformer outputs
        into:
            one sentence-level semantic embedding.

        Why Masked Pooling?
        -------------------
        Padding tokens should NOT contribute
        to semantic representations.

        The attention mask ensures:
            only valid tokens participate.

        Args
        ----
        token_embeddings:
            Shape:
                (N, L, hidden_dim)

            Transformer token embeddings.

        attention_mask:
            Shape:
                (N, L)

            Binary mask:
                1 = valid token
                0 = padding token

        Returns
        -------
        pooled:
            Shape:
                (N, hidden_dim)

            Sentence-level semantic embeddings.
        """

        # ------------------------------------------------------
        # Expand attention mask
        #
        # (N, L)
        #     →
        # (N, L, 1)
        #
        # Enables broadcasting across embedding dimension.
        # ------------------------------------------------------
        mask = attention_mask.unsqueeze(-1).float()

        # ------------------------------------------------------
        # Remove padding-token contribution
        # ------------------------------------------------------
        masked_embeddings = token_embeddings * mask

        # ------------------------------------------------------
        # Sum valid token embeddings
        # ------------------------------------------------------
        sum_embeddings = masked_embeddings.sum(dim=1)

        # ------------------------------------------------------
        # Count valid tokens
        #
        # clamp prevents divide-by-zero instability.
        # ------------------------------------------------------
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)

        # ------------------------------------------------------
        # Mean pooling
        # ------------------------------------------------------
        pooled = sum_embeddings / sum_mask

        return pooled

    # ==========================================================
    # Forward Pass
    # ==========================================================
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Input
        -----
        input_ids:
            Shape:
                (N, L)

            Transformer token IDs.

        attention_mask:
            Shape:
                (N, L)

            Padding mask.

        Returns
        -------
        z_text:
            Shape:
                (N, embed_dim)

            L2-normalized semantic retrieval embeddings.
        """

        # ======================================================
        # Transformer Forward Pass
        # ======================================================
        #
        # DistilBERT contextualizes token representations
        # using self-attention.
        #
        # ======================================================
        outputs = self.backbone(

            input_ids=input_ids,

            attention_mask=attention_mask,
        )

        # ======================================================
        # Token-Level Contextual Embeddings
        # ======================================================
        #
        # Shape:
        #   (N, L, hidden_dim)
        #
        # Each token embedding is contextualized
        # by surrounding tokens via self-attention.
        #
        # ======================================================
        token_embeddings = outputs.last_hidden_state

        # ======================================================
        # Sentence-Level Pooling
        # ======================================================
        pooled = self.mean_pooling(

            token_embeddings,

            attention_mask,
        )

        # ======================================================
        # Projection Head
        # ======================================================
        #
        # Maps transformer semantics
        # into retrieval embedding space.
        #
        # ======================================================
        z = self.proj(pooled)

        # ======================================================
        # L2 Normalization
        # ======================================================
        #
        # Places embeddings on unit hypersphere.
        #
        # Benefits:
        #   - stable cosine similarity
        #   - improved retrieval geometry
        #   - stabilized contrastive learning
        #
        # ======================================================
        z = F.normalize(z, dim=-1)

        return z