#!/usr/bin/env python3
"""
whisper_encoder.py
==================

Project 2A — Whisper Transcript Semantic Encoder
------------------------------------------------

Defines the language encoder used for:
    Whisper ASR transcript embeddings.

This encoder converts:
    spoken disaster narration transcripts

into:
    dense semantic retrieval embeddings.

Pipeline Role
-------------
disaster audio
    ↓
Whisper ASR
    ↓
transcript text
    ↓
DistilBERT tokenizer
    ↓
input_ids + attention_mask
    ↓
WhisperEncoder
    ↓
semantic embedding z_whisper
    ↓
multimodal contrastive alignment

Purpose
-------
This encoder provides:
    retrieval-oriented spoken-language embeddings
for multimodal contrastive learning.

The learned embeddings are aligned with:
    - vision change embeddings
    - written disaster-report embeddings

inside a shared multimodal embedding space.

Important Clarification
-----------------------
This encoder does NOT process raw audio.

Whisper ASR already converted:
    audio → transcript text

before this stage.

This encoder therefore operates purely on:
    natural-language transcript text.

Core Design Goals
-----------------
- spoken-language semantic retrieval
- cross-event generalization
- multimodal alignment
- contrastive learning compatibility
- lightweight transformer inference

Architecture
-------------
Input:
    tokenized Whisper transcripts

Backbone:
    DistilBERT transformer encoder

Pooling:
    masked mean pooling

Projection:
    MLP projection head

Output:
    L2-normalized semantic embedding

Why Separate Whisper Encoder?
-----------------------------
Although both:
    - written reports
    - Whisper transcripts

use DistilBERT,

spoken-language semantics differ from:
    formal written language.

Whisper transcripts often contain:
    - conversational phrasing
    - eyewitness descriptions
    - operational narration
    - informal disaster language

This encoder enables:
    spoken-language semantic supervision.

Important Design Decisions
--------------------------
1. DistilBERT backbone
    lightweight semantic transformer encoder

2. Masked mean pooling
    empirically strong retrieval behavior

3. Projection head
    maps transformer semantics into retrieval geometry

4. L2 normalization
    stabilizes contrastive similarity learning

5. Optional frozen backbone
    stabilizes low-resource contrastive training

Embedding Geometry
------------------
The final output:
    z_whisper ∈ R^d

is optimized so:
    semantically related spoken disaster narratives
cluster together in embedding space.

Example:
    "roads washed away"
    "bridges submerged by floodwaters"

should become nearby embeddings.

Run Command
-----------
This file is imported by training scripts
and is not executed directly.

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


class WhisperEncoder(nn.Module):
    """
    Whisper Transcript Semantic Encoder
    -----------------------------------

    Encodes Whisper ASR transcripts into:
        dense semantic retrieval embeddings.

    Role
    ----
    Converts:
        spoken disaster-language transcripts

    into:
        retrieval-oriented semantic vectors.

    Design Philosophy
    -----------------
    This encoder is designed for:
        - semantic retrieval
        - multimodal alignment
        - contrastive representation learning
        - cross-event generalization

    Spoken-Language Retrieval
    -------------------------
    Unlike written disaster reports,
    Whisper transcripts contain:
        - conversational narration
        - eyewitness phrasing
        - operational spoken language

    This encoder helps capture:
        spoken semantic structure.

    Pooling Strategy
    ----------------
    Uses:
        masked mean pooling

    instead of:
        CLS pooling.

    Rationale:
        mean pooling often produces more stable
        retrieval embeddings for semantic search.

    Output
    ------
    Returns:
        L2-normalized semantic embeddings

    compatible with:
        - vision embeddings
        - text embeddings
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embed_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        """
        Initialize Whisper transcript encoder.

        Args
        ----
        model_name:
            HuggingFace transformer backbone.

        embed_dim:
            Final retrieval embedding dimension.

        freeze_backbone:
            If True:
                transformer weights remain frozen.

            Useful for:
                stable low-resource contrastive learning.
        """

        super().__init__()

        # ======================================================
        # Transformer Backbone
        # ======================================================
        #
        # DistilBERT provides:
        #   - contextual language understanding
        #   - efficient transformer inference
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
        #   - lowers GPU memory usage
        #   - reduces overfitting
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
        to semantic meaning.

        The attention mask ensures:
            only real tokens participate.

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

            Token IDs generated from
            Whisper transcript text.

        attention_mask:
            Shape:
                (N, L)

            Padding mask.

        Returns
        -------
        z_whisper:
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