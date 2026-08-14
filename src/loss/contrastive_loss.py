#!/usr/bin/env python3
"""
contrastive_loss.py
===================

Project 2A — Symmetric Multi-Positive Contrastive Loss
------------------------------------------------------

Defines the core multimodal contrastive objective used for:

    vision ↔ language alignment

where:
    language = text + Whisper transcript embeddings.

This loss learns a shared semantic embedding space in which:

    semantically related disaster representations
    become nearby vectors

regardless of:
    - modality
    - event identity
    - wording style

Core Objective
--------------
The model learns alignment between:

    z_patch    → disaster image embeddings
    z_text     → written-language embeddings
    z_whisper  → spoken-language embeddings

using:
    multi-positive InfoNCE contrastive learning.

Unlike traditional contrastive learning:
    positives are defined by DAMAGE SEMANTICS
    rather than exact sample identity.

Example
-------
A flood-damage image patch may align with:

    "roads submerged after flooding"
    "bridges washed away"
    "entire neighborhoods inundated"

even if:
    - different disasters
    - different countries
    - different events

This encourages:
    cross-event semantic generalization.

Key Features
------------
1. Multi-positive InfoNCE
    Multiple valid positives per anchor.

2. Symmetric retrieval objective
    - vision → language
    - language → vision

3. Shared language candidate space
    Text and Whisper embeddings are unified.

4. Cross-event positives
    Semantics matter more than event identity.

5. Stable logsumexp formulation
    Numerically stable multi-positive aggregation.

6. Safe empty-anchor handling
    Anchors with no positives are skipped.

Contrastive Learning Goal
-------------------------
The loss optimizes embedding geometry such that:

    sim(z_anchor, z_positive) ↑

and:

    sim(z_anchor, z_negative) ↓

using:
    cosine similarity on L2-normalized embeddings.

Mathematical Formulation
------------------------

Similarity:
    sim(a,b) = a^T b

since embeddings are already normalized.

Multi-positive numerator:

                 Σ exp(sim(a,p)/τ)
    positives p

InfoNCE objective:

            numerator
    -log ----------------
          all candidates

where:
    τ = temperature parameter.

Embedding Requirements
----------------------
IMPORTANT:
All embeddings MUST already be:

    L2-normalized

before entering this loss.

Run Command
-----------
This file is imported during training
and is not executed directly.

Example training usage:

PYTHONPATH=2A_v2 python 2A_v2/scripts/train_encoder.py \
    --finetune layer4 \
    --positive multi

Dependencies
------------
pip install torch
"""

# ==============================================================
# Imports
# ==============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Symmetric Multi-Positive Contrastive Loss
    -----------------------------------------

    Learns a shared multimodal semantic embedding space.

    Objective
    ---------
    Align:
        vision ↔ text ↔ whisper

    using:
        semantic similarity rather than exact identity.

    Core Idea
    ---------
    Embeddings sharing:
        similar disaster semantics

    should become:
        nearby vectors in embedding space.

    Example:
        flooded imagery
            ↔
        "roads submerged"

    even across:
        different disaster events.

    Major Design Features
    ---------------------
    1. Multi-positive supervision
        Multiple valid semantic positives.

    2. Symmetric retrieval learning
        Both retrieval directions are optimized.

    3. Shared language candidate space
        Text + Whisper treated jointly.

    4. Cross-event retrieval
        Event identity is NOT the target.

    5. Numerically stable formulation
        Uses logsumexp aggregation.

    6. Robust empty-anchor handling
        Invalid anchors safely skipped.

    Important Assumption
    --------------------
    All embeddings must already be:
        L2-normalized.

    This allows:
        cosine similarity = dot product.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_v2l: float = 1.0,
        lambda_l2v: float = 1.0,
    ):
        """
        Initialize contrastive loss.

        Args
        ----
        temperature:
            InfoNCE temperature scaling factor.

            Smaller temperature:
                sharper similarity distribution.

        lambda_v2l:
            Weight for:
                vision → language loss.

        lambda_l2v:
            Weight for:
                language → vision loss.
        """

        super().__init__()

        # ------------------------------------------------------
        # Temperature parameter
        #
        # Controls sharpness of similarity distribution.
        #
        # Lower τ:
        #   stronger contrastive separation.
        # ------------------------------------------------------
        self.temperature = temperature

        # ------------------------------------------------------
        # Directional loss weights
        # ------------------------------------------------------
        self.lambda_v2l = lambda_v2l
        self.lambda_l2v = lambda_l2v

    # ==========================================================
    # Build Shared Language Candidate Space
    # ==========================================================
    def build_language_space(
        self,
        z_text: torch.Tensor,
        z_whisper: torch.Tensor,
    ):
        """
        Build unified language embedding space.

        Purpose
        -------
        Combines:
            text embeddings
            +
            whisper embeddings

        into:
            one shared candidate space.

        Why?
        ----
        Both modalities represent:
            language semantics.

        This enables:
            modality-agnostic retrieval learning.

        Inputs
        ------
        z_text:
            Shape:
                (Nt, D)

            Text embeddings.

        z_whisper:
            Shape:
                (Nw, D)

            Whisper transcript embeddings.

        Returns
        -------
        z_lang:
            Shape:
                (N, D)

            Concatenated language embeddings.

        offsets:
            Dict mapping modality ranges.

            Example:
                {
                    "text": (0, Nt),
                    "whisper": (Nt, Nt+Nw)
                }
        """

        candidates = []
        offsets = {}

        offset = 0

        # ------------------------------------------------------
        # Add text embeddings
        # ------------------------------------------------------
        if z_text is not None and len(z_text) > 0:

            candidates.append(z_text)

            offsets["text"] = (
                offset,
                offset + len(z_text),
            )

            offset += len(z_text)

        # ------------------------------------------------------
        # Add Whisper embeddings
        # ------------------------------------------------------
        if z_whisper is not None and len(z_whisper) > 0:

            candidates.append(z_whisper)

            offsets["whisper"] = (
                offset,
                offset + len(z_whisper),
            )

            offset += len(z_whisper)

        # ------------------------------------------------------
        # Safety check
        # ------------------------------------------------------
        if not candidates:

            raise ValueError(
                "No language embeddings provided."
            )

        # ------------------------------------------------------
        # Concatenate language embeddings
        # ------------------------------------------------------
        z_lang = torch.cat(candidates, dim=0)

        return z_lang, offsets

    # ==========================================================
    # Build Language Bucket Mapping
    # ==========================================================
    def build_language_bucket_map(
        self,
        positive_pools: dict,
        z_lang: torch.Tensor,
    ):
        """
        Build language-index → semantic-bucket mapping.

        Purpose
        -------
        Tracks which semantic bucket
        each language embedding belongs to.

        Example:
            flooding
            wildfire
            structural_damage

        Returns
        -------
        lang_bucket_map:
            Dictionary:
                (modality, idx) → bucket
        """

        lang_bucket_map = {}

        for bucket, pool in positive_pools.items():

            # --------------------------------------------------
            # Text bucket assignments
            # --------------------------------------------------
            if "text" in pool:

                for idx in pool["text"]:

                    lang_bucket_map[("text", idx)] = bucket

            # --------------------------------------------------
            # Whisper bucket assignments
            # --------------------------------------------------
            if "whisper" in pool:

                for idx in pool["whisper"]:

                    lang_bucket_map[("whisper", idx)] = bucket

        return lang_bucket_map

    # ==========================================================
    # Vision → Language Loss
    # ==========================================================
    def vision_to_language_loss(
        self,
        z_patch: torch.Tensor,
        patch_buckets: list,
        z_lang: torch.Tensor,
        positive_pools: dict,
        offsets: dict,
    ):
        """
        Multi-positive InfoNCE:
            vision anchor → language candidates.

        Objective
        ---------
        Pull:
            semantically matching language embeddings

        toward:
            disaster image embeddings.

        Inputs
        ------
        z_patch:
            Shape:
                (B, D)

            Vision embeddings.

        patch_buckets:
            Semantic bucket labels per patch.

        z_lang:
            Shared language embedding space.

        positive_pools:
            Bucket → positive indices.

        offsets:
            Modality index offsets.

        Returns
        -------
        loss:
            Scalar contrastive loss.
        """

        device = z_patch.device

        losses = []

        # ------------------------------------------------------
        # Similarity matrix
        #
        # Shape:
        #   (B, N)
        #
        # Since embeddings are normalized:
        #   dot product = cosine similarity.
        # ------------------------------------------------------
        sim_all = (
            torch.matmul(z_patch, z_lang.T)
            / self.temperature
        )

        # ------------------------------------------------------
        # Compute per-anchor InfoNCE
        # ------------------------------------------------------
        for i, bucket in enumerate(patch_buckets):

            pool = positive_pools.get(bucket, None)

            if pool is None:
                continue

            pos_indices = []

            # --------------------------------------------------
            # Text positives
            # --------------------------------------------------
            if "text" in pool and "text" in offsets:

                start, _ = offsets["text"]

                pos_indices += [
                    start + idx
                    for idx in pool["text"]
                ]

            # --------------------------------------------------
            # Whisper positives
            # --------------------------------------------------
            if "whisper" in pool and "whisper" in offsets:

                start, _ = offsets["whisper"]

                pos_indices += [
                    start + idx
                    for idx in pool["whisper"]
                ]

            # --------------------------------------------------
            # Skip anchors without positives
            # --------------------------------------------------
            if len(pos_indices) == 0:
                continue

            # --------------------------------------------------
            # Numerator:
            # aggregate positive similarities
            #
            # logsumexp improves:
            #   numerical stability
            #   multi-positive aggregation
            # --------------------------------------------------
            pos_sim = sim_all[i, pos_indices]

            numerator = torch.logsumexp(
                pos_sim,
                dim=0,
            )

            # --------------------------------------------------
            # Denominator:
            # all candidates
            # --------------------------------------------------
            denominator = torch.logsumexp(
                sim_all[i],
                dim=0,
            )

            # --------------------------------------------------
            # InfoNCE loss
            # --------------------------------------------------
            loss_i = -(numerator - denominator)

            losses.append(loss_i)

        # ------------------------------------------------------
        # Safe empty return
        # ------------------------------------------------------
        if not losses:

            return torch.tensor(
                0.0,
                device=device,
                requires_grad=True,
            )

        return torch.stack(losses).mean()

    # ==========================================================
    # Language → Vision Loss
    # ==========================================================
    def language_to_vision_loss(
        self,
        z_patch: torch.Tensor,
        patch_buckets: list,
        z_lang: torch.Tensor,
        positive_pools: dict,
        offsets: dict,
    ):
        """
        Symmetric multi-positive InfoNCE:
            language anchor → vision candidates.

        Purpose
        -------
        Enables:
            bidirectional retrieval learning.

        This improves:
            multimodal alignment symmetry.

        Returns
        -------
        loss:
            Scalar contrastive loss.
        """

        device = z_patch.device

        losses = []

        # ------------------------------------------------------
        # Build reverse mapping:
        # language idx → bucket
        # ------------------------------------------------------
        lang_to_bucket = {}

        # ------------------------------------------------------
        # Text mappings
        # ------------------------------------------------------
        if "text" in offsets:

            start, _ = offsets["text"]

            for bucket, pool in positive_pools.items():

                if "text" in pool:

                    for idx in pool["text"]:

                        lang_to_bucket[start + idx] = bucket

        # ------------------------------------------------------
        # Whisper mappings
        # ------------------------------------------------------
        if "whisper" in offsets:

            start, _ = offsets["whisper"]

            for bucket, pool in positive_pools.items():

                if "whisper" in pool:

                    for idx in pool["whisper"]:

                        lang_to_bucket[start + idx] = bucket

        # ------------------------------------------------------
        # Similarity matrix
        #
        # Shape:
        #   (N, B)
        # ------------------------------------------------------
        sim_all = (
            torch.matmul(z_lang, z_patch.T)
            / self.temperature
        )

        # ------------------------------------------------------
        # Per-language-anchor InfoNCE
        # ------------------------------------------------------
        for lang_idx in range(len(z_lang)):

            bucket = lang_to_bucket.get(lang_idx, None)

            if bucket is None:
                continue

            # --------------------------------------------------
            # Positive vision matches
            # --------------------------------------------------
            pos_patch_indices = [
                i
                for i, b in enumerate(patch_buckets)
                if b == bucket
            ]

            if len(pos_patch_indices) == 0:
                continue

            # --------------------------------------------------
            # Numerator
            # --------------------------------------------------
            pos_sim = sim_all[
                lang_idx,
                pos_patch_indices,
            ]

            numerator = torch.logsumexp(
                pos_sim,
                dim=0,
            )

            # --------------------------------------------------
            # Denominator
            # --------------------------------------------------
            denominator = torch.logsumexp(
                sim_all[lang_idx],
                dim=0,
            )

            loss_i = -(numerator - denominator)

            losses.append(loss_i)

        # ------------------------------------------------------
        # Safe empty return
        # ------------------------------------------------------
        if not losses:

            return torch.tensor(
                0.0,
                device=device,
                requires_grad=True,
            )

        return torch.stack(losses).mean()

    # ==========================================================
    # Forward Pass
    # ==========================================================
    def forward(
        self,
        z_patch: torch.Tensor,            # (B, D)
        patch_buckets: list,              # length B
        z_text: torch.Tensor,             # (Nt, D)
        z_whisper: torch.Tensor,          # (Nw, D)
        positive_pools: dict,             # bucket -> indices
    ):
        """
        Forward pass for symmetric multimodal InfoNCE.

        Inputs
        ------
        z_patch:
            Vision embeddings.

        patch_buckets:
            Semantic bucket labels.

        z_text:
            Text embeddings.

        z_whisper:
            Whisper embeddings.

        positive_pools:
            Semantic supervision pools.

        Returns
        -------
        total_loss:
            Final weighted contrastive loss.

        metrics:
            Dictionary of directional losses.
        """

        # ======================================================
        # Build unified language embedding space
        # ======================================================
        z_lang, offsets = self.build_language_space(
            z_text,
            z_whisper,
        )

        # ======================================================
        # Vision → Language
        # ======================================================
        loss_v2l = self.vision_to_language_loss(
            z_patch=z_patch,
            patch_buckets=patch_buckets,
            z_lang=z_lang,
            positive_pools=positive_pools,
            offsets=offsets,
        )

        # ======================================================
        # Language → Vision
        # ======================================================
        loss_l2v = self.language_to_vision_loss(
            z_patch=z_patch,
            patch_buckets=patch_buckets,
            z_lang=z_lang,
            positive_pools=positive_pools,
            offsets=offsets,
        )

        # ======================================================
        # Final weighted objective
        # ======================================================
        total_loss = (
            self.lambda_v2l * loss_v2l
            +
            self.lambda_l2v * loss_l2v
        )

        # ------------------------------------------------------
        # Metrics for logging/debugging
        # ------------------------------------------------------
        metrics = {
            "loss_total": total_loss.detach(),
            "loss_v2l": loss_v2l.detach(),
            "loss_l2v": loss_l2v.detach(),
        }

        return total_loss, metrics