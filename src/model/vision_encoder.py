#!/usr/bin/env python3
"""
vision_encoder.py
=================

Project 2A — Change-Aware Multimodal Vision Encoder
---------------------------------------------------

Defines the multimodal vision encoder used for:
    disaster-change semantic representation learning.

The encoder processes:

    RGB_pre
    RGB_post
    SAR_pre
    SAR_post

and learns:
    a unified semantic disaster-change embedding.

Pipeline Role
-------------
multimodal disaster imagery
    ↓
VisionEncoder
    ↓
change-aware embedding z_patch
    ↓
multimodal contrastive alignment

Purpose
-------
This encoder learns:
    retrieval-oriented disaster-change embeddings

for:
    - semantic retrieval
    - cross-event similarity learning
    - multimodal contrastive alignment
    - retrieval-augmented reasoning

Core Design Philosophy
----------------------
Instead of performing:
    segmentation or classification,

the encoder learns:
    embedding geometry.

Goal:
    semantically similar disaster changes
    become nearby vectors in embedding space.

Example:
    flooded urban regions
should retrieve:
    semantically similar flooding events
across different disasters.

Architecture Overview
---------------------

RGB Branch
----------
Uses:
    pretrained ResNet18

for:
    semantic RGB feature extraction.

Pipeline:
    RGB_pre  → shared ResNet18 → f_rgb_pre
    RGB_post → shared ResNet18 → f_rgb_post

Change representation:
    Δ_rgb = f_rgb_post - f_rgb_pre

SAR Branch
----------
Uses:
    lightweight CNN encoder

for:
    SAR feature extraction.

Pipeline:
    SAR_pre  → shared SAR encoder → f_sar_pre
    SAR_post → shared SAR encoder → f_sar_post

Change representation:
    Δ_sar = f_sar_post - f_sar_pre

Fusion
------
The RGB and SAR temporal-difference features are fused:

    concat(Δ_rgb, Δ_sar)
            ↓
        fusion MLP

Projection Head
---------------
The fusion representation is projected into:
    retrieval embedding space.

Output:
    z_patch ∈ R^256

L2-normalized for:
    cosine-similarity retrieval.

Why Temporal Difference?
------------------------
The encoder explicitly models:
    disaster-induced change.

Instead of:
    static appearance learning.

Temporal differencing improves:
    - disaster change sensitivity
    - semantic retrieval
    - multimodal alignment

Important Design Decisions
--------------------------
1. Shared RGB encoder
    ensures consistent temporal feature extraction

2. Shared SAR encoder
    preserves temporal SAR comparability

3. Explicit temporal differencing
    models disaster change directly

4. Late multimodal fusion
    preserves modality-specific learning

5. Projection head
    optimizes retrieval geometry

6. L2 normalization
    stabilizes contrastive learning

7. Partial backbone finetuning support
    enables controlled adaptation

Embedding Geometry
------------------
The final embedding:
    z_patch ∈ R^d

is optimized so:
    semantically similar disaster changes
cluster together in embedding space.

Example:
    flood damage
should cluster with:
    flood damage from other events.

Run Command
-----------
This file is imported by training scripts
and is not executed directly.

Example training usage:

PYTHONPATH=2A python 2A/scripts/train_encoder.py \
    --finetune layer4 \
    --positive multi

Dependencies
------------
pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class VisionEncoder(nn.Module):
    """
    Change-Aware Multimodal Vision Encoder
    -------------------------------------

    Learns semantic disaster-change embeddings from:

        RGB_pre
        RGB_post
        SAR_pre
        SAR_post

    Goal
    ----
    Produce:
        retrieval-oriented disaster-change embeddings

    suitable for:
        - multimodal contrastive learning
        - semantic retrieval
        - cross-event similarity learning

    Retrieval-Oriented Design
    -------------------------
    Unlike segmentation networks,
    this encoder learns:
        embedding geometry

    rather than:
        pixel predictions.

    Temporal Change Modeling
    ------------------------
    The encoder explicitly computes:

        post-disaster features
        minus
        pre-disaster features

    to isolate:
        disaster-induced change semantics.

    Multimodal Fusion
    -----------------
    RGB and SAR are processed separately
    before:
        late fusion.

    This preserves:
        modality-specific feature learning.

    Output
    ------
    Returns:
        L2-normalized disaster embeddings

    compatible with:
        - text embeddings
        - Whisper embeddings
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        sar_channels: int = 8,
        backbone_out_dim: int = 512,
        sar_feature_dim: int = 512,
        fusion_dim: int = 512,
        embed_dim: int = 256,
        freeze_rgb_backbone: bool = True,
    ):
        """
        Initialize multimodal vision encoder.

        Args
        ----
        rgb_channels:
            Number of RGB channels.

        sar_channels:
            Number of SAR channels.

        backbone_out_dim:
            Output feature dimension from ResNet18.

        sar_feature_dim:
            Output feature dimension from SAR encoder.

        fusion_dim:
            Hidden dimension used during multimodal fusion.

        embed_dim:
            Final retrieval embedding dimension.

        freeze_rgb_backbone:
            If True:
                freezes pretrained RGB backbone.

            Useful for:
                stable low-resource training.
        """

        super().__init__()

        # ======================================================
        # RGB BACKBONE
        # ======================================================
        #
        # Pretrained ResNet18 used for:
        #   semantic RGB feature extraction.
        #
        # Benefits:
        #   - strong pretrained visual priors
        #   - stable low-resource learning
        #   - efficient inference
        #
        # ======================================================
        rgb_backbone = resnet18(pretrained=True)

        # ------------------------------------------------------
        # Remove classification head
        #
        # ResNet now outputs:
        #   feature embeddings
        #
        # instead of:
        #   class logits.
        # ------------------------------------------------------
        rgb_backbone.fc = nn.Identity()

        self.rgb_backbone = rgb_backbone

        # ======================================================
        # Optional Backbone Freezing
        # ======================================================
        #
        # Frozen RGB training:
        #   - stabilizes optimization
        #   - reduces overfitting
        #   - lowers GPU memory usage
        #
        # ======================================================
        if freeze_rgb_backbone:

            for param in self.rgb_backbone.parameters():

                param.requires_grad = False

        # ======================================================
        # SAR ENCODER
        # ======================================================
        #
        # Lightweight CNN encoder for:
        #   SAR feature extraction.
        #
        # Unlike RGB:
        #   no ImageNet-pretrained SAR backbone exists.
        #
        # Architecture:
        #   Conv → BN → ReLU stacks
        #
        # Final output:
        #   global SAR feature vector.
        #
        # ======================================================
        self.sar_encoder = nn.Sequential(

            # --------------------------------------------------
            # Low-level SAR feature extraction
            # --------------------------------------------------
            nn.Conv2d(
                sar_channels,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # --------------------------------------------------
            # Mid-level SAR feature extraction
            # --------------------------------------------------
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # --------------------------------------------------
            # High-level SAR semantic features
            # --------------------------------------------------
            nn.Conv2d(
                256,
                sar_feature_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(sar_feature_dim),
            nn.ReLU(inplace=True),

            # --------------------------------------------------
            # Global feature aggregation
            # --------------------------------------------------
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
        )

        # ======================================================
        # MULTIMODAL FUSION
        # ======================================================
        #
        # Fuses:
        #   Δ_rgb + Δ_sar
        #
        # into:
        #   shared multimodal representation.
        #
        # ======================================================
        self.fusion = nn.Sequential(

            nn.Linear(
                backbone_out_dim + sar_feature_dim,
                fusion_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        # ======================================================
        # PROJECTION HEAD
        # ======================================================
        #
        # Maps fused semantic features
        # into:
        #   retrieval-oriented embedding geometry.
        #
        # ======================================================
        self.proj = nn.Sequential(

            nn.Linear(
                fusion_dim,
                fusion_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                fusion_dim,
                embed_dim
            ),
        )

    # ==========================================================
    # Fine-Tuning Helper
    # ==========================================================
    def unfreeze_layer4(self):
        """
        Unfreeze only layer4 of ResNet18.

        Purpose
        -------
        Enables:
            partial backbone finetuning.

        Benefits
        --------
        - preserves pretrained low-level features
        - adapts high-level semantics
        - improves retrieval specialization
        """

        for name, param in self.rgb_backbone.named_parameters():

            if name.startswith("layer4"):

                param.requires_grad = True

    # ==========================================================
    # Optional BatchNorm Freeze
    # ==========================================================
    def freeze_bn(self):
        """
        Freeze BatchNorm running statistics.

        Purpose
        -------
        Prevent unstable BatchNorm updates during:
            low-batch contrastive training.

        Common retrieval-training stabilization technique.
        """

        for module in self.rgb_backbone.modules():

            if isinstance(module, nn.BatchNorm2d):

                module.eval()

    # ==========================================================
    # RGB ENCODING
    # ==========================================================
    def encode_rgb(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode RGB image.

        Args
        ----
        x:
            Shape:
                (N, 3, H, W)

        Returns
        -------
        rgb_features:
            Shape:
                (N, backbone_out_dim)
        """

        return self.rgb_backbone(x)

    # ==========================================================
    # SAR ENCODING
    # ==========================================================
    def encode_sar(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode SAR image.

        Args
        ----
        x:
            Shape:
                (N, sar_channels, H, W)

        Returns
        -------
        sar_features:
            Shape:
                (N, sar_feature_dim)
        """

        return self.sar_encoder(x)

    # ==========================================================
    # FORWARD PASS
    # ==========================================================
    def forward(
        self,
        rgb_pre: torch.Tensor,
        sar_pre: torch.Tensor,
        rgb_post: torch.Tensor,
        sar_post: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Inputs
        ------
        rgb_pre:
            Pre-disaster RGB image.

        sar_pre:
            Pre-disaster SAR image.

        rgb_post:
            Post-disaster RGB image.

        sar_post:
            Post-disaster SAR image.

        Returns
        -------
        z_patch:
            Shape:
                (N, embed_dim)

            L2-normalized disaster-change embeddings.
        """

        # ======================================================
        # RGB FEATURE EXTRACTION
        # ======================================================
        f_rgb_pre = self.encode_rgb(rgb_pre)

        f_rgb_post = self.encode_rgb(rgb_post)

        # ======================================================
        # SAR FEATURE EXTRACTION
        # ======================================================
        f_sar_pre = self.encode_sar(sar_pre)

        f_sar_post = self.encode_sar(sar_post)

        # ======================================================
        # TEMPORAL DIFFERENCING
        # ======================================================
        #
        # Explicitly models:
        #   disaster-induced change.
        #
        # ======================================================
        delta_rgb = f_rgb_post - f_rgb_pre

        delta_sar = f_sar_post - f_sar_pre

        # ======================================================
        # MULTIMODAL FUSION
        # ======================================================
        delta = torch.cat(
            [delta_rgb, delta_sar],
            dim=1
        )

        fused = self.fusion(delta)

        # ======================================================
        # PROJECTION HEAD
        # ======================================================
        z = self.proj(fused)

        # ======================================================
        # L2 NORMALIZATION
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