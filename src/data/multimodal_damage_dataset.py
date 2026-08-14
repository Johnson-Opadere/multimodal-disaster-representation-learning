#!/usr/bin/env python3
"""
multimodal_damage_dataset.py
============================

Project 2A_v2 — Multimodal Vision Dataset
-----------------------------------------

Defines the PyTorch dataset used for:
    multimodal disaster-change representation learning.

This dataset provides:
    - pre/post RGB imagery
    - pre/post SAR imagery
    - semantic damage buckets
    - lightweight metadata

for:
    multimodal contrastive learning.

Core Purpose
------------
The dataset supplies:
    change-aware multimodal vision inputs

to the:
    VisionEncoder

during:
    semantic retrieval training.

Unlike segmentation datasets,
this dataset does NOT provide:
    - masks
    - bounding boxes
    - pixel labels
    - class logits

Instead, supervision is:
    semantic bucket based.

Input Modalities
----------------

1. RGB Pre-disaster
    Optical imagery before disaster.

2. RGB Post-disaster
    Optical imagery after disaster.

3. SAR Pre-disaster
    Radar imagery before disaster.

4. SAR Post-disaster
    Radar imagery after disaster.

Together these enable:
    temporal disaster-change modeling.

Directory Structure
-------------------

normalized_data/
    train/
        rgb_pre_norm/
        rgb_post_norm/
        sar_pre_norm/
        sar_post_norm/

    hold/
        ...

    test/
        ...

Expected File Naming
--------------------

RGB:
    hurricane-harvey_00000000_pre_disaster_norm.npy
    hurricane-harvey_00000000_post_disaster_norm.npy

SAR:
    hurricane-harvey_00000000_pre_sar_norm.npy
    hurricane-harvey_00000000_post_sar_norm.npy

Returned Sample Structure
-------------------------

{
    "rgb_pre": tensor,
    "rgb_post": tensor,
    "sar_pre": tensor,
    "sar_post": tensor,

    "bucket": damage_bucket,

    "event": event_name,
    "sample_id": patch_id
}

Important Design Philosophy
---------------------------
The dataset intentionally avoids:
    event identity supervision.

The model learns:
    semantic damage alignment

rather than:
    event memorization.

Example:
    flooding in Hurricane Harvey
should align with:
    flooding in Midwest Flooding.

Metadata Usage
--------------
Metadata is returned ONLY for:
    - debugging
    - retrieval auditing
    - leakage analysis
    - evaluation
    - visualization

NOT for:
    training supervision.

Tensor Shape Convention
-----------------------

Stored .npy arrays:
    (H, W, C)

Converted tensors:
    (C, H, W)

This matches:
    PyTorch CNN convention.

Run Command
-----------
This dataset is imported during training
and is not executed directly.

Example usage:

PYTHONPATH=2A_v2 python 2A_v2/scripts/train_encoder.py \
    --finetune layer4 \
    --positive multi

Dependencies
------------
pip install torch numpy
"""

# ==============================================================
# Imports
# ==============================================================

import os
import numpy as np
import torch

from torch.utils.data import Dataset


class MultimodalDamageDataset(Dataset):
    """
    Multimodal Disaster Representation Dataset
    ------------------------------------------

    Provides:
        change-aware multimodal disaster inputs

    for:
        semantic retrieval training.

    Modalities
    ----------
    - RGB pre-disaster
    - RGB post-disaster
    - SAR pre-disaster
    - SAR post-disaster

    Learning Objective
    ------------------
    The dataset supports:
        semantic contrastive learning

    rather than:
        segmentation or classification.

    Key Philosophy
    --------------
    Event identity is NOT the target.

    Instead:
        semantic damage type
    defines supervision.

    Example:
        flooding patches across different disasters
    should align together.

    Metadata
    --------
    Metadata fields are returned for:
        - debugging
        - auditing
        - evaluation
        - retrieval analysis

    NOT for:
        direct supervision.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        event_to_bucket: dict,
        verify_shapes: bool = False,
    ):
        """
        Initialize multimodal dataset.

        Args
        ----
        root_dir:
            Path to:
                normalized_data/

        split:
            Dataset split:
                train / hold / test

        event_to_bucket:
            Mapping:
                event_name → semantic damage bucket

            Example:
                {
                    "hurricane-harvey": "flooding"
                }

        verify_shapes:
            If True:
                prints one-time tensor shape verification.

            Useful for:
                debugging preprocessing pipelines.
        """

        # ------------------------------------------------------
        # Root split directory
        # ------------------------------------------------------
        self.root = os.path.join(root_dir, split)

        # ------------------------------------------------------
        # Event → semantic bucket mapping
        # ------------------------------------------------------
        self.event_to_bucket = event_to_bucket

        # ------------------------------------------------------
        # Optional debugging utilities
        # ------------------------------------------------------
        self.verify_shapes = verify_shapes

        self._verified_shapes = False

        # ======================================================
        # Modality Directories
        # ======================================================

        # ------------------------------------------------------
        # RGB pre-disaster directory
        # ------------------------------------------------------
        self.rgb_pre_dir = os.path.join(
            self.root,
            "rgb_pre_norm",
        )

        # ------------------------------------------------------
        # RGB post-disaster directory
        # ------------------------------------------------------
        self.rgb_post_dir = os.path.join(
            self.root,
            "rgb_post_norm",
        )

        # ------------------------------------------------------
        # SAR pre-disaster directory
        # ------------------------------------------------------
        self.sar_pre_dir = os.path.join(
            self.root,
            "sar_pre_norm",
        )

        # ------------------------------------------------------
        # SAR post-disaster directory
        # ------------------------------------------------------
        self.sar_post_dir = os.path.join(
            self.root,
            "sar_post_norm",
        )

        # ======================================================
        # Build Sample IDs
        # ======================================================
        #
        # Example filename:
        #
        #   hurricane-harvey_00000000_pre_disaster_norm.npy
        #
        # Converted sample ID:
        #
        #   hurricane-harvey_00000000
        #
        # ======================================================
        self.ids = sorted(
            fname.replace(
                "_pre_disaster_norm.npy",
                ""
            )
            for fname in os.listdir(self.rgb_pre_dir)
            if fname.endswith(
                "_pre_disaster_norm.npy"
            )
        )

    # ==========================================================
    # Dataset Length
    # ==========================================================
    def __len__(self):
        """
        Return dataset size.

        Returns
        -------
        int:
            Number of multimodal samples.
        """

        return len(self.ids)

    # ==========================================================
    # Tensor Loading Helper
    # ==========================================================
    def _load(
        self,
        path,
        modality: str,
    ):
        """
        Load normalized .npy tensor.

        Expected Input Shape
        --------------------
        Stored arrays:
            (H, W, C)

        Converted Output Shape
        ----------------------
        PyTorch tensors:
            (C, H, W)

        Why Permute?
        ------------
        NumPy image convention:
            channel-last

        PyTorch CNN convention:
            channel-first

        Args
        ----
        path:
            Path to .npy file.

        modality:
            Modality name for debugging.

        Returns
        -------
        tensor:
            Float tensor:
                (C, H, W)
        """

        # ------------------------------------------------------
        # Load NumPy array
        # ------------------------------------------------------
        x = np.load(path)

        # ------------------------------------------------------
        # Optional one-time debugging verification
        # ------------------------------------------------------
        if self.verify_shapes and not self._verified_shapes:

            print("\n===================================================")
            print("Dataset Shape Verification")
            print("===================================================")
            print(f"{modality} path  : {path}")
            print(f"{modality} shape : {x.shape}")
            print("===================================================\n")

        # ------------------------------------------------------
        # Safety validation
        # ------------------------------------------------------
        if x.ndim != 3:

            raise ValueError(
                f"{modality} array must be 3D. "
                f"Got shape: {x.shape}"
            )

        # ------------------------------------------------------
        # Convert:
        #   (H, W, C)
        #
        # Into:
        #   (C, H, W)
        #
        # Required for PyTorch CNNs.
        # ------------------------------------------------------
        x = torch.from_numpy(x).permute(
            2,
            0,
            1
        ).float()

        return x

    # ==========================================================
    # Retrieve One Sample
    # ==========================================================
    def __getitem__(self, idx):
        """
        Retrieve one multimodal disaster sample.

        Inputs Loaded
        -------------
        - RGB pre-disaster
        - RGB post-disaster
        - SAR pre-disaster
        - SAR post-disaster

        Also Returns
        ------------
        - semantic bucket
        - event metadata
        - sample identifier

        Args
        ----
        idx:
            Dataset index.

        Returns
        -------
        sample:
            Dictionary containing:
                multimodal tensors + metadata.
        """

        # ------------------------------------------------------
        # Sample ID
        # ------------------------------------------------------
        sid = self.ids[idx]

        # ------------------------------------------------------
        # Parse event name
        #
        # Example:
        #   hurricane-harvey_00001234
        #
        # -> hurricane-harvey
        # ------------------------------------------------------
        event = sid.rsplit("_", 1)[0]

        # ------------------------------------------------------
        # Semantic bucket lookup
        # ------------------------------------------------------
        if event not in self.event_to_bucket:

            raise KeyError(
                f"No bucket mapping for event: {event}"
            )

        bucket = self.event_to_bucket[event]

        # ======================================================
        # Load RGB pre-disaster tensor
        # ======================================================
        rgb_pre = self._load(
            os.path.join(
                self.rgb_pre_dir,
                f"{sid}_pre_disaster_norm.npy",
            ),
            modality="rgb_pre",
        )

        # ======================================================
        # Load RGB post-disaster tensor
        # ======================================================
        rgb_post = self._load(
            os.path.join(
                self.rgb_post_dir,
                f"{sid}_post_disaster_norm.npy",
            ),
            modality="rgb_post",
        )

        # ======================================================
        # Load SAR pre-disaster tensor
        # ======================================================
        sar_pre = self._load(
            os.path.join(
                self.sar_pre_dir,
                f"{sid}_pre_sar_norm.npy",
            ),
            modality="sar_pre",
        )

        # ======================================================
        # Load SAR post-disaster tensor
        # ======================================================
        sar_post = self._load(
            os.path.join(
                self.sar_post_dir,
                f"{sid}_post_sar_norm.npy",
            ),
            modality="sar_post",
        )

        # ------------------------------------------------------
        # Disable future shape verification prints
        # ------------------------------------------------------
        self._verified_shapes = True

        # ======================================================
        # Return Sample
        # ======================================================
        return {

            # --------------------------------------------------
            # Vision tensors
            # --------------------------------------------------
            "rgb_pre": rgb_pre,
            "sar_pre": sar_pre,
            "rgb_post": rgb_post,
            "sar_post": sar_post,

            # --------------------------------------------------
            # Semantic supervision
            # --------------------------------------------------
            "bucket": bucket,

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------
            #
            # Useful for:
            #   - debugging
            #   - retrieval auditing
            #   - leakage analysis
            #   - evaluation
            #
            # NOT used directly as labels.
            #
            # --------------------------------------------------
            "event": event,
            "sample_id": sid,
        }