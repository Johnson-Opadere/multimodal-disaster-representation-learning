#!/usr/bin/env python3
"""
build_metadata.py
=================

Project 2A — Multimodal Dataset Manifest Builder
------------------------------------------------

Rebuilds event-level metadata manifests from normalized multimodal
image patches.

Pipeline Role
-------------
normalized_data/
    ↓
build_metadata.py
    ↓
metadata/
    ↓
dataset loading
    ↓
multimodal alignment
    ↓
contrastive training

Purpose
-------
This script constructs the canonical multimodal dataset index
used throughout Project 2A.

The generated metadata files:
    - synchronize RGB + SAR modalities
    - preserve patch relationships
    - preserve dataset splits
    - organize samples by disaster event

Conceptually, this script converts:
    disorganized modality folders
into:
    structured multimodal manifests

Output Structure
----------------
metadata/
    hurricane-harvey.json
    palu-tsunami.json
    socal-fire.json
    ...

Each JSON contains:

{
   "event_id": "...",
   "image_patches": [
       {
         "patch_id": "00000000",
         "rgb_pre":  "...",
         "rgb_post": "...",
         "sar_pre":  "...",
         "sar_post": "...",
         "split": "train"
       }
   ]
}

Input Structure
---------------
Assumes normalized multimodal patches stored as:

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

Important Design Notes
----------------------
- metadata stores relative paths only
- image arrays are NOT loaded into memory
- patch_id acts as the multimodal synchronization key
- one metadata file is created per event
- split information is preserved for evaluation integrity

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/build_metadata.py \
    --normalized_root /mnt/ebs-data/cv_project2/data/normalized_data \
    --outdir /mnt/ebs-data/cv_project2/data/metadata
"""

import os
import json
import argparse

from collections import defaultdict


# ============================================================
# Helper: Parse Filename
# ============================================================
def parse_filename(fname):
    """
    Extract:
        - event_id
        - patch_id

    from normalized patch filenames.

    Example filenames:
        hurricane-harvey_00000000_pre_disaster_norm.npy
        guatemala-volcano_00001234_post_disaster_norm.npy
        hurricane-harvey_00000000_pre_disaster_pre_sar_norm.npy

    Returns
    -------
    event_id : str
        Disaster event identifier.

    patch_id : str
        Spatial patch identifier used to synchronize:
            RGB_pre
            RGB_post
            SAR_pre
            SAR_post
    """

    parts = fname.split("_")

    event_id = parts[0]

    patch_id = parts[1]

    return event_id, patch_id


# ============================================================
# Main Metadata Builder
# ============================================================
def build_metadata(normalized_root, outdir):
    """
    Build event-level multimodal metadata manifests.

    Parameters
    ----------
    normalized_root : str
        Path to normalized multimodal patch directory.

    outdir : str
        Directory where metadata JSON files will be written.

    Behavior
    --------
    - scans normalized_data/
    - synchronizes RGB/SAR modalities
    - groups patches by event
    - preserves dataset split membership
    - writes one JSON manifest per event
    """

    os.makedirs(
        outdir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # event_id -> list of patch metadata entries
    # --------------------------------------------------------
    events = defaultdict(list)

    # --------------------------------------------------------
    # Iterate through dataset splits
    # --------------------------------------------------------
    for split in ["train", "hold", "test"]:

        split_dir = os.path.join(
            normalized_root,
            split
        )

        if not os.path.isdir(split_dir):
            continue

        # ----------------------------------------------------
        # Modality folders
        # ----------------------------------------------------
        rgb_pre_dir = os.path.join(
            split_dir,
            "rgb_pre_norm"
        )

        rgb_post_dir = os.path.join(
            split_dir,
            "rgb_post_norm"
        )

        sar_pre_dir = os.path.join(
            split_dir,
            "sar_pre_norm"
        )

        sar_post_dir = os.path.join(
            split_dir,
            "sar_post_norm"
        )

        # ----------------------------------------------------
        # Use RGB pre files as canonical patch index
        # ----------------------------------------------------
        rgb_pre_files = sorted(
            f for f in os.listdir(rgb_pre_dir)
            if f.endswith(".npy")
        )

        # ----------------------------------------------------
        # Build synchronized multimodal patch entries
        # ----------------------------------------------------
        for f in rgb_pre_files:

            event_id, patch_id = parse_filename(f)

            # ------------------------------------------------
            # Relative paths improve portability
            # ------------------------------------------------
            rgb_pre = (
                f"{split}/rgb_pre_norm/{f}"
            )

            rgb_post = (
                f"{split}/rgb_post_norm/"
                f"{event_id}_{patch_id}_post_disaster_norm.npy"
            )

            sar_pre = (
                f"{split}/sar_pre_norm/"
                f"{event_id}_{patch_id}_pre_disaster_pre_sar_norm.npy"
            )

            sar_post = (
                f"{split}/sar_post_norm/"
                f"{event_id}_{patch_id}_post_disaster_post_sar_norm.npy"
            )

            # ------------------------------------------------
            # Store synchronized multimodal tuple
            # ------------------------------------------------
            events[event_id].append({

                "patch_id": patch_id,

                "rgb_pre": rgb_pre,

                "rgb_post": rgb_post,

                "sar_pre": sar_pre,

                "sar_post": sar_post,

                "split": split
            })

    # ========================================================
    # Write Event-Level Metadata Files
    # ========================================================
    for event_id, patches in events.items():

        out = os.path.join(
            outdir,
            f"{event_id}.json"
        )

        json.dump(
            {
                "event_id": event_id,
                "image_patches": patches
            },
            open(out, "w"),
            indent=2
        )

        print(
            f" Saved {out} "
            f"({len(patches)} patches)"
        )


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--normalized_root",
        required=True,
        help="Path to normalized_data/"
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where metadata JSON files will be saved."
    )

    args = parser.parse_args()

    build_metadata(
        args.normalized_root,
        args.outdir
    )