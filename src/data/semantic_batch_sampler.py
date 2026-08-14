#!/usr/bin/env python3
"""
semantic_batch_sampler.py
=========================

Project 2A_v2 — Semantic Batch Sampler
--------------------------------------

Defines a diversity-aware PyTorch batch sampler designed for:

    multimodal contrastive representation learning.

This sampler constructs:
    semantically diverse training batches

to improve:
    contrastive retrieval geometry.

Core Purpose
------------
Contrastive learning quality depends heavily on:

    - semantic diversity
    - hard negatives
    - cross-bucket competition
    - retrieval pressure

Random batching often produces:
    semantically homogeneous batches

which weakens:
    - negative diversity
    - contrastive competition
    - embedding separation

This sampler instead encourages:
    semantically mixed batches.

Example
-------

Bad random batch:
    [flooding, flooding, flooding]

Better semantic batch:
    [flooding,
     wildfire,
     tsunami,
     structural_damage]

This creates:
    stronger contrastive supervision.

Why This Matters
----------------
Contrastive learning is fundamentally:
    geometry learning.

Embedding quality depends heavily on:
    which negatives compete during training.

Semantically diverse batches improve:
    - semantic separation
    - retrieval structure
    - embedding robustness
    - cross-bucket discrimination

Key Design Philosophy
---------------------

OLD STRATEGY
------------
Strict one-sample-per-bucket batching.

Problem:
    failed under severe dataset imbalance.

Example:
    volcanic_damage may contain very few samples.

NEW STRATEGY
------------
Probabilistic diversity-aware batching.

Benefits:
    - prefers diversity
    - allows bucket reuse
    - handles imbalance gracefully
    - avoids empty DataLoader failures

Sampling Strategy
-----------------

1. Build:
       bucket → dataset indices

2. Compute rare-bucket weights:
       weight = 1 / (count ^ alpha)

3. Prefer unseen buckets within batch

4. Reuse buckets only when necessary

5. Reshuffle exhausted buckets dynamically

Rare Bucket Balancing
---------------------
Rare semantic categories receive:
    higher sampling probability.

This improves:
    representation quality for minority buckets.

Scientific Importance
---------------------
This sampler is one of the highest-ROI
components in the retrieval pipeline because:

contrastive learning quality often depends MORE on:
    - batch semantics
    - negative diversity
    - retrieval competition

than on:
    - larger backbones
    - deeper networks
    - additional losses

Future Possible Extensions
--------------------------
- hard-negative mining
- cross-event retrieval sampling
- curriculum batching
- semantic hardness scheduling
- adaptive temperature-aware sampling

Requirements
------------
Dataset items MUST contain:

    dataset[idx]["bucket"]

already provided by:
    MultimodalDamageDataset

Run Command
-----------
This file is imported during training
and is not executed directly.

Example usage:

from torch.utils.data import DataLoader

from src.data.semantic_batch_sampler import (
    SemanticBatchSampler
)

sampler = SemanticBatchSampler(
    dataset,
    batch_size=8,
)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
)

Dependencies
------------
pip install torch
"""

# ==============================================================
# Imports
# ==============================================================

import math
import random

from collections import defaultdict

from torch.utils.data import Sampler


class SemanticBatchSampler(Sampler):
    """
    Diversity-Aware Semantic Batch Sampler
    -------------------------------------

    Constructs semantically diverse batches for:
        contrastive representation learning.

    Primary Goals
    -------------
    - improve semantic diversity
    - strengthen contrastive negatives
    - reduce homogeneous batches
    - stabilize learning under imbalance

    Why This Exists
    ---------------
    Random batching often creates:
        semantically weak batches.

    Example:
        [flooding, flooding]

    Such batches reduce:
        - negative competition
        - embedding separation
        - retrieval quality

    This sampler instead encourages:
        semantically mixed batches.

    Example:
        [flooding,
         wildfire,
         tsunami,
         structural_damage]

    This improves:
        contrastive embedding geometry.

    Important Design Philosophy
    ---------------------------
    Diversity is preferred,
    but not strictly enforced.

    This avoids:
        failure under severe class imbalance.

    Rare Bucket Handling
    --------------------
    Rare buckets receive:
        probabilistic upweighting.

    This improves:
        representation quality
        for minority semantic classes.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
        rare_bucket_alpha: float = 0.15,
    ):
        """
        Initialize semantic batch sampler.

        Args
        ----
        dataset:
            MultimodalDamageDataset.

        batch_size:
            Desired batch size.

        drop_last:
            Whether incomplete final batches
            should be discarded.

        seed:
            Random seed for reproducibility.

        rare_bucket_alpha:
            Controls rare-bucket upweighting.

            Formula:
                weight = 1 / (count ^ alpha)

            Interpretation:
                0.0:
                    no balancing

                1.0:
                    strong balancing
        """

        self.dataset = dataset

        self.batch_size = batch_size

        self.drop_last = drop_last

        self.seed = seed

        self.rare_bucket_alpha = rare_bucket_alpha

        # ======================================================
        # Build:
        #   bucket → dataset indices
        # ======================================================
        #
        # Example:
        #
        # flooding:
        #   [0, 5, 11, 20]
        #
        # wildfire:
        #   [1, 7, 19]
        #
        # ======================================================
        self.bucket_to_indices = defaultdict(list)

        print("\n===================================================")
        print("Building Semantic Batch Sampler")
        print("===================================================\n")

        # ------------------------------------------------------
        # Iterate through dataset
        # ------------------------------------------------------
        for idx in range(len(dataset)):

            item = dataset[idx]

            bucket = item["bucket"]

            self.bucket_to_indices[bucket].append(idx)

        # ======================================================
        # Bucket names
        # ======================================================
        self.buckets = sorted(
            self.bucket_to_indices.keys()
        )

        # ======================================================
        # Logging
        # ======================================================
        print("Buckets discovered:")
        print("---------------------------------------------------")

        for bucket in self.buckets:

            count = len(
                self.bucket_to_indices[bucket]
            )

            print(f"{bucket:<25} {count}")

        print("\n===================================================\n")

        # ======================================================
        # Rare-Bucket Sampling Weights
        # ======================================================
        #
        # Rare buckets receive:
        #   higher probability.
        #
        # Formula:
        #
        #   weight = 1 / (count ^ alpha)
        #
        # ======================================================
        raw_weights = []

        for bucket in self.buckets:

            count = len(
                self.bucket_to_indices[bucket]
            )

            weight = 1.0 / (
                math.pow(
                    count,
                    rare_bucket_alpha
                )
            )

            raw_weights.append(weight)

        # ------------------------------------------------------
        # Normalize weights
        # ------------------------------------------------------
        total = sum(raw_weights)

        self.bucket_weights = [

            w / total

            for w in raw_weights
        ]

        # ======================================================
        # Epoch sample estimate
        # ======================================================
        self.num_samples = len(dataset)

    # ==========================================================
    # Batch Iterator
    # ==========================================================
    def __iter__(self):
        """
        Generate semantic training batches.

        Strategy
        --------
        1. Prefer unseen buckets within batch
        2. Use weighted probabilistic sampling
        3. Allow reuse when necessary
        4. Reshuffle exhausted buckets

        Returns
        -------
        batch:
            List of dataset indices.
        """

        # ------------------------------------------------------
        # Deterministic RNG
        # ------------------------------------------------------
        rng = random.Random(self.seed)

        # ======================================================
        # Shuffle bucket pools
        # ======================================================
        #
        # Each bucket maintains:
        #   shuffled dataset indices.
        #
        # ======================================================
        bucket_pools = {}

        for bucket, indices in self.bucket_to_indices.items():

            shuffled = indices.copy()

            rng.shuffle(shuffled)

            bucket_pools[bucket] = shuffled

        # ======================================================
        # Per-bucket pointers
        # ======================================================
        bucket_ptrs = {

            bucket: 0

            for bucket in self.buckets
        }

        # ======================================================
        # Number of batches
        # ======================================================
        num_batches = len(self)

        # ======================================================
        # Batch loop
        # ======================================================
        for _ in range(num_batches):

            batch = []

            # --------------------------------------------------
            # Tracks buckets already used in current batch
            # --------------------------------------------------
            used_buckets = set()

            # ==================================================
            # Fill one batch
            # ==================================================
            while len(batch) < self.batch_size:

                # --------------------------------------------------
                # Prefer unseen buckets first
                # --------------------------------------------------
                candidate_buckets = [

                    b for b in self.buckets

                    if b not in used_buckets
                ]

                # --------------------------------------------------
                # If all buckets already used:
                # allow reuse.
                # --------------------------------------------------
                if len(candidate_buckets) == 0:

                    candidate_buckets = self.buckets

                # ==================================================
                # Candidate weights
                # ==================================================
                candidate_weights = []

                for bucket in candidate_buckets:

                    idx = self.buckets.index(bucket)

                    candidate_weights.append(
                        self.bucket_weights[idx]
                    )

                # --------------------------------------------------
                # Normalize candidate weights
                # --------------------------------------------------
                total = sum(candidate_weights)

                candidate_weights = [

                    w / total

                    for w in candidate_weights
                ]

                # ==================================================
                # Sample semantic bucket
                # ==================================================
                bucket = rng.choices(
                    candidate_buckets,
                    weights=candidate_weights,
                    k=1,
                )[0]

                used_buckets.add(bucket)

                # ==================================================
                # Bucket pool + pointer
                # ==================================================
                pool = bucket_pools[bucket]

                ptr = bucket_ptrs[bucket]

                # --------------------------------------------------
                # Reshuffle exhausted buckets
                # --------------------------------------------------
                if ptr >= len(pool):

                    rng.shuffle(pool)

                    bucket_ptrs[bucket] = 0

                    ptr = 0

                # ==================================================
                # Add dataset sample
                # ==================================================
                batch.append(pool[ptr])

                bucket_ptrs[bucket] += 1

            # ======================================================
            # Yield final batch
            # ======================================================
            yield batch

    # ==========================================================
    # Number of Batches
    # ==========================================================
    def __len__(self):
        """
        Return number of batches per epoch.

        Returns
        -------
        int:
            Total batches generated per epoch.
        """

        # ------------------------------------------------------
        # Drop incomplete batch
        # ------------------------------------------------------
        if self.drop_last:

            return (
                self.num_samples
                // self.batch_size
            )

        # ------------------------------------------------------
        # Keep incomplete final batch
        # ------------------------------------------------------
        return math.ceil(
            self.num_samples / self.batch_size
        )