#!/usr/bin/env python3
"""
merge_whisper_transcripts.py
============================

Project 2A — Whisper Transcript Consolidation Pipeline
------------------------------------------------------

Merges all Whisper transcript chunks for each disaster event
into one consolidated event-level transcript.

Pipeline Role
-------------
event_whisper/
    ↓
merge_whisper_transcripts.py
    ↓
event_whisper_merged/
    ↓
raw_whisper/
    ↓
filter_whisper.py
    ↓
whisper_filtered/fragments.jsonl

Purpose
-------
During Whisper transcription, each downloaded audio/video source
produces an individual transcript chunk:

    hurricane-harvey_audio_001.txt
    hurricane-harvey_audio_002.txt
    ...

This script merges all transcript chunks for an event into
a single consolidated transcript.

This simplifies:
    - downstream filtering
    - semantic extraction
    - tokenization
    - multimodal supervision

Input
-----
event_whisper/
    <event>/
        *.txt

Output
------
event_whisper_merged/
    <event>.txt

Behavior
--------
- loads all transcript chunks
- preserves transcript ordering
- removes empty transcripts
- joins transcripts with paragraph spacing
- creates one event-level transcript

Example
-------
Input:
    event_whisper/
        hurricane-harvey/
            hurricane-harvey_audio_001.txt
            hurricane-harvey_audio_002.txt

Output:
    event_whisper_merged/
        hurricane-harvey.txt

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/merge_whisper_transcripts.py

Custom Paths
------------
PYTHONPATH=. python3 2A/scripts/merge_whisper_transcripts.py \
    --whisper_root event_whisper \
    --out_root event_whisper_merged
"""

import os
from pathlib import Path
import argparse


# ============================================================
# Merge Single Event
# ============================================================
def merge_event(event_dir, out_dir):
    """
    Merge all transcript chunks for one event.

    Example:
        hurricane-harvey_audio_001.txt
        hurricane-harvey_audio_002.txt
            ↓
        hurricane-harvey.txt
    """

    event = event_dir.name

    txt_files = sorted(
        event_dir.glob("*.txt")
    )

    # --------------------------------------------------------
    # No transcript files
    # --------------------------------------------------------
    if not txt_files:

        print(f"⚠️ No transcripts found for {event}")

        return

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out_file = out_dir / f"{event}.txt"

    print(f"\n=== Merging transcripts for: {event} ===")

    print(f"  Found {len(txt_files)} transcript chunks")

    merged = []

    for f in txt_files:

        text = f.read_text(
            encoding="utf-8"
        ).strip()

        # ----------------------------------------------------
        # Skip empty transcript chunks
        # ----------------------------------------------------
        if text:
            merged.append(text)

    # --------------------------------------------------------
    # Preserve transcript chunk boundaries
    # --------------------------------------------------------
    final_text = "\n\n".join(merged)

    # --------------------------------------------------------
    # Save merged transcript
    # --------------------------------------------------------
    out_file.write_text(
        final_text,
        encoding="utf-8"
    )

    print(f"  ✓ Saved merged file → {out_file}")

    print(
        f"  📏 Length: "
        f"{len(final_text):,} characters\n"
    )


# ============================================================
# Main Pipeline
# ============================================================
def main(
    whisper_root="event_whisper",
    out_root="event_whisper_merged"
):
    """
    Merge transcripts for all events.
    """

    whisper_root = Path(whisper_root)

    out_root = Path(out_root)

    # --------------------------------------------------------
    # Validate input folder
    # --------------------------------------------------------
    if not whisper_root.exists():

        print(
            f"❌ ERROR: "
            f"No folder found: {whisper_root}"
        )

        return

    # --------------------------------------------------------
    # Discover events
    # --------------------------------------------------------
    events = [
        d for d in whisper_root.iterdir()
        if d.is_dir()
    ]

    print(f"🎧 Found {len(events)} events to merge\n")

    # --------------------------------------------------------
    # Merge transcripts per event
    # --------------------------------------------------------
    for event_dir in events:

        merge_event(
            event_dir,
            out_root
        )

    print("🎉 DONE — All Whisper transcripts merged!")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--whisper_root",
        default="event_whisper",
        help="Directory containing per-event Whisper transcripts."
    )

    parser.add_argument(
        "--out_root",
        default="event_whisper_merged",
        help="Directory to save merged event transcripts."
    )

    args = parser.parse_args()

    main(
        args.whisper_root,
        args.out_root
    )