#!/usr/bin/env python3
"""
download_and_transcribe.py
==========================

Project 2A — Whisper Audio Collection + Transcription Pipeline
--------------------------------------------------------------

Downloads disaster-related audio sources and transcribes them
into text using OpenAI Whisper.

Pipeline Role
-------------
audio_sources.txt
    ↓
download_and_transcribe.py
    ↓
event_whisper_raw/
    ↓
event_whisper/
    ↓
raw_whisper/
    ↓
filter_whisper.py
    ↓
whisper_filtered/fragments.jsonl
    ↓
build_whisper_tokens.py

Purpose
-------
This script was used to build the Whisper modality for
Project 2A's multimodal retrieval system.

The resulting transcripts provide:
    - spoken disaster descriptions
    - human narration
    - audio-based semantic supervision
    - cross-modal retrieval grounding

Main Steps
----------
1. Read audio_sources.txt
2. Download audio using yt-dlp
3. Transcribe audio using Whisper
4. Save:
       .json transcription metadata
       .txt transcript text
5. Normalize filenames into stable format

Directory Structure
-------------------
event_whisper_raw/
    <event>/
        *.mp3

event_whisper/
    <event>/
        <event>_audio_001.json
        <event>_audio_001.txt

Example audio_sources.txt
-------------------------
hurricane-harvey:
https://youtube.com/...

palu-tsunami:
https://youtube.com/...

Dependencies
------------
pip install openai-whisper yt-dlp

FFmpeg is also required:
    sudo apt install ffmpeg

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/download_and_transcribe.py

Notes
-----
- Designed for deterministic archival collection
- Uses Whisper "medium" by default
- Safe filename normalization included
- Skips already-transcribed files
- JupyterLab-compatible subprocess output
"""

import os
import re
import subprocess
import json
from pathlib import Path


# =====================================================
# Utility Helpers
# =====================================================

def safe_name(name):
    """
    Normalize filenames to avoid:
        - spaces
        - shell-breaking characters
        - filesystem issues

    Example:
        "CNN Report (Part 1)"
            →
        "CNN_Report_Part_1"
    """

    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

    return name.strip("_")


def run_cmd(cmd):
    """
    Execute subprocess command and display output.

    Designed for:
        - JupyterLab
        - terminal execution
        - debugging download issues
    """

    print("▶", " ".join(cmd))

    try:

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )

        print(result.stdout)

        return result.returncode == 0

    except Exception as e:

        print(" ERROR running command:", e)

        return False


# =====================================================
# Step 1: Load Audio Sources
# =====================================================

def load_audio_sources(file_path):
    """
    Parse audio_sources.txt.

    Expected format:

        hurricane-harvey:
        https://youtube.com/...

        palu-tsunami:
        https://youtube.com/...

    Returns:
        dict:
            event_name -> list of URLs
    """

    events = {}

    current_event = None

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # -------------------------------------------------
            # Event name
            # -------------------------------------------------
            if line.endswith(":"):

                current_event = line.replace(":", "").strip()

                events[current_event] = []

                continue

            # -------------------------------------------------
            # URL
            # -------------------------------------------------
            if current_event:

                events[current_event].append(line)

    return events


# =====================================================
# Step 2: Download Audio
# =====================================================

def download_event_audio(event, urls, raw_root):
    """
    Download audio for a single event using yt-dlp.

    Output:
        event_whisper_raw/<event>/*.mp3
    """

    event_dir = Path(raw_root) / event

    event_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Downloading audio for: {event} =====")

    for url in urls:

        print(f"\n→ URL: {url}")

        out_template = str(
            event_dir / "%(title)s.%(ext)s"
        )

        cmd = [
            "yt-dlp",
            "-x",
            "--no-playlist",
            "--audio-format", "mp3",
            "-o", out_template,
            url
        ]

        ok = run_cmd(cmd)

        if ok:
            print("✓ Downloaded successfully")
        else:
            print(" Failed to download")


# =====================================================
# Step 3: Whisper Transcription
# =====================================================

def transcribe_event_audio(
    event,
    raw_root,
    out_root,
    whisper_model="medium"
):
    """
    Transcribe downloaded audio using OpenAI Whisper.

    Saves:
        .json metadata
        .txt transcript
    """

    print(f"\n===== Transcribing event: {event} =====")

    try:
        import whisper

    except ImportError:

        print(
            " whisper library not installed.\n"
            "Run: pip install openai-whisper"
        )

        return

    # -------------------------------------------------
    # Load Whisper model
    # -------------------------------------------------
    model = whisper.load_model(whisper_model)

    in_dir = Path(raw_root) / event

    out_dir = Path(out_root) / event

    out_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in sorted(in_dir.glob("*.mp3")):

        base = audio_file.stem

        clean_name = safe_name(base)

        json_out = out_dir / f"{clean_name}.json"

        txt_out = out_dir / f"{clean_name}.txt"

        # -------------------------------------------------
        # Skip existing transcripts
        # -------------------------------------------------
        if json_out.exists():

            print(
                f" Skipping "
                f"(already transcribed): {audio_file.name}"
            )

            continue

        print(f" Transcribing: {audio_file.name}")

        result = model.transcribe(str(audio_file))

        # -------------------------------------------------
        # Save JSON metadata
        # -------------------------------------------------
        with open(json_out, "w", encoding="utf-8") as f:

            json.dump(result, f, indent=2)

        # -------------------------------------------------
        # Save transcript text
        # -------------------------------------------------
        with open(txt_out, "w", encoding="utf-8") as f:

            f.write(
                result.get("text", "").strip()
            )

        print(f" Saved: {json_out.name}")


# =====================================================
# Step 4: Normalize Filenames
# =====================================================

def normalize_event_filenames(event, out_root):
    """
    Normalize Whisper outputs into stable naming.

    Example:
        hurricane-harvey_audio_001.json
        hurricane-harvey_audio_001.txt

    Important:
        Stable naming improves:
            - reproducibility
            - downstream processing
            - dataset consistency
    """

    event_dir = Path(out_root) / event

    items = sorted(event_dir.glob("*.json"))

    print(
        f"\n Normalizing filenames for "
        f"{event}: {len(items)} files"
    )

    for idx, json_file in enumerate(items, start=1):

        stem = f"{event}_audio_{idx:03d}"

        json_new = json_file.with_name(
            stem + ".json"
        )

        txt_old = json_file.with_suffix(".txt")

        txt_new = json_file.with_name(
            stem + ".txt"
        )

        # -------------------------------------------------
        # Rename JSON
        # -------------------------------------------------
        json_file.rename(json_new)

        # -------------------------------------------------
        # Rename TXT
        # -------------------------------------------------
        if txt_old.exists():

            txt_old.rename(txt_new)

        print(
            f"   ✓ {json_file.name} "
            f"→ {json_new.name}"
        )


# =====================================================
# Main Entry
# =====================================================

def main(
    audio_sources="audio_sources.txt",
    raw_root="event_whisper_raw",
    out_root="event_whisper",
    whisper_model="medium"
):
    """
    Full audio collection + transcription pipeline.
    """

    print(
        f"📄 Loading audio source list "
        f"→ {audio_sources}"
    )

    events = load_audio_sources(audio_sources)

    print(
        f" Found {len(events)} events: "
        f"{list(events.keys())}"
    )

    for event, urls in events.items():

        # -------------------------------------------------
        # Download audio
        # -------------------------------------------------
        download_event_audio(
            event,
            urls,
            raw_root
        )

        # -------------------------------------------------
        # Whisper transcription
        # -------------------------------------------------
        transcribe_event_audio(
            event,
            raw_root,
            out_root,
            whisper_model
        )

        # -------------------------------------------------
        # Normalize filenames
        # -------------------------------------------------
        normalize_event_filenames(
            event,
            out_root
        )

    print(
        "\n ALL DONE — "
        "Audio downloaded, transcribed, normalized!\n"
    )


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    main()