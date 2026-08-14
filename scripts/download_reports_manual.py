#!/usr/bin/env python3
"""
download_reports_manual.py
==========================

Project 2A — Disaster Report Collection Pipeline
------------------------------------------------

Downloads disaster-related webpages, extracts readable article
content, performs lightweight cleaning, and stores the resulting
text files for downstream semantic processing.

Pipeline Role
-------------
report_sources.txt
    ↓
download_reports_manual.py
    ↓
reports/
    ↓
preprocess_text.py
    ↓
raw_text/
    ↓
filter_text.py
    ↓
text_filtered/fragments.jsonl

Purpose
-------
This script was used to build the text modality for
Project 2A's multimodal retrieval system.

The resulting reports provide:
    - disaster descriptions
    - structural damage narratives
    - flooding/wildfire/earthquake semantics
    - cross-event textual supervision

Main Steps
----------
1. Read report_sources.txt
2. Download webpages
3. Extract readable article text
4. Clean extracted text
5. Save event-organized report files

Input
-----
data/report_sources.txt

Expected format:

    hurricane-harvey:
    https://...

    palu-tsunami:
    https://...

Output
------
data/reports/
    <event>/
        01.txt
        02.txt
        ...

Behavior
--------
- downloads webpages using requests
- extracts readable article content using readability-lxml
- removes:
      URLs
      javascript snippets
      excessive whitespace
- saves cleaned text files grouped by event

Dependencies
------------
pip install requests beautifulsoup4 readability-lxml lxml

Run Command
-----------
PYTHONPATH=. python3 2A/scripts/download_reports_manual.py

Notes
-----
- Uses readability-lxml for article extraction
- Lightweight cleaning only
- Semantic filtering happens later in:
      filter_text.py
- Includes polite request throttling
"""

import os
import re
import json
import time
import requests

from bs4 import BeautifulSoup
from readability import Document
from urllib.parse import urlparse


# ============================================================
# Config
# ============================================================

SOURCE_FILE = "data/report_sources.txt"

OUTPUT_ROOT = "data/reports"

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
}


# ============================================================
# Utility: Lightweight Text Cleaning
# ============================================================
def clean_text(t):
    """
    Perform lightweight webpage text cleaning.

    Operations:
        - collapse whitespace
        - remove javascript snippets
        - remove URLs
        - normalize spacing

    Important:
        Semantic filtering is intentionally deferred
        to:
            filter_text.py
    """

    # --------------------------------------------------------
    # Collapse repeated whitespace
    # --------------------------------------------------------
    t = re.sub(r"\s+", " ", t)

    # --------------------------------------------------------
    # Remove javascript snippets
    # --------------------------------------------------------
    t = re.sub(r"(javascript:.*?;)", "", t)

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------
    t = re.sub(r"http\S+", "", t)

    # --------------------------------------------------------
    # Normalize line spacing
    # --------------------------------------------------------
    t = t.replace("\n", " ").strip()

    return t


# ============================================================
# Download + Readability Extraction
# ============================================================
def download_and_extract(url):
    """
    Download webpage and extract readable article text.

    Uses:
        readability-lxml
        BeautifulSoup

    Returns:
        cleaned readable text
        OR
        None if extraction fails
    """

    try:

        # ----------------------------------------------------
        # Download webpage
        # ----------------------------------------------------
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        if r.status_code != 200:

            print(f" Failed ({r.status_code}): {url}")

            return None

        # ----------------------------------------------------
        # Extract readable article content
        # ----------------------------------------------------
        doc = Document(r.text)

        readable_html = doc.summary()

        soup = BeautifulSoup(
            readable_html,
            "lxml"
        )

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return clean_text(text)

    except Exception as e:

        print(f" Error reading {url}: {e}")

        return None


# ============================================================
# Parse report_sources.txt
# ============================================================
def load_sources(path):
    """
    Parse report_sources.txt.

    Expected format:

        event_id:
        https://link1
        https://link2

        another_event:
        https://...

    Returns:
        dict:
            event_id -> list of URLs
    """

    events = {}

    current_event = None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Event header
            # ------------------------------------------------
            if line.endswith(":"):

                current_event = (
                    line.replace(":", "").strip()
                )

                events[current_event] = []

            # ------------------------------------------------
            # URL
            # ------------------------------------------------
            else:

                if current_event:

                    events[current_event].append(line)

    return events


# ============================================================
# Main Pipeline
# ============================================================
def main():
    """
    Download and process disaster reports.
    """

    events = load_sources(SOURCE_FILE)

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    print(
        f" Loaded {len(events)} "
        f"disaster events from {SOURCE_FILE}"
    )

    # --------------------------------------------------------
    # Process each event
    # --------------------------------------------------------
    for event_id, urls in events.items():

        event_dir = os.path.join(
            OUTPUT_ROOT,
            event_id
        )

        os.makedirs(
            event_dir,
            exist_ok=True
        )

        print(
            f"\n Event: {event_id}   "
            f"({len(urls)} URLs)"
        )

        # ----------------------------------------------------
        # Download each report
        # ----------------------------------------------------
        for idx, url in enumerate(urls, start=1):

            print(f"   → Downloading {url}")

            text = download_and_extract(url)

            if not text:

                print(
                    "      No text extracted — skipping."
                )

                continue

            # ------------------------------------------------
            # Save extracted text
            # ------------------------------------------------
            out_path = os.path.join(
                event_dir,
                f"{idx:02d}.txt"
            )

            with open(
                out_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(text)

            print(
                f"      Saved {out_path} "
                f"({len(text)} chars)"
            )

            # ------------------------------------------------
            # Polite request throttling
            # ------------------------------------------------
            time.sleep(1)

    print("\n DONE — All reports downloaded & cleaned!")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":

    main()