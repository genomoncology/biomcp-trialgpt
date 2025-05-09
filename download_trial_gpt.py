#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "tqdm",
# ]
# ///
"""download_trialgpt_data.py

Utility script to fetch the public datasets used in the
"Matching Patients to Clinical Trials with Large Language Models" (TrialGPT)
paper and to pre‑extract the patient IDs (topic numbers) and trial NCT IDs
needed for experimentation.

The script is **functional**, uses only the standard library plus `requests`
and `tqdm`, and writes deterministic plain‑text ID lists under `./data/ids/`.

Datasets pulled
---------------
* **trial_info.json** – pre‑parsed master dump of ClinicalTrials.gov used by the authors.
* **TREC Clinical Trials 2021 corpus** – jsonl, restricted to the 26 k judged trials.
* **TREC Clinical Trials 2022 corpus** – jsonl.
* **topics2021.xml / topics2022.xml** – synthetic patient notes (75 + 50 patients).

The SIGIR‑2016 test collection can’t be fetched automatically (CSIRO portal
requires a click‑through).  If you already have it, drop the XML file or the
`topics.xml` inside `./data/sigir/` and re‑run the script – it will be parsed
automatically.

Outputs (all UTF‑8, one ID per line)
-----------------------------------
* `data/ids/trec2021_patient_ids.txt`
* `data/ids/trec2022_patient_ids.txt`
* `data/ids/trec2021_trial_ids.txt`
* `data/ids/trec2022_trial_ids.txt`
* `data/ids/all_trial_ids.txt` – union of 2021/2022 trial IDs

Run
---
```bash
python download_trialgpt_data.py  # add -q for quiet mode
```
Replace URLs with mirrors if needed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Iterable, Set

import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
ID_DIR = DATA_DIR / "ids"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ID_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    # NCBI FTP mirrors – anonymous access
    "trial_info": "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trial_info.json",
    "trec_2021": "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl",
    "trec_2022": "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2022_corpus.jsonl",
    # topic files are small XMLs hosted on trec‑cds.org
    "topics2021": "https://www.trec-cds.org/topics2021.xml",
    "topics2022": "https://www.trec-cds.org/topics2022.xml",
}

CHUNK = 1 << 16  # 64 KiB


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def stream_download(url: str, dest: Path, quiet: bool = False):
    """Download *url* to *dest* with a streaming GET (resumes if possible)."""
    if dest.exists():
        return  # skip
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0)) or None
    bar = (
        None if quiet else tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
    )
    with dest.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=CHUNK):
            fh.write(chunk)
            if bar:
                bar.update(len(chunk))
    if bar:
        bar.close()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def extract_trial_ids(jsonl: Path) -> Set[str]:
    ids: Set[str] = set()
    open_fn = gzip.open if jsonl.suffix == ".gz" else open
    with open_fn(jsonl, "rt", encoding="utf8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            nct = obj.get("nct_id") or obj.get("id") or ""
            if nct:
                ids.add(nct)
    return ids


def extract_topic_ids(topics_xml: Path) -> Set[str]:
    tree = ET.parse(topics_xml)
    root = tree.getroot()
    ids: Set[str] = set()
    for topic in root.findall(".//topic"):
        num = topic.attrib.get("number") or topic.attrib.get("id")
        if num:
            ids.add(str(num).strip())
    return ids


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress bars")
    args = p.parse_args(argv)

    quiet = args.quiet

    # 1. Download corpora
    targets = {
        "trial_info": DATA_DIR / "trial_info.json",
        "trec_2021": DATA_DIR / "trec_2021_corpus.jsonl",
        "trec_2022": DATA_DIR / "trec_2022_corpus.jsonl",
        "topics2021": DATA_DIR / "topics2021.xml",
        "topics2022": DATA_DIR / "topics2022.xml",
    }

    for key, url in URLS.items():
        try:
            stream_download(url, targets[key], quiet=quiet)
        except Exception as e:
            print(f"[warn] failed to download {url}: {e}", file=sys.stderr)

    # 2. Extract patient (topic) IDs
    for year in (2021, 2022):
        xml_path = targets[f"topics{year}"]
        if not xml_path.exists():
            print(f"[warn] missing {xml_path}; skip patient IDs for {year}")
            continue
        ids = sorted(extract_topic_ids(xml_path))
        (ID_DIR / f"trec{year}_patient_ids.txt").write_text("\n".join(ids), "utf8")
        if not quiet:
            print(f"[*] wrote {len(ids):>3} patient IDs for {year}")

    # SIGIR patients (optional)
    sigir_dir = DATA_DIR / "sigir"
    sigir_xml = next(sigir_dir.glob("*.xml"), None) if sigir_dir.exists() else None
    if sigir_xml:
        ids = sorted(extract_topic_ids(sigir_xml))
        (ID_DIR / "sigir_patient_ids.txt").write_text("\n".join(ids), "utf8")
        if not quiet:
            print(f"[*] wrote {len(ids):>3} SIGIR patient IDs")
    else:
        print(
            "[info] SIGIR topics XML not found – drop it in data/sigir/ to enable parsing."
        )

    # 3. Extract trial IDs from corpora
    trial_union: Set[str] = set()
    for year in (2021, 2022):
        corpus_path = targets[f"trec_{year}"]
        if not corpus_path.exists():
            print(f"[warn] missing {corpus_path}; skip trials for {year}")
            continue
        ids = extract_trial_ids(corpus_path)
        trial_union.update(ids)
        (ID_DIR / f"trec{year}_trial_ids.txt").write_text(
            "\n".join(sorted(ids)), "utf8"
        )
        if not quiet:
            print(f"[*] wrote {len(ids):>6} trial IDs for {year}")

    (ID_DIR / "all_trial_ids.txt").write_text("\n".join(sorted(trial_union)), "utf8")
    if not quiet:
        print(f"[*] wrote {len(trial_union):>6} unique trial IDs (union)")


if __name__ == "__main__":
    main()
