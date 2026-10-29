"""Download the public datasets used by the project (stdlib only).

Rules followed here:
  * never overwrite an existing file (skip it and keep the old copy)
  * raw files stay untouched; processed files go to data/processed/
  * every download is recorded in data/provenance.json (url, sha256, size, date, license)

Run:  python scripts/download_data.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROV = DATA / "provenance.json"

UCI_URL = ("https://archive.ics.uci.edu/static/public/498/"
           "incident+management+process+enriched+event+log.zip")
TICKETS_URL = ("https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/"
               "resolve/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
LOGHUB_RAW = "https://raw.githubusercontent.com/logpai/loghub/master"
# Small 2k-line samples (with ground-truth templates) instead of the multi-GB Zenodo files.
LOGHUB_SYSTEMS = ["Apache", "Hadoop", "HDFS", "OpenStack", "Proxifier"]

DATASETS = {
    "servicenow": {
        "name": "Incident management process enriched event log (UCI id 498)",
        "page": "https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log",
        "license": "CC BY 4.0",
        "use": "Manual-process MTTR baseline (resolution-time distribution). No free text.",
    },
    "tickets": {
        "name": "Tobi-Bueck/customer-support-tickets (Hugging Face)",
        "page": "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets",
        "license": "CC BY-NC 4.0",
        "use": "Queue and priority triage models; incident-index resolution hints.",
    },
    "loghub": {
        "name": "Loghub 2k samples (Apache, Hadoop, HDFS, OpenStack, Proxifier)",
        "page": "https://github.com/logpai/loghub",
        "license": "Loghub research use; cite Zhu et al., ISSRE 2023 / He et al. 2020",
        "use": "Drain log-parsing check and the log retrieval index.",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> dict:
    """Download url -> dest unless dest exists. Returns a provenance record."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    status = "existing"
    if not dest.exists():
        print(f"  downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "incedentagent/0.1"})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        status = "downloaded"
    else:
        print(f"  exists, skipped: {dest.relative_to(ROOT)}")
    return {"url": url, "file": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "bytes": dest.stat().st_size, "sha256": sha256(dest), "status": status}


def main() -> int:
    prov = json.loads(PROV.read_text()) if PROV.exists() else {}
    today = dt.date.today().isoformat()

    print("[1/3] ServiceNow incident event log (UCI 498)")
    rec = fetch(UCI_URL, DATA / "servicenow" / "uci_498_incident_event_log.zip")
    with zipfile.ZipFile(DATA / "servicenow" / "uci_498_incident_event_log.zip") as z:
        for member in z.namelist():
            out = DATA / "servicenow" / member
            if not out.exists():
                z.extract(member, DATA / "servicenow")
        rec["extracted"] = z.namelist()
    prov.setdefault("servicenow", {**DATASETS["servicenow"], "files": [], "download_date": today})
    prov["servicenow"]["files"] = [rec]

    print("[2/3] Customer support tickets")
    rec = fetch(TICKETS_URL, DATA / "tickets" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
    prov.setdefault("tickets", {**DATASETS["tickets"], "files": [], "download_date": today})
    prov["tickets"]["files"] = [rec]

    print("[3/3] Loghub 2k samples")
    files = []
    for system in LOGHUB_SYSTEMS:
        for suffix in ["_2k.log", "_2k.log_structured.csv", "_2k.log_templates.csv"]:
            name = f"{system}{suffix}"
            try:
                files.append(fetch(f"{LOGHUB_RAW}/{system}/{name}", DATA / "loghub" / system / name))
            except Exception as e:  # keep going; report what is missing
                print(f"  WARNING could not fetch {name}: {e}")
    prov.setdefault("loghub", {**DATASETS["loghub"], "files": [], "download_date": today})
    prov["loghub"]["files"] = files

    PROV.write_text(json.dumps(prov, indent=2))
    print(f"provenance written to {PROV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
