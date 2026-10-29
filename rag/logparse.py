"""Log parsing with Drain (drain3) and chunking into citable log windows.

Loghub lines are parsed into templates; chunks are built around ERROR/WARN-like
lines (with a few lines of context) plus fixed time-ordered windows. Each chunk
keeps metadata: system/service, first timestamp, severity and template ids.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

LOGHUB = Path(__file__).resolve().parents[1] / "data" / "loghub"
SYSTEMS = ["Apache", "Hadoop", "HDFS", "OpenStack", "Proxifier"]
_SEV = re.compile(r"\b(FATAL|CRIT|CRITICAL|ERROR|error|WARN|WARNING|warn|INFO|DEBUG|notice)\b")
_SEV_RANK = {"fatal": 4, "crit": 4, "critical": 4, "error": 3, "warn": 2, "warning": 2,
             "notice": 1, "info": 1, "debug": 0}


def new_miner(mask: bool = True) -> TemplateMiner:
    """Drain miner. Masking only affects template ids; chunk text keeps raw identifiers,
    so error codes / hostnames stay keyword-searchable either way."""
    from drain3.masking import MaskingInstruction
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = 0.4
    cfg.drain_depth = 4
    # generic masks (same for every system, not tuned per dataset)
    cfg.masking_instructions = [
        MaskingInstruction(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "IP"),
        MaskingInstruction(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-f]{8}-[0-9a-f-]{27}\b", "HEX"),
        MaskingInstruction(r"(?<![A-Za-z])-?\d+(?:\.\d+)?(?![A-Za-z])", "NUM"),
    ] if mask else []
    return TemplateMiner(config=cfg)


def severity(line: str) -> str:
    m = _SEV.search(line)
    return m.group(1).lower().replace("critical", "crit").replace("warning", "warn") if m else "info"


def load_structured(system: str) -> pd.DataFrame:
    """Loghub ground-truth parse (Content + EventId) for one 2k sample."""
    return pd.read_csv(LOGHUB / system / f"{system}_2k.log_structured.csv")


def parse_contents(contents: list[str], mask: bool = True) -> list[int]:
    miner = new_miner(mask)
    return [miner.add_log_message(c)["cluster_id"] for c in contents]


def grouping_accuracy(true_ids: list[str], pred_ids: list[int]) -> float:
    """Standard log-parsing Grouping Accuracy (Zhu et al., ICSE-SEIP 2019).

    A line counts as correct only if its predicted group contains exactly the same
    set of lines as its ground-truth group.
    """
    df = pd.DataFrame({"t": true_ids, "p": pred_ids})
    t_groups = df.groupby("t").groups
    p_groups = df.groupby("p").groups
    correct = 0
    for _, idx in p_groups.items():
        t_vals = df.loc[idx, "t"].unique()
        if len(t_vals) == 1 and len(t_groups[t_vals[0]]) == len(idx):
            correct += len(idx)
    return correct / len(df)


def loghub_chunks(window: int = 10, context: int = 2) -> list[dict]:
    """Citable log chunks from the Loghub samples."""
    chunks = []
    for system in SYSTEMS:
        df = load_structured(system)
        raw = (LOGHUB / system / f"{system}_2k.log").read_text(encoding="utf8", errors="replace").splitlines()
        tmpl = parse_contents(df["Content"].astype(str).tolist())
        n = min(len(raw), len(df))
        sev = [severity(raw[i]) for i in range(n)]
        spans = [(s, min(s + window, n)) for s in range(0, n, window)]
        # extra chunks centred on error-like lines
        spans += [(max(0, i - context), min(n, i + context + 1))
                  for i in range(n) if _SEV_RANK.get(sev[i], 1) >= 3][:150]
        seen = set()
        for a, b in spans:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            top_sev = max(sev[a:b], key=lambda s: _SEV_RANK.get(s, 1))
            chunks.append({
                "id": f"LOG:{system.lower()}:{a:04d}-{b:04d}",
                "text": "\n".join(raw[a:b]),
                "meta": {"kind": "loghub", "service": system, "severity": top_sev,
                         "line_start": a, "line_end": b,
                         "template_ids": sorted({int(t) for t in tmpl[a:b]})},
            })
    return chunks
