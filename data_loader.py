"""Load and prepare the real datasets.

prepare_tickets():   English tickets -> data/processed/tickets_{train,val,test}.csv
                     (70/15/15, stratified by queue, seed 42; never overwrites)
servicenow_resolution_hours(): per-incident resolution time (hours) from UCI 498.

Run:  python data_loader.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PROC = DATA / "processed"
SEED = 42
IT_QUEUES = ["Technical Support", "IT Support", "Service Outages and Maintenance"]
PRIORITIES = ["low", "medium", "high"]
TICKETS_RAW = DATA / "tickets" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
SERVICENOW_RAW = DATA / "servicenow" / "incident_event_log.csv"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def prepare_tickets(force: bool = False) -> dict[str, pd.DataFrame]:
    out = {s: PROC / f"tickets_{s}.csv" for s in ("train", "val", "test")}
    if all(p.exists() for p in out.values()) and not force:
        return {s: pd.read_csv(p) for s, p in out.items()}
    df = pd.read_csv(TICKETS_RAW)
    df = df[df["language"] == "en"].copy()
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = (df["subject"] + " " + df["body"]).str.strip()
    df = df[df["text"].str.len() > 0]
    df["_key"] = df["text"].map(_norm)
    before = len(df)
    df = df.drop_duplicates("_key").drop(columns="_key")
    print(f"tickets: {before} English rows, {before - len(df)} exact duplicates removed -> {len(df)}")
    df = df[df["priority"].isin(PRIORITIES)].reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    keep = ["ticket_id", "text", "subject", "body", "answer", "type", "queue", "priority"]
    df = df[keep]
    train, rest = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["queue"])
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=rest["queue"])
    PROC.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    for s, d in splits.items():
        d.to_csv(out[s], index=False)
    print({s: len(d) for s, d in splits.items()})
    return splits


def load_sim_incidents() -> pd.DataFrame:
    import json
    df = pd.read_csv(PROC / "sim_incidents.csv")
    df["logs"] = df["logs"].map(json.loads)
    df["correct_fix"] = df["correct_fix"].fillna("")
    return df


def servicenow_resolution_hours() -> pd.Series:
    """Resolution time in hours per incident (opened_at -> last resolved_at).

    The event log has many rows per incident (one per state change); we take the
    first opened_at and the last valid resolved_at of each incident number.
    """
    cache = PROC / "servicenow_resolution_hours.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0)["hours"]
    df = pd.read_csv(SERVICENOW_RAW, usecols=["number", "opened_at", "resolved_at", "sys_updated_at"],
                     na_values=["?"])
    fmt = "%d/%m/%Y %H:%M"
    df["opened_at"] = pd.to_datetime(df["opened_at"], format=fmt, errors="coerce")
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], format=fmt, errors="coerce")
    df["sys_updated_at"] = pd.to_datetime(df["sys_updated_at"], format=fmt, errors="coerce")
    df = df.sort_values(["number", "sys_updated_at"])
    g = df.groupby("number")
    opened = g["opened_at"].min()
    resolved = g["resolved_at"].last()   # last non-null value in update order
    hours = ((resolved - opened).dt.total_seconds() / 3600.0).dropna()
    hours = hours[hours >= 0]
    PROC.mkdir(parents=True, exist_ok=True)
    hours.rename("hours").to_csv(cache)
    return hours


def describe_hours(h: pd.Series) -> dict:
    return {"n_incidents_with_resolution": int(len(h)), "median_h": float(np.median(h)),
            "mean_h": float(h.mean()), "p25_h": float(h.quantile(.25)),
            "p75_h": float(h.quantile(.75)), "p90_h": float(h.quantile(.90)),
            "max_h": float(h.max())}


def main() -> int:
    prepare_tickets()
    n_inc = pd.read_csv(SERVICENOW_RAW, usecols=["number"])["number"].nunique()
    print(f"servicenow: {n_inc} distinct incidents")
    print("servicenow resolution hours:", describe_hours(servicenow_resolution_hours()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
