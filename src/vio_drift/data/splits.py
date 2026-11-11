"""
Read the leave-one-sequence-out folds in ``data/processed/splits.json``.

Folds are re-checked on load: a fold whose train and test sequences overlap
raises instead of silently leaking overlapping windows into the test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fold:
    """One outer cross-validation fold: train on some sequences, test on one."""
    fold_id: str
    train: tuple[str, ...]
    test: tuple[str, ...]


def load_splits(splits_path: Path) -> list[Fold]:
    """Read leave-one-sequence-out folds from ``splits.json`` and validate them."""
    data = json.loads(Path(splits_path).read_text())
    folds = [
        Fold(fold_id=f["fold_id"], train=tuple(f["train"]), test=tuple(f["test"]))
        for f in data["folds"]
    ]
    for fold in folds:
        overlap = set(fold.train) & set(fold.test)
        if overlap:
            raise ValueError(f"{fold.fold_id}: {sorted(overlap)} in both train and test")
    return folds


def make_dl_splits(windows, outer_folds, validation_fraction=0.2,
                   gap_seconds=4.0, seed=42):
    """Final ceil(fraction*N) windows validate; purge earlier windows until
    max(train.t_end_ns) + gap <= min(validation.t_start_ns).

    Uses actual half-open Phase 1 intervals, not window IDs or stride counts.
    Entries retain original IDs and timestamps; excluded windows are explicit.
    Seed is recorded for downstream experiments; this split uses no randomness.
    """
    import math
    import pandas as pd

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not math.isfinite(gap_seconds) or gap_seconds < 0:
        raise ValueError("gap_seconds must be finite and nonnegative")
    gap_ns = round(gap_seconds * 1_000_000_000)
    columns = ["seq_id", "window_id", "t_start_ns", "t_end_ns"]
    tables = {}
    for seq, frame in windows.items():
        frame = frame[columns].copy()
        if frame.empty or frame.isna().any().any() or set(frame.seq_id) != {seq}:
            raise ValueError(f"{seq}: empty or invalid window metadata")
        for col in columns[1:]:
            if not pd.api.types.is_integer_dtype(frame[col]):
                raise ValueError(f"{seq}: {col} must contain integers")
        if frame.window_id.duplicated().any() or (frame.t_end_ns <= frame.t_start_ns).any():
            raise ValueError(f"{seq}: duplicate IDs or invalid intervals")
        if frame.t_start_ns.duplicated().any():
            raise ValueError(f"{seq}: duplicate start timestamps")
        tables[seq] = frame.sort_values(["t_start_ns", "window_id"])

    folds = []
    for outer in outer_folds:
        if len(outer.train) != 1 or len(outer.test) != 1 or set(outer.train) & set(outer.test):
            raise ValueError("DL stage requires one distinct training and test flight")
        source = tables[outer.train[0]]
        n_val = math.ceil(len(source) * validation_fraction)
        validation = source.iloc[-n_val:]
        candidates = source.iloc[:-n_val]
        cutoff = int(validation.t_start_ns.min()) - gap_ns
        train = candidates[candidates.t_end_ns <= cutoff]
        excluded = candidates[candidates.t_end_ns > cutoff]
        if train.empty:
            raise ValueError(f"{outer.fold_id}: no training windows remain after purge")
        folds.append({
            "fold_id": outer.fold_id,
            "training_source": outer.train[0], "test_source": outer.test[0],
            "train": train.to_dict("records"),
            "validation": validation.to_dict("records"),
            "excluded_gap": excluded.to_dict("records"),
            "test": tables[outer.test[0]].to_dict("records"),
        })
    return {
        "schema_version": 1, "scheme": "leave_one_sequence_out",
        "sequences": list(windows), "seed": seed, "primary_target": "err_mag_m",
        "validation_fraction": validation_fraction, "validation_rounding": "ceil",
        "gap_ns": gap_ns,
        "gap_rule": "max(train.t_end_ns) + gap_ns <= min(validation.t_start_ns)",
        "selection_rule": "Use validation only for early stopping and hyperparameters; test only for final evaluation.",
        "folds": folds,
    }


def validate_dl_fold(fold, windows, gap_ns=4_000_000_000):
    """Validate IDs, exact integer intervals, coverage and temporal separation."""
    sequences = {"MH_01_easy", "MH_02_easy"}
    if {fold["training_source"], fold["test_source"]} != sequences:
        raise ValueError("DL folds must use the two distinct easy flights")
    seen = set()
    for part in ("train", "validation", "test", "excluded_gap", "excluded_subset"):
        refs = fold.get(part, [])
        if part in ("train", "validation", "test") and not refs:
            raise ValueError(f"Empty {part} partition")
        expected_seq = fold["test_source"] if part == "test" else fold["training_source"]
        table = windows[expected_seq].set_index("window_id")
        if not table.index.is_unique:
            raise ValueError("Duplicate window IDs")
        for ref in refs:
            key = (ref["seq_id"], ref["window_id"])
            if key in seen or ref["seq_id"] != expected_seq:
                raise ValueError("Duplicate or wrong-flight split reference")
            seen.add(key)
            if any(not isinstance(ref[c], int) for c in ("window_id", "t_start_ns", "t_end_ns")):
                raise ValueError("Split IDs/timestamps must be integers")
            if ref["window_id"] not in table.index:
                raise ValueError("Unknown window ID")
            row = table.loc[ref["window_id"]]
            if any(ref[c] != row[c] for c in ("seq_id", "t_start_ns", "t_end_ns")):
                raise ValueError("Split metadata differs from original window")
    expected_keys = {(s, int(i)) for s in sequences for i in windows[s].window_id}
    if seen != expected_keys:
        raise ValueError("Split partitions do not cover the source windows")
    if max(w["t_end_ns"] for w in fold["train"]) + gap_ns > min(w["t_start_ns"] for w in fold["validation"]):
        raise ValueError("Training and validation violate the temporal gap")


def training_subset(fold, fraction):
    """Nested chronological prefixes of eligible inner training; fixed val/test."""
    import copy
    import math
    if fraction not in (0.5, 0.75, 1.0):
        raise ValueError("A2 supports 50%, 75%, 100% only")
    result = copy.deepcopy(fold)
    ordered = sorted(result["train"], key=lambda row: row["t_start_ns"])
    count = math.ceil(len(ordered) * fraction)
    result["train"] = ordered[:count]
    result["excluded_subset"] = ordered[count:]
    return result
