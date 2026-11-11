"""DL-specific calibration using only the union of inner-training intervals.

Reuse Phase 1 physics algorithms; keep original physics tables unchanged.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from vio_drift.data.windowing import Window
from vio_drift.features.physics import BIAS_COLS, physics_features, training_bias


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_interval_mask(aligned, train_refs):
    """Unique sensor rows in the union, including only in-range GT matches."""
    mask = np.zeros(len(aligned), dtype=bool)
    for ref in train_refs:
        start, end = ref["t_start_ns"], ref["t_end_ns"]
        inside = (aligned.t_ns >= start) & (aligned.t_ns < end)
        if "gt_t_ns" in aligned:
            inside &= (aligned.gt_t_ns >= start) & (aligned.gt_t_ns < end)
        mask |= inside.to_numpy()
    if not mask.any():
        raise ValueError("No aligned samples in inner-training intervals")
    return mask


def fit_training_bias(fold, aligned):
    source = fold["training_source"]
    frame = aligned[source]
    selected = frame.loc[training_interval_mask(frame, fold["train"])]
    if not np.isfinite(selected[BIAS_COLS]).all().all():
        raise ValueError("Nonfinite training calibration")
    return training_bias({source: selected}, [source])


def prepare_physics(root, artifact_dir, fold, windows, aligned, integration):
    """Write/check a provenance-bound fold table; cache only exact input matches."""
    root, artifact_dir = Path(root), Path(artifact_dir)
    out_dir = artifact_dir / "physics_dl"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"physics_{fold['fold_id']}.csv"
    meta_path = path.with_suffix(".json")
    inputs = {}
    for seq in windows:
        for relative in (f"data/interim/aligned_{seq}.csv", f"data/processed/windows_{seq}.csv"):
            inputs[relative] = sha256(root / relative)
    provenance = {"schema_version": 1, "fold_id": fold["fold_id"],
                  "train": fold["train"], "integration": integration, "inputs": inputs,
                  "calibration_rule": "union of inner-training sensor and matched-GT intervals"}
    if path.exists() and meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("provenance") == provenance and previous.get("table_sha256") == sha256(path):
            return path
    bias = fit_training_bias(fold, aligned)
    tables = []
    for seq, metadata in windows.items():
        frame = aligned[seq]
        ts = frame.t_ns.to_numpy()
        if not frame.t_ns.is_monotonic_increasing:
            raise ValueError("Aligned timestamps must be sorted")
        chunks = []
        for row in metadata.itertuples(index=False):
            lo, hi = np.searchsorted(ts, [row.t_start_ns, row.t_end_ns])
            rows = frame.iloc[lo:hi].reset_index(drop=True)
            if len(rows) < 2:
                raise ValueError("Insufficient aligned rows in physics window")
            chunks.append(Window(seq, row.window_id, row.t_start_ns, row.t_end_ns, rows))
        tables.append(physics_features(chunks, bias, **integration))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(path, index=False)
    meta_path.write_text(json.dumps({"provenance": provenance, "bias": bias.tolist(),
                                    "table_sha256": sha256(path)}, indent=2) + "\n")
    return path
