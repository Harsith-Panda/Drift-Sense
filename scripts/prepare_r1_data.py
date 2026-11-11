"""Complete R1 preparation: base data + safe physics + A1/A2 manifests.

No model code is invoked. Existing Phase 1 outputs are read-only inputs.
"""
import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_dl_data import main as prepare_base
from vio_drift.data.dl_physics import prepare_physics, sha256
from vio_drift.data.normalize import fit_fold_normalizer
from vio_drift.data.splits import training_subset, validate_dl_fold


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(include_a2=True):
    prepare_base()
    config = yaml.safe_load((ROOT / "configs/dl_config.yaml").read_text())
    ablations = yaml.safe_load((ROOT / "configs/r1_ablations.yaml").read_text())
    phase1 = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    processed = ROOT / config["data"]["processed_dir"]
    sequences = config["data"]["sequences"]
    if sequences != ["MH_01_easy", "MH_02_easy"]:
        raise ValueError("Only the two easy flights are allowed")
    windows = {s: pd.read_csv(processed / f"windows_{s}.csv") for s in sequences}
    raw = {s: np.load(processed / f"windows_raw_{s}.npy", allow_pickle=False) for s in sequences}
    features = {s: pd.read_csv(processed / f"features_{s}.csv") for s in sequences}
    aligned = {s: pd.read_csv(ROOT / phase1["data"]["interim_dir"] / f"aligned_{s}.csv") for s in sequences}
    base = json.loads((processed / "splits_dl.json").read_text())
    fractions = ablations["A2"]["fractions"] if include_a2 else [1.0]
    jobs, counts = [], []
    for fraction in fractions:
        artifacts = processed if fraction == 1.0 else processed / "ablations_dl" / f"train_{round(fraction*100):03d}"
        split = deepcopy(base)
        if fraction != 1:
            split["folds"] = [training_subset(f, fraction) for f in base["folds"]]
            split["training_fraction"] = fraction
            split["subset_rule"] = "ceil(fraction * eligible inner-training windows), chronological prefix"
            write_json(artifacts / "splits_dl.json", split)
        for fold in split["folds"]:
            validate_dl_fold(fold, windows, split["gap_ns"])
            stats_dir = artifacts / "normalization_dl"
            stats_dir.mkdir(parents=True, exist_ok=True)
            normalizer = fit_fold_normalizer(fold, windows, raw, features)
            stats_path = stats_dir / f"{fold['fold_id']}.json"
            normalizer.save(stats_path)
            source = fold["training_source"]
            source_files = [processed / f"{prefix}_{source}.{suffix}" for prefix, suffix in
                            (("windows", "csv"), ("windows_raw", "npy"), ("features", "csv"))]
            write_json(stats_path.with_suffix(".sources.json"), {
                "statistics_sha256": sha256(stats_path),
                "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in source_files}})
            print(f"Preparing {fraction:.0%} {fold['fold_id']} physics (cached on repeat)", flush=True)
            prepare_physics(ROOT, artifacts, fold, windows, aligned, phase1["integration"])
            counts.append({"fraction": fraction, "fold_id": fold["fold_id"],
                           **{part: len(fold[part]) for part in ("train", "validation", "test")}})
            common = {"fold_id": fold["fold_id"], "artifact_dir": str(artifacts.relative_to(ROOT)),
                      "processed_dir": str(processed.relative_to(ROOT)), "model": ablations["model"],
                      "batch_size": config["batch_size"], "learning_rate": config["learning_rate"],
                      "device": "cpu", "use_camera": False,
                      "early_stopping_patience": config["early_stopping"]["patience"],
                      "early_stopping_min_delta": config["early_stopping"]["min_delta"],
                      "monitor": "validation_loss", "test_policy": "final_evaluation_only"}
            for seed in ablations["seeds"]:
                jobs.append({**common, "experiment": "A2", "seed": seed, "training_fraction": fraction,
                             "run_id": f"A2_{round(fraction*100):03d}_{fold['fold_id']}_s{seed}",
                             **{k: ablations["A2"][k] for k in ("dropout", "weight_decay", "early_stopping", "epochs", "scheduler")},
                             "augmentation": {"gaussian_noise": False, "random_crop": False},
                             "checkpoint_selection": "best_validation_loss"})
                if fraction != 1:
                    continue
                a1 = ablations["A1"]
                variants = [(f"d{int(d>0)}w{int(w>0)}e{int(e)}", d, w, e, False, False)
                            for d, w, e in product(a1["dropout"], a1["weight_decay"], a1["early_stopping"])]
                variants += [("noise_only", .3, .0001, True, True, False),
                             ("crop_mask_only", .3, .0001, True, False, True),
                             ("noise_and_crop_mask", .3, .0001, True, True, True)]
                for name, dropout, decay, early, noise, crop in variants:
                    jobs.append({**common, "experiment": "A1", "seed": seed,
                                 "run_id": f"A1_{name}_{fold['fold_id']}_s{seed}",
                                 "dropout": dropout, "weight_decay": decay, "early_stopping": early,
                                 "epochs": a1["epochs"], "scheduler": a1["scheduler"],
                                 "augmentation": {"gaussian_noise": noise, "random_crop": crop,
                                                  "noise_fraction": config["augmentation"]["noise_fraction"],
                                                  "crop_max_samples": config["augmentation"]["crop_max_samples"]},
                                 "checkpoint_selection": "best_validation_loss" if early else "final_epoch"})
    out = ROOT / "results/ablations/r1_data_manifest.json"
    write_json(out, {"schema_version": 1, "status": "data_ready_models_not_run",
                     "required_components": ["R2 models/dl/cnn1d.py", "R2 models/dl/train.py", "R5 evaluation/dl_eval.py"],
                     "a2_fraction_denominator": "eligible inner-training windows, after validation and gap",
                     "counts": counts, "jobs": jobs})
    print(f"Prepared {len(jobs)} experiment jobs; no model experiments run. {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-only", action="store_true", help="Skip 50/75 percent A2 preparation")
    prepare(include_a2=not parser.parse_args().base_only)
