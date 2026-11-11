"""CPU-only R1 DataLoader demonstration; no model training."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from vio_drift.data.dl_dataset import DriftDataset, make_dataloader

def demonstrate():
    rows=[]
    for fold in json.loads((ROOT/"data/processed/splits_dl.json").read_text())["folds"]:
        for part in ("train","validation","test"):
            ds=DriftDataset(ROOT/"data/processed",fold["fold_id"],part)
            batch=next(iter(make_dataloader(ds)))
            shapes={k:list(v.shape) for k,v in batch.items() if k!="meta"}
            row={"fold":fold["fold_id"],"split":part,"windows":len(ds),"shapes":shapes}
            rows.append(row);print(row)
    return rows

if __name__=="__main__":
    demonstrate()
