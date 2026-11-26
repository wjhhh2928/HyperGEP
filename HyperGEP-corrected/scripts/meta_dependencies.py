"""Label and summarize frozen held-out Meta-Fusion dependency weights."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from hypergep.cli import completed_valid

def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for path in sorted(args.root.rglob("meta_attention_mean.csv")):
        state=json.loads((path.parent/"run_manifest.json").read_text(encoding="utf-8"))
        if state["status"]!="COMPLETED":
            continue
        if not completed_valid(path.parent,state):
            raise ValueError("Corrupt Meta-Fusion run")
        weights=pd.read_csv(path,index_col=0)
        if list(weights.index)!=list(weights.columns) or not np.allclose(weights.sum(axis=1),1,atol=1e-6):
            raise ValueError("Attention must sum to one over sources for each target")
        independent=weights.copy()
        np.fill_diagonal(independent.values,0)
        tag=state["run_id"][:16]
        weights.to_csv(args.output_dir/f"{tag}_target_by_source_weights.csv")
        independent.to_csv(args.output_dir/f"{tag}_off_diagonal.csv")
        pd.DataFrame({"incoming":independent.sum(axis=1),"outgoing":independent.sum(axis=0)}).to_csv(
            args.output_dir/f"{tag}_incoming_outgoing.csv")

if __name__=="__main__":
    main()
