"""Generate Fig. 5e–f coordinate plots from verified held-out environment runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hypergep.cli import completed_valid

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("results",type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--trait",default="Y")
    args=parser.parse_args()
    tables=[]
    for file in sorted(args.results.rglob("loeo_environment.csv")):
        record=json.loads((file.parent/"run_manifest.json").read_text(encoding="utf-8"))
        if record["status"]!="COMPLETED" or record["synthetic"] or record.get("scope")=="real_smoke":
            continue
        if not completed_valid(file.parent,record):
            raise ValueError("One LOEO result failed its SHA256 check")
        frame=pd.read_csv(file)
        tables.append(frame[frame.trait.eq(args.trait)].assign(data_hash=record["data_hash"]))
    if not tables:
        raise SystemExit("BLOCKED: verified formal environment results required")
    data=pd.concat(tables,ignore_index=True)
    if data[["dataset","data_hash","seed","model_version"]].drop_duplicates().shape[0]!=1 or data.environment_id.duplicated().any():
        raise ValueError("Choose one data version, model and seed")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/"heldout_environment_metrics.csv",index=False)
    if not {"latitude","longitude"}<=set(data) or data[["latitude","longitude"]].isna().any().any():
        raise SystemExit("BLOCKED: geocodes are missing for at least one held-out environment")
    for metric,panel in (("PCC","fig5e"),("MAE","fig5f")):
        fig,axis=plt.subplots(figsize=(7,5),constrained_layout=True)
        scatter=axis.scatter(data.longitude,data.latitude,c=data[metric],cmap="viridis",plotnonfinite=True)
        axis.set(xlabel="Longitude",ylabel="Latitude",title=f"{args.trait} held-out environment {metric}")
        fig.colorbar(scatter,ax=axis,label=metric)
        fig.savefig(out/f"{panel}_{metric}.png",dpi=160)
        plt.close(fig)

if __name__=="__main__":
    main()
