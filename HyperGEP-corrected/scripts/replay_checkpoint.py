"""Replay a local deep run's frozen checkpoints and verify its held-out predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from hypergep.cli import completed_valid
from hypergep.data import Transform, demo_data, fingerprint, load_data
from hypergep.model import MetaTraitFusion
from hypergep.train import load_checkpoint, restore_base

def replay(run,data_dir,device):
    state=json.loads((run/"run_manifest.json").read_text(encoding="utf-8"))
    if state["status"]!="COMPLETED" or not completed_valid(run,state):
        raise ValueError("A verified completed local run is required")
    cfg=yaml.safe_load((run/"config_resolved.yaml").read_text(encoding="utf-8"))
    data=demo_data() if cfg["dataset"]=="demo" else load_data(cfg,data_dir)
    if fingerprint(data.source)!=state["data_hash"]:
        raise ValueError("Replay data identity differs from training")
    mapping=pd.read_csv(run/"split_indices.tsv",sep="\t",dtype={"sample_id":str})
    outer=mapping[mapping.role=="outer_test"]
    rows=outer.source_index.to_numpy(dtype=int)
    if data.ids.iloc[rows].sample_id.tolist()!=outer.sample_id.tolist():
        raise ValueError("Held-out rows no longer align with the original inputs")
    directory=run/"checkpoints"
    index=load_checkpoint(run/"checkpoint.pt",device)
    torch.set_num_threads(cfg["hardware"]["cpu_threads"])
    size=cfg["training_parameters"]["batch_size"]
    def branches(names):
        return np.column_stack([restore_base(data,directory/name,cfg,device).predict(rows,size)
                                for name in names])
    if index["strategy"]=="ensemble":
        stages=sorted({name.rsplit("_branch_",1)[0] for name in index["files"] if name.startswith("inner_")})
        base=np.mean([branches(sorted((name for name in index["files"] if name.startswith(stage+"_branch_")),
                                      key=lambda n:int(n.rsplit("_",1)[1].split(".")[0])))
                      for stage in stages],axis=0)
    else:
        base=branches(index["outer_branches"])
    traits=data.traits if index["variant"] not in ("STL","wo_MetaFusion") else cfg["output_traits"]
    if index["meta"]:
        meta_state=load_checkpoint(directory/index["meta"],device)
        inputs=Transform.from_state(meta_state["metadata"]["preprocessing"])
        n=len(traits)
        model=MetaTraitFusion(n,cfg["model_parameters"]["width"],cfg["model_parameters"]["dropout"]).to(device)
        model.load_state_dict(meta_state["model"])
        model.eval()
        with torch.no_grad():
            base=model(torch.as_tensor(inputs.apply(base)[:,:,None],device=device)).cpu().numpy()
        base=base*np.asarray(meta_state["scale"])+np.asarray(meta_state["mean"])
    saved=pd.read_csv(run/"predictions.csv",dtype={"sample_id":str},float_precision="round_trip")
    for name in cfg["output_traits"]:
        target=saved[saved.trait==name]
        if target.sample_id.tolist()!=outer.sample_id.tolist():
            raise ValueError("Prediction identities do not match the fold")
        np.testing.assert_allclose(base[:,traits.index(name)],target.predicted_value,rtol=1e-5,atol=1e-6)
    return len(rows),len(cfg["output_traits"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--data-dir",type=Path,default=Path("data"))
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    print("Checkpoint replay verified",replay(args.run,args.data_dir,args.device))

if __name__=="__main__":
    main()
