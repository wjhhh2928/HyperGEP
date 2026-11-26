"""Out-of-fold input gradients from frozen single-trait branches."""
import json

import numpy as np
import pandas as pd
import torch

from .masking import masking_rows
from .train import load_checkpoint, restore_base

def derivatives(branch, data, rows):
    transform = branch.inputs.snp.transform
    chunks = []
    for j in range(0,len(rows),32):
        selection = rows[j:j+32]
        env,snp = branch.inputs.encode(selection)
        snp.values.requires_grad_()
        score = branch.model(env,snp,branch.inputs.env.ref,branch.inputs.snp.ref)
        derivative = torch.autograd.grad(score.sum(),snp.values)[0].detach().cpu().numpy()
        if transform.pca is not None:
            derivative = derivative @ transform.pca.components_
        selected = derivative * branch.scale[0] / transform.scale
        raw = data.snp[data.snp_lookup[selection]]
        untransformed = np.zeros(raw.shape,dtype=np.float32)
        untransformed[:, transform.columns] = selected
        untransformed[~np.isfinite(raw)] = 0
        chunks.append(untransformed)
    return np.concatenate(chunks)

def gradients(data, rows, train, cfg, device, result):
    index = load_checkpoint(result / "checkpoint.pt", device)
    checkpoint_dir = result / "checkpoints"
    outputs, masking = [], []
    options = cfg.get("attribution", {})
    settings = options.get("masking_k", [30,100,300])
    repeats = options.get("masking_repeats", 5)
    if cfg["dataset"] == "demo":
        settings = [3,5]
    rng = np.random.default_rng(options.get("masking_seed", 42))
    for i, trait in enumerate(data.traits):
        names = ([index["outer_branches"][i]] if index["strategy"]=="refit"
                 else sorted(name for name in index["files"] if name.endswith(f"_branch_{i}.pt")))
        branches = [restore_base(data,checkpoint_dir/name,cfg,device) for name in names]
        outputs.append(np.abs(np.mean([derivatives(branch,data,rows) for branch in branches],axis=0)))
        if cfg.get("analysis","all") != "gradient" and trait in {"Y","PH","KNPR"}:
            ordered = train[np.argsort(data.ids.iloc[train].sample_id.to_numpy(),kind="stable")[:256]]
            importance = np.abs(np.mean([derivatives(branch,data,ordered) for branch in branches],axis=0)).mean(0)
            masking.extend(masking_rows(branches,importance,data,train,rows,trait,rng,settings,repeats))
    if masking:
        pd.concat([frame for frame,_ in masking],ignore_index=True).to_csv(result/"s6_heldout_masking.csv",index=False)
        (result/"s6_marker_selections.json").write_text(json.dumps([m for _,m in masking],indent=2),encoding="utf-8")
    ids = data.ids.iloc[rows]
    np.savez_compressed(result / "heldout_snp_gradients.npz", gradients=np.stack(outputs,axis=1),
                        traits=np.asarray(data.traits,dtype=str), markers=np.asarray(data.markers,dtype=str),
                        sample_id=ids.sample_id.to_numpy(dtype=str),
                        genotype_id=ids.genotype_id.to_numpy(dtype=str),
                        environment_id=ids.environment_id.to_numpy(dtype=str))
    spec = {"source":"held-out base branches", "score":"absolute gradient in raw target and marker units",
            "ensemble":"mean signed derivative before absolute value",
            "checkpoint_sha256":index["files"]}
    (result / "attribution_manifest.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
