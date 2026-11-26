"""Independent implementations of published genotype-only CNN topology."""
import json

import numpy as np
import torch
from torch import nn

from .data import Blocked, file_hash
from .train import FoldInputs, atomic_checkpoint, fit_network, seed_all

VARIANTS = ("DNNGP_reconstructed","DeepGS_reconstructed")

def build(variant, markers):
    if variant=="DeepGS_reconstructed":
        if markers<21:
            raise Blocked("DeepGS requires at least 21 ordered markers")
        return nn.Sequential(nn.Conv1d(1,8,18),nn.ReLU(),nn.MaxPool1d(4,4),
                             nn.Dropout(.2),nn.Flatten(),nn.Linear(8*((markers-17)//4),32),
                             nn.Sigmoid(),nn.Dropout(.1),nn.Dropout(.05),nn.Linear(32,1))
    if variant=="DNNGP_reconstructed":
        return nn.Sequential(nn.Conv1d(1,16,5,padding=2),nn.ReLU(),
                             nn.Conv1d(16,24,5,padding=2),nn.ReLU(),
                             nn.Conv1d(24,32,5,padding=2),nn.BatchNorm1d(32),nn.ReLU(),
                             nn.Dropout(.2),nn.Flatten(),nn.Linear(32*markers,32),
                             nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
    raise ValueError(variant)

def run_comparator(data, train, test, cfg, variant, seed, device, out, resume):
    # Fit preprocessing with the same reference fold. This method uses only SNP inputs.
    prep = FoldInputs.fit(data,train,cfg,variant,device,baseline=True)
    _, snp = prep.encode(train)
    train_x = snp.values[:,None]
    _, snp_test = prep.encode(test)
    test_x = snp_test.values[:,None]
    if prep.snp.transform.pca is not None:
        raise Blocked("Configured genomic CNN reconstruction requires marker-space features")
    settings = cfg["training_parameters"]
    method = dict(cfg, training_parameters=dict(settings, pearson_weight=0,
                                                loss_kind="mae" if variant.startswith("DeepGS") else "mse"))
    directory=out/"checkpoints"
    directory.mkdir(exist_ok=True)
    predictions, history = [], []
    for index,trait in enumerate(data.traits):
        branch_seed=seed+index*101
        seed_all(branch_seed)
        model=build(variant,train_x.shape[-1])
        path=directory/f"outer_comparator_{index}.pt"
        mean,scale,log=fit_network(model,lambda i:model(train_x[i]),data.y[train,index:index+1],
                                   cfg["baselines"]["epochs"],method,branch_seed,device,path,resume,
                                   {"marker_count":train_x.shape[-1],"variant":variant,"target":trait})
        rows=[]
        with torch.no_grad():
            for i in range(0,len(test),settings["batch_size"]):
                rows.append(model(test_x[i:i+settings["batch_size"]]).cpu().numpy())
        predictions.append(np.concatenate(rows)*scale+mean)
        history.extend(log)
    (out/"preprocessing_state.json").write_text(json.dumps(prep.state()),encoding="utf-8")
    atomic_checkpoint(out/"checkpoint.pt",{"variant":variant,"files":{p.name:file_hash(p) for p in directory.glob("*.pt")}})
    return np.column_stack(predictions),history,[],[]
