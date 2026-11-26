"""Held-out perturbation against train-selected variance-matched marker controls."""
import numpy as np
import pandas as pd
import torch

from .model import Query

def select_control(rng, order, variance, top):
    """Sample one unused variance-near marker per important marker."""
    eligible = np.ones(len(variance),dtype=bool)
    eligible[top] = False
    positions = np.empty(len(order),dtype=int)
    positions[order] = np.arange(len(order))
    selected = []
    for marker in top:
        available = np.flatnonzero(eligible[order])
        closest = available[np.argsort(abs(available-positions[marker]),kind="stable")[:10]]
        if not len(closest):
            raise ValueError("Insufficient distinct variance-matched controls")
        chosen = order[rng.choice(closest)]
        eligible[chosen] = False
        selected.append(chosen)
    return np.asarray(selected)

def predict_masked(branch, data, rows, columns):
    outputs = []
    transform = branch.inputs.snp.transform
    with torch.no_grad():
        for start in range(0,len(rows),32):
            selected = rows[start:start+32]
            env,snp = branch.inputs.encode(selected)
            raw = data.snp[data.snp_lookup[selected]].copy()
            raw[:,columns] = transform.fill[columns]
            new_snp = Query(torch.as_tensor(transform.apply(raw),device=branch.inputs.snp.device),
                             snp.seen,snp.neighbors)
            outputs.append(branch.model(env,new_snp,branch.inputs.env.ref,branch.inputs.snp.ref).cpu().numpy()[:,0])
    return np.concatenate(outputs)*branch.scale[0]+branch.mean[0]

def masking_rows(branches, derivative, data, train, test, trait, rng, sizes, repeats):
    unique = data.snp[np.unique(data.snp_lookup[train])]
    valid = np.isfinite(unique)
    mean = np.where(valid,unique,0).sum(0,dtype=np.float64)/np.maximum(valid.sum(0),1)
    variance = np.where(valid,(unique-mean)**2,0).sum(0)/np.maximum(valid.sum(0),1)
    order = np.argsort(variance,kind="stable")
    common = data.ids.iloc[test][["sample_id","genotype_id","environment_id"]].copy()
    original = np.mean([branch.predict(test,32)[:,0] for branch in branches],axis=0)
    matrix = []
    for k in sizes:
        if k < 1 or k*2 > len(variance):
            continue
        important = np.argsort(-derivative,kind="stable")[:k]
        variants = [("importance",0,important)] + [
            ("variance_matched",i+1,select_control(rng,order,variance,important)) for i in range(repeats)]
        for strategy,replicate,columns in variants:
            prediction = np.mean([predict_masked(branch,data,test,columns) for branch in branches],axis=0)
            matrix.append((common.assign(trait=trait,k=k,strategy=strategy,replicate=replicate,
                           original=original,masked=prediction,abs_change=abs(prediction-original)),
                           dict(trait=trait,k=k,strategy=strategy,replicate=replicate,
                                markers=[data.markers[i] for i in columns],
                                training_marker_variance_mean=float(variance[columns].mean()))))
    return matrix
