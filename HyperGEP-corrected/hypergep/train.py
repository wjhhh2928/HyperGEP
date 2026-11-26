"""Fold-local fitting, cross-fitting, model snapshots and held-out prediction."""
from dataclasses import dataclass
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

from .data import Blocked, Transform, file_hash, fingerprint, splits
from .model import BaseModel, MetaTraitFusion, Query, Reference, nearest

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

def atomic_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)

def load_checkpoint(path, device):
    # Only load checkpoints written locally by this fold runner.
    if "weights_only" in __import__("inspect").signature(torch.load).parameters:
        return torch.load(path, map_location=device, weights_only=True)
    return torch.load(path, map_location=device)

@dataclass
class InputSpace:
    transform: Transform
    reference_indices: np.ndarray
    ref: object
    values: np.ndarray
    device: str

    @classmethod
    def fit(cls, raw, lookup, train, params, preprocess, device, graph):
        indices = np.unique(lookup[train])
        transform = Transform.fit(raw[indices], preprocess.get("select"), preprocess.get("pca"))
        return cls.from_state(raw, {"indices": indices.tolist(), "transform": transform.state()},
                              params, device, graph)

    @classmethod
    def from_state(cls, raw, state, params, device, graph):
        indices = np.asarray(state["indices"], dtype=int)
        transform = Transform.from_state(state["transform"])
        # Transform fitted exclusively on reference IDs. Test rows are transformed independently on demand.
        values = transform.apply(raw[indices])
        ref = Reference.build(values, params["k_neighbors"], device) if graph else None
        return cls(transform, indices, ref, raw, device)

    def query(self, ids):
        values = self.transform.apply(self.values[ids])
        tensor = torch.as_tensor(values, device=self.device)
        seen = np.searchsorted(self.reference_indices, ids)
        seen = np.minimum(seen, len(self.reference_indices) - 1)
        seen = np.where(self.reference_indices[seen] == ids, seen, -1)
        neighbors = nearest(values, self.ref.values.detach().cpu().numpy(),
                            self.ref.node_for_edge.shape[1]) if self.ref is not None else np.zeros((len(ids), 1), dtype=int)
        return Query(tensor, torch.as_tensor(seen, dtype=torch.long, device=self.device),
                     torch.as_tensor(neighbors, dtype=torch.long, device=self.device))

    def state(self):
        return {"indices":self.reference_indices.tolist(), "transform":self.transform.state()}

@dataclass
class FoldInputs:
    env: InputSpace
    snp: InputSpace
    env_lookup: np.ndarray
    snp_lookup: np.ndarray

    @classmethod
    def fit(cls, data, train, cfg, variant, device, baseline=False):
        params = cfg["model_parameters"]
        pp = cfg["preprocessing"]
        env = InputSpace.fit(data.env, data.env_lookup, train, params, pp["environment"], device,
                             not baseline and variant != "wo_HGAT")
        snp = InputSpace.fit(data.snp, data.snp_lookup, train, params, pp["snp"], device,
                             not baseline and variant == "wo_CNN")
        return cls(env, snp, data.env_lookup, data.snp_lookup)

    @classmethod
    def restore(cls, data, state, cfg, variant, device):
        p = cfg["model_parameters"]
        return cls(InputSpace.from_state(data.env,state["env"],p,device,variant!="wo_HGAT"),
                   InputSpace.from_state(data.snp,state["snp"],p,device,variant=="wo_CNN"),
                   data.env_lookup, data.snp_lookup)

    def encode(self, i):
        return self.env.query(self.env_lookup[i]), self.snp.query(self.snp_lookup[i])

    def state(self):
        return {"env":self.env.state(), "snp":self.snp.state()}

def label_state(y):
    valid = np.isfinite(y)
    if not valid.any(0).all():
        raise Blocked("A training split lacks all observations for at least one active trait")
    mean = np.nanmean(y, axis=0).astype(np.float32)
    scale = np.nanstd(y, axis=0).astype(np.float32)
    scale[scale < 1e-12] = 1
    return mean, scale

def masked_loss(pred, label, pearson_weight=0.0, kind='mse'):
    tasks = []
    for column in range(label.shape[1]):
        valid = torch.isfinite(label[:, column])
        if not valid.any():
            continue
        x, y = pred[valid, column], label[valid, column]
        error = x-y
        loss = error.abs().mean() if kind == 'mae' else error.square().mean()
        if pearson_weight and len(y) >= 3:
            a, b = x-x.mean(), y-y.mean()
            if b.norm() > 1e-8:
                loss = loss + pearson_weight * (1 - (a*b).sum() / (a.norm()*b.norm()).clamp_min(1e-8))
        tasks.append(loss)
    return torch.stack(tasks).mean() if tasks else None

def fit_network(model, forward, labels, epochs, cfg, seed, device, path, resume, metadata):
    """Checkpoint at epoch boundaries. Deterministic epoch seeds restore dropout and shuffling."""
    p = cfg["training_parameters"]
    model.to(device)
    mean, scale = label_state(labels)
    labels = torch.as_tensor((labels - mean) / scale, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    first, log = 0, []
    if resume and path.is_file():
        state = load_checkpoint(path, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first, log = state["epoch"], state["log"]
    for epoch in range(first, epochs):
        seed_all((seed + epoch + 1) % (2**32))
        model.train()
        losses = []
        start = time.perf_counter()
        order = np.random.default_rng(seed + epoch).permutation(len(labels))
        for i in range(0, len(order), p["batch_size"]):
            batch = order[i:i+p["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(forward(batch), labels[batch], p["pearson_weight"], p.get("loss_kind","mse"))
            if loss is not None:
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())
        log.append({"stage":path.stem, "epoch":epoch+1, "loss":float(np.mean(losses)),
                    "seconds":time.perf_counter()-start,
                    "parameter_count":sum(p.numel() for p in model.parameters())})
        payload = {"model":model.state_dict(), "optimizer":optimizer.state_dict(),
                   "epoch":epoch+1, "log":log, "mean":mean.tolist(), "scale":scale.tolist(),
                   "metadata":metadata}
        if (epoch + 1) % p["checkpoint_every"] == 0 or epoch + 1 == epochs:
            atomic_checkpoint(path, payload)
    model.eval()
    return mean, scale, log

def make_base(inputs, traits, cfg, variant):
    return BaseModel(len(inputs.env.transform.apply(inputs.env.values[:1])[0]),
                     len(inputs.snp.transform.apply(inputs.snp.values[:1])[0]), traits,
                     cfg["model_parameters"], variant)

@dataclass
class FrozenBase:
    inputs: FoldInputs
    model: BaseModel
    mean: np.ndarray
    scale: np.ndarray

    def predict(self, i, batch_size):
        rows = []
        with torch.no_grad():
            for start in range(0, len(i), batch_size):
                env, snp = self.inputs.encode(i[start:start+batch_size])
                rows.append(self.model(env, snp, self.inputs.env.ref, self.inputs.snp.ref).cpu().numpy())
        return np.concatenate(rows) * self.scale + self.mean

def fit_base(inputs, train, labels, cfg, variant, seed, device, path, resume):
    seed_all(seed)
    model = make_base(inputs, labels.shape[1], cfg, variant)
    env, snp = inputs.encode(train)
    def forward(idx):
        return model(env[idx], snp[idx], inputs.env.ref, inputs.snp.ref)
    mean, scale, log = fit_network(model, forward, labels, cfg["training_parameters"]["base_epochs"],
                                   cfg, seed, device, path, resume,
                                   {"preprocessing":inputs.state(), "variant":variant,
                                    "outputs":labels.shape[1]})
    return FrozenBase(inputs, model, mean, scale), log

def restore_base(data, path, cfg, device):
    state = load_checkpoint(path, device)
    m = state["metadata"]
    inputs = FoldInputs.restore(data, m["preprocessing"], cfg, m["variant"], device)
    model = make_base(inputs, m["outputs"], cfg, m["variant"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return FrozenBase(inputs, model, np.asarray(state["mean"]), np.asarray(state["scale"]))

def id_hash(data, rows):
    return fingerprint(sorted(data.ids.iloc[rows].sample_id.tolist()))

BASELINES = ("GBLUP_Proxy", "AdaBoost", "RandomForest", "XGBoost", "LightGBM", "MLP", "TabNet")

def baseline_estimator(name, settings, seed, device):
    if name == "GBLUP_Proxy":
        return KernelRidge(alpha=settings["kernel_alpha"], kernel="linear")
    if name == "AdaBoost":
        return AdaBoostRegressor(n_estimators=settings["trees"], random_state=seed)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=settings["trees"], random_state=seed, n_jobs=1)
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(64,32), max_iter=settings["epochs"], random_state=seed)
    try:
        if name == "XGBoost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=settings["trees"], random_state=seed, n_jobs=1)
        if name == "LightGBM":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=settings["trees"], random_state=seed, n_jobs=1, verbosity=-1)
        if name == "TabNet":
            from pytorch_tabnet.tab_model import TabNetRegressor
            return TabNetRegressor(seed=seed, device_name=device, verbose=0)
    except ImportError as exc:
        raise Blocked(f"Optional baseline dependency missing for {name}: {exc.name}") from exc
    raise ValueError(f"Unknown baseline {name}")

def run_baseline(data, train, test, cfg, variant, seed, device, out):
    from joblib import dump
    prep = FoldInputs.fit(data, train, cfg, variant, "cpu", baseline=True)
    def features(rows):
        env, snp = prep.encode(rows)
        g = snp.values.numpy()
        if variant == "GBLUP_Proxy":
            return g / np.sqrt(g.shape[1])
        return np.concatenate((g, env.values.numpy()), 1)
    x, x_test = features(train), features(test)
    predictions, models, log = [], [], []
    for i, trait in enumerate(data.traits):
        start = time.perf_counter()
        y = data.y[train, i]
        valid = np.isfinite(y)
        if not valid.any():
            raise Blocked("No train labels for " + trait)
        method = baseline_estimator(variant, cfg["baselines"], seed, device)
        if variant == "TabNet":
            method.fit(x[valid], y[valid, None], max_epochs=cfg["baselines"]["epochs"],
                       patience=0, batch_size=64, virtual_batch_size=16, drop_last=False)
        else:
            method.fit(x[valid], y[valid])
        predictions.append(method.predict(x_test).reshape(-1))
        models.append(method)
        log.append({"stage":trait, "epoch":None, "seconds":time.perf_counter()-start,
                    "training_count":int(valid.sum())})
    dump(models, out / "checkpoint.joblib", compress=3)
    (out / "preprocessing_state.json").write_text(json.dumps(prep.state()), encoding="utf-8")
    return np.column_stack(predictions), log, [], []

def run_deep(data, train, test, cfg, variant, seed, device, out, resume):
    """Fit using outer-train labels only. Callers access outer-test labels after return."""
    strategy = cfg["inner_cv"]["strategy"]
    count = len(data.traits)
    params, tr = cfg["model_parameters"], cfg["training_parameters"]
    size = tr["batch_size"]
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    log, ledger, preprocessing, graph_stats = [], [], {}, []
    use_meta = variant not in {"STL", "MTL", "wo_MetaFusion"}
    repl = cfg["evaluation"]["test_strategy"]
    def branches(ref, requested, stage, inner_fold=None):
        inputs = FoldInputs.fit(data, ref, cfg, variant, device)
        prediction_list, checkpoint_names = [], []
        targets = [list(range(count))] if variant == "MTL" else [[i] for i in range(count)]
        supervision_hash = id_hash(data, ref)
        for target in targets:
            checkpoint = checkpoints / f"{stage}_branch_{target[0]}.pt"
            labels = data.y[np.ix_(ref,target)]
            fitted, history = fit_base(inputs, ref, labels, cfg, variant, seed + target[0]*101,
                                       device, checkpoint, resume)
            prediction = fitted.predict(requested, size)
            prediction_list.append(prediction)
            checkpoint_names.append(checkpoint.name)
            log.extend(history)
            if inner_fold is not None:
                if np.intersect1d(ref, requested).size:
                    raise AssertionError("Inner prediction trained on its own sample")
                valid_train = ref[np.isfinite(labels[:, 0])]
                supervised = id_hash(data, valid_train)
                checkpoint_id = file_hash(checkpoint)
                for j, row in enumerate(requested):
                    record = data.ids.iloc[row].to_dict()
                    ledger.append(dict(record, inner_fold=inner_fold, trait=data.traits[target[0]],
                                       true_value=data.y[row,target[0]], oof_prediction=float(prediction[j,0]),
                                       training_indices_hash=supervision_hash,
                                       supervised_indices_hash=supervised,
                                       checkpoint_id=checkpoint_id, checkpoint=checkpoint.name))
        preprocessing[stage] = inputs.state()
        for name, space in (("environment",inputs.env),("genotype",inputs.snp)):
            if space.ref is not None:
                membership = space.ref.node_for_edge
                degrees = (space.ref.edge_for_node>=0).sum(1)
                graph_stats.append(dict(stage=stage,graph=name,nodes=len(membership),edges=len(membership),
                                        requested_k=params["k_neighbors"],effective_k=membership.shape[1],
                                        incidences=membership.numel(),min_degree=int(degrees.min()),
                                        max_degree=int(degrees.max())))
        return np.column_stack(prediction_list), checkpoint_names

    split_records = []
    if use_meta:
        oof = np.full((len(train), count), np.nan, dtype=np.float32)
        visits = np.zeros(len(train), dtype=int)
        row_position = {row:i for i,row in enumerate(train)}
        inner_tests = []
        for fold, subtrain, validation in splits(data, train, strategy, cfg["inner_cv"]["folds"], seed):
            if np.intersect1d(subtrain,test).size or np.intersect1d(validation,test).size:
                raise AssertionError("Inner splits escaped outer train")
            requested = np.concatenate((validation,test)) if repl == "ensemble" else validation
            # Outer queries travel through a frozen branch and are excluded from the OOF ledger.
            fitted, names = branches(subtrain, validation, f"inner_{fold}", fold)
            position = [row_position[row] for row in validation]
            oof[position], visits[position] = fitted, visits[position]+1
            split_records.extend((fold, int(row),"train") for row in subtrain)
            split_records.extend((fold, int(row),"validation") for row in validation)
            if repl == "ensemble":
                inner_tests.append(np.column_stack([
                    restore_base(data, checkpoints / name, cfg, device).predict(requested[len(validation):], size)
                    for name in names]))
        if not np.all(visits == 1) or not np.isfinite(oof).all():
            raise AssertionError("OOF features require exactly one finite prediction per row and trait")
        xform = Transform.fit(oof)
        train_meta = torch.as_tensor(xform.apply(oof)[:, :, None], device=device)
        seed_all(seed + 2603)
        meta = MetaTraitFusion(count, params["width"], params["dropout"])
        meta_path = checkpoints / "meta.pt"
        mean, scale, history = fit_network(meta, lambda idx: meta(train_meta[idx]), data.y[train],
                                           tr["meta_epochs"], cfg, seed+2603, device, meta_path,
                                           resume, {"preprocessing":xform.state(), "outputs":count})
        log.extend(history)
        preprocessing["meta"] = xform.state()
        if repl == "ensemble":
            base_test = np.mean(inner_tests, axis=0)
            names = []
        else:
            base_test, names = branches(train,test,"outer")
        with torch.no_grad():
            predictions, weights = meta(torch.as_tensor(xform.apply(base_test)[:, :, None], device=device),
                                        return_weights=True)
            predictions = predictions.cpu().numpy() * scale + mean
            pd.DataFrame(weights.mean(0).cpu().numpy(), index=data.traits, columns=data.traits).rename_axis(
                "target_trait").to_csv(out / "meta_attention_mean.csv")
    else:
        predictions, names = branches(train,test,"outer")
    (out / "preprocessing_state.json").write_text(json.dumps(preprocessing, allow_nan=False), encoding="utf-8")
    if graph_stats:
        pd.DataFrame(graph_stats).to_csv(out/"graph_statistics.csv",index=False)
    atomic_checkpoint(out / "checkpoint.pt", {
        "files":{p.name:file_hash(p) for p in sorted(checkpoints.glob("*.pt"))},
        "outer_branches":names, "meta":"meta.pt" if use_meta else "",
        "strategy":repl if use_meta else "refit", "variant":variant})
    return predictions, log, ledger, split_records

def metrics(frame):
    group_keys = ["dataset","evaluation_protocol","experiment","model","model_version","seed","fold","trait"]
    results = []
    for keys, group in frame.groupby(group_keys, sort=False, dropna=False):
        valid = group.label_valid.astype(bool).to_numpy()
        y = group.true_value.to_numpy(dtype=float)[valid]
        pred = group.predicted_value.to_numpy(dtype=float)[valid]
        if not np.isfinite(pred).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite prediction or observed target")
        n = len(y)
        residual = pred-y
        mse = float(np.mean(residual**2)) if n else np.nan
        undefined = ("insufficient_samples" if n < 2 else "constant_observed" if np.ptp(y) == 0
                     else "constant_prediction" if np.ptp(pred) == 0 else "")
        pcc = np.nan if undefined else float(np.corrcoef(y,pred)[0,1])
        r2 = 1-float(np.sum(residual**2) / np.sum((y-y.mean())**2)) if n > 1 and np.ptp(y) > 0 else np.nan
        results.append(dict(zip(group_keys,keys), valid_sample_count=n, PCC=pcc, PCC_reason=undefined,
                            MAE=float(np.mean(abs(residual))) if n else np.nan, MSE=mse,
                            RMSE=np.sqrt(mse), R2=r2,
                            signed_mean_error=float(np.mean(residual)) if n else np.nan))
    return pd.DataFrame(results)
