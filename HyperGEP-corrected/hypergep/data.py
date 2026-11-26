"""Identifier alignment, provenance and fold-local transformations."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold

class Blocked(RuntimeError):
    """Required experimental evidence or resources are unavailable."""

def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()

def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()

def read_table(path, key, id_columns=()):
    path = Path(path)
    types = dict.fromkeys((key, *id_columns), str)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=types, keep_default_na=False, na_values=[""])
    elif path.suffix.lower() == ".tsv":
        frame = pd.read_csv(path, sep="\t", dtype=types, keep_default_na=False, na_values=[""])
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    if key not in frame or frame[key].isna().any():
        raise ValueError(f"Missing identifier values in {key}")
    ids = frame[key].astype(str)
    if ids.duplicated().any() or ids.str.strip().ne(ids).any():
        raise ValueError(f"Duplicate or padded identifier in {key}")
    return frame.set_index(key).sort_index()

def numeric(frame):
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or np.isinf(values).any():
        raise ValueError("Expected a nonempty numeric feature matrix without infinite values")
    return values

@dataclass
class Data:
    ids: pd.DataFrame
    env: np.ndarray
    snp: np.ndarray
    y: np.ndarray
    traits: list
    markers: list
    source: dict
    env_lookup: np.ndarray = None
    snp_lookup: np.ndarray = None
    environment_locations: object = None

    def __post_init__(self):
        if self.env_lookup is None:
            self.env_lookup = np.arange(len(self.ids))
        if self.snp_lookup is None:
            self.snp_lookup = np.arange(len(self.ids))

    def subset_traits(self, names):
        return replace(self, y=self.y[:, [self.traits.index(n) for n in names]], traits=list(names))

    def audit(self):
        pairs = self.ids[["genotype_id", "environment_id"]]
        return {
            "records": len(self.ids), "genotypes": int(pairs.genotype_id.nunique()),
            "environments": int(pairs.environment_id.nunique()),
            "markers": len(self.markers), "traits": self.traits,
            "repeated_genotype_environment_records": int(pairs.duplicated(keep=False).sum()),
            "valid_labels": dict(zip(self.traits, np.isfinite(self.y).sum(0).tolist())),
        }

def load_data(cfg, root):
    root = Path(root).resolve()
    paths = {key: (root / cfg["data_paths"][key]).resolve() for key in
             ("phenotypes", "genotypes", "environments", "provenance")}
    if any(root not in path.parents for path in paths.values()):
        raise ValueError("Input files must be contained in --data-dir")
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise Blocked("Data assets missing for " + cfg["dataset"] + ": " + ", ".join(missing))
    manifest = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    for key in ("dataset_version", "origin", "license", "label_provenance", "environment_availability"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise Blocked("Provenance is missing " + key)
    if manifest.get("preprocessing") != "raw" or manifest.get("row_alignment_verified") is not True:
        raise Blocked("Supply raw inputs with verified identifier alignment")
    hashes = {key: file_hash(path) for key, path in paths.items()}
    for key in ("phenotypes", "genotypes", "environments"):
        if manifest.get("sha256", {}).get(key) != hashes[key]:
            raise Blocked("Data SHA256 mismatch for " + key)
    sample = cfg["sample_id_column"]
    genotype = cfg["genotype_id_column"]
    environment = cfg["environment_id_column"]
    site = cfg.get("site_id_column")
    records = read_table(paths["phenotypes"], sample, [k for k in (genotype, environment, site) if k])
    geno = read_table(paths["genotypes"], genotype)
    env = read_table(paths["environments"], environment)
    traits = cfg["trait_names"]
    if len(set(traits)) != len(traits) or not traits:
        raise ValueError("Trait names must be unique and nonempty")
    ids = records[[genotype, environment]].rename(columns={genotype:"genotype_id", environment:"environment_id"})
    ids.insert(0, "sample_id", records.index)
    if site and site in records:
        ids["site_id"] = records[site]
    if ids.isna().any().any():
        raise ValueError("Phenotype identifiers must be present")
    if "site_id" in ids and ids.groupby("environment_id").site_id.nunique().gt(1).any():
        raise ValueError("Environment IDs must map to one site")
    for key in ("genotype_id", "environment_id"):
        ids[key] = ids[key].astype(str)
    env_lookup = env.index.get_indexer(ids["environment_id"])
    snp_lookup = geno.index.get_indexer(ids["genotype_id"])
    if np.any(env_lookup < 0) or np.any(snp_lookup < 0):
        raise ValueError("Unmatched phenotype environment/genotype identifiers")
    return Data(ids.reset_index(drop=True), numeric(env), numeric(geno), numeric(records[traits]),
                traits, geno.columns.to_list(), {"file_hashes": hashes, "provenance": manifest},
                env_lookup, snp_lookup,
                env[["latitude","longitude"]].copy() if {"latitude","longitude"} <= set(env) else None)

def demo_data():
    rng = np.random.default_rng(90)
    n, markers = 72, 48
    env, snp = rng.normal(size=(8, 5)), rng.integers(0, 3, (24, markers)).astype(float)
    eid, gid = np.arange(n) % 8, rng.permutation(np.tile(np.arange(24), 3))
    y = snp[gid, :3] + env[eid, :3] + rng.normal(0, .2, (n, 3))
    y[::13, 1] = np.nan
    ids = pd.DataFrame({"sample_id": [f"s{i:03d}" for i in range(n)],
                        "genotype_id": [f"g{i:03d}" for i in gid],
                        "environment_id": [f"e{i:03d}" for i in eid],
                        "site_id": [f"p{i//2}" for i in eid]})
    return Data(ids, env, snp, y, ["Y", "PH", "GM"], [f"marker_{i}" for i in range(markers)],
                {"dataset_version":"synthetic-demo-1", "synthetic":True}, eid, gid)

GROUPS = {"genotype": "genotype_id", "environment": "environment_id", "site": "site_id"}

def splits(data, indices, strategy, count, seed, minimum=1):
    indices = np.asarray(indices, dtype=int)
    if strategy == "record":
        pairs = KFold(count, shuffle=True, random_state=seed).split(indices)
        groups = None
    elif strategy == "loeo":
        groups = data.ids.iloc[indices]["environment_id"].to_numpy()
        labels = sorted(set(groups))
        pairs = ((np.flatnonzero(groups != label), np.flatnonzero(groups == label)) for label in labels)
    elif strategy == "genotype_environment":
        frame = data.ids.iloc[indices]
        groups = pd.factorize(pd.MultiIndex.from_frame(frame[["genotype_id", "environment_id"]]))[0]
        pairs = GroupKFold(count).split(indices, groups=groups)
    elif strategy in GROUPS:
        column = GROUPS[strategy]
        if column not in data.ids:
            raise Blocked("Group identifier required: " + column)
        groups = data.ids.iloc[indices][column].to_numpy()
        pairs = GroupKFold(count).split(indices, groups=groups)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    yielded = False
    for fold, (train, test) in enumerate(pairs):
        if len(test) < minimum:
            continue
        if not len(train) or np.intersect1d(train, test).size:
            raise ValueError("Invalid fold")
        if groups is not None and np.intersect1d(groups[train], groups[test]).size:
            raise ValueError("Grouped fold leaked an identifier")
        yielded = True
        yield fold, indices[train], indices[test]
    if not yielded:
        raise Blocked("No folds meet the requested minimum test size")

@dataclass
class Transform:
    fill: np.ndarray
    columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    pca: object

    @classmethod
    def fit(cls, train, select=None, components=None):
        observed = np.where(np.isfinite(train), train, 0)
        counts = np.isfinite(train).sum(0)
        fill = observed.sum(0, dtype=np.float64) / np.maximum(counts, 1)
        complete = np.where(np.isfinite(train), train, fill)
        columns = np.arange(train.shape[1])
        if select is not None:
            if not 0 < select <= train.shape[1]:
                raise ValueError("Feature count must be within the available feature dimensions")
            columns = np.sort(np.argsort(-complete.var(0), kind="stable")[:select])
        complete = complete[:, columns]
        mean, scale = complete.mean(0), complete.std(0)
        scale[scale < 1e-12] = 1
        pca = PCA(n_components=components, svd_solver="full").fit((complete - mean) / scale) if components else None
        return cls(fill, columns, mean, scale, pca)

    def apply(self, x):
        x = np.where(np.isfinite(x), x, self.fill)
        x = (x[:, self.columns] - self.mean) / self.scale
        if self.pca is not None:
            x = self.pca.transform(x)
        return x.astype(np.float32)

    def state(self):
        return {"fill":self.fill.tolist(), "columns":self.columns.tolist(), "mean":self.mean.tolist(),
                "scale":self.scale.tolist(), "pca_mean": self.pca.mean_.tolist() if self.pca else None,
                "pca_components":self.pca.components_.tolist() if self.pca else None}

    @classmethod
    def from_state(cls, state):
        pca = None
        if state["pca_components"] is not None:
            pca = PCA()
            pca.mean_ = np.asarray(state["pca_mean"])
            pca.components_ = np.asarray(state["pca_components"])
        return cls(*(np.asarray(state[name]) for name in ("fill","columns","mean","scale")), pca)
