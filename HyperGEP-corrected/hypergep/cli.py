"""Experiment planning, immutable run identity and transactional persistence."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from . import __version__
from .comparators import VARIANTS, run_comparator
from .data import Blocked, demo_data, file_hash, fingerprint, load_data, splits
from .train import BASELINES, id_hash, metrics, run_baseline, run_deep

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "main": ("Full_Model",),
    "baselines": BASELINES,
    "genomic_comparators": VARIANTS,
    "ablation": ("Full_Model", "wo_HGAT", "wo_CNN", "wo_CrossAttention", "wo_MetaFusion", "legacy_fusion"),
    "negative_transfer": ("STL", "MTL", "Full_Model"),
    "sensitivity": (),
    "loeo": ("Full_Model",),
    "interpretability": ("Full_Model",),
}
STRATEGIES = ("record","genotype","environment","site","genotype_environment")

def arguments(default):
    parser = argparse.ArgumentParser(description="HyperGEP corrected evaluation")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=default)
    parser.add_argument("--dataset", choices=("maizegep","g2f","demo"), default="maizegep")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", "--model", "--strategy", dest="variant", default="all")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--analysis", choices=("all","gradient"), default="all")
    parser.add_argument("--parameter", choices=("k_neighbors",), default="k_neighbors")
    parser.add_argument("--fold", type=int, help="Zero-based outer fold")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("DATA_DIR","data")))
    parser.add_argument("--output-dir", type=Path, default=os.getenv("OUTPUT_DIR") or None)
    parser.add_argument("--split", choices=STRATEGIES)
    parser.add_argument("--test-strategy", choices=("refit","ensemble"))
    parser.add_argument("--min-test-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser

def resolved(args):
    mode = "demo" if args.dataset == "demo" else "paper"
    source = args.config or ROOT / "configs" / (mode + ".yaml")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    datasets = cfg.pop("datasets")
    if args.dataset not in datasets:
        raise ValueError("This configuration does not describe " + args.dataset)
    cfg.update(datasets[args.dataset])
    cfg.update(dataset=args.dataset, experiment=args.experiment)
    if args.experiment=="interpretability":
        cfg["analysis"]=args.analysis
    if cfg["model_version"] != __version__ or cfg["graph_mode"] != "inductive":
        raise ValueError("This code implements corrected-v1 inductive evaluation")
    if args.experiment == "loeo" and args.split:
        raise ValueError("LOEO supplies its own outer split")
    if args.min_test_samples is not None and (args.experiment != "loeo" or args.min_test_samples < 1):
        raise ValueError("Minimum test samples applies only to LOEO and must be positive")
    if args.split:
        cfg["outer_cv"]["strategy"] = args.split
    if args.experiment == "loeo":
        cfg["outer_cv"]["strategy"] = "loeo"
        if args.min_test_samples:
            cfg["evaluation"]["loeo_minimum"] = args.min_test_samples
    outer = cfg["outer_cv"]["strategy"]
    inner = cfg["inner_cv"]["strategy"]
    expected_inner = "environment" if outer == "loeo" else outer
    if inner == "inherit":
        cfg["inner_cv"]["strategy"] = expected_inner
    elif inner not in STRATEGIES or (outer != "record" and inner != expected_inner):
        raise ValueError("Inner CV must preserve the outer grouping unit")
    if args.test_strategy:
        cfg["evaluation"]["test_strategy"] = args.test_strategy
    if cfg["evaluation"]["test_strategy"] not in ("refit","ensemble"):
        raise ValueError("Unknown test strategy")
    if args.seed is not None:
        cfg["random_seeds"] = [args.seed]
    if any(not 0 <= s < 2**32-10000 for s in cfg["random_seeds"]):
        raise ValueError("Seed out of range")
    if args.device:
        cfg["hardware"]["device"] = args.device
    if args.trait:
        invalid = set(args.trait)-set(cfg["trait_names"])
        if invalid:
            raise ValueError("Unknown traits: " + str(sorted(invalid)))
    if min(cfg["inner_cv"]["folds"],cfg["outer_cv"]["folds"]) < 2:
        raise ValueError("CV requires at least two folds")
    for field in ("base_epochs","meta_epochs","batch_size","checkpoint_every"):
        if cfg["training_parameters"][field] < 1:
            raise ValueError("Positive training setting required: " + field)
    if args.experiment == "sensitivity":
        variants = tuple(f"k{n}" for n in cfg["evaluation"]["k_candidates"])
    else:
        variants = EXPERIMENTS[args.experiment]
    if args.variant != "all":
        if args.variant not in variants:
            raise ValueError(f"Variant must be one of {', '.join(variants)}")
        variants = (args.variant,)
    return cfg, variants

def utc():
    return datetime.now(timezone.utc).isoformat()

def atomic_text(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)

def manifest(path, result):
    atomic_text(path, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

@contextmanager
def task_lock(path):
    # OS locks are released on process exit, including crashes.
    with path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            def lock():
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            def unlock():
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def lock():
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def unlock():
                fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise Blocked("This exact task is already running") from exc
        try:
            yield
        finally:
            stream.seek(0)
            unlock()

def code_identity():
    paths = [p for sub in ("hypergep","experiments") for p in (ROOT/sub).glob("*.py")]
    content = fingerprint({p.relative_to(ROOT).as_posix():file_hash(p) for p in paths})
    try:
        commit = subprocess.run(["git","rev-parse","--show-toplevel","HEAD"], cwd=ROOT, text=True,
                                capture_output=True, check=True).stdout.splitlines()
        commit = commit[1] if Path(commit[0]).resolve() == ROOT else None
    except (IndexError, OSError, subprocess.CalledProcessError):
        commit = None
    return commit, content

def provenance_version(data):
    return data.source.get("dataset_version") or data.source["provenance"]["dataset_version"]

def environment_info(device):
    return {"python":platform.python_version(), "pytorch":str(torch.__version__),
            "numpy":np.__version__, "pandas":pd.__version__, "sklearn":sklearn.__version__,
            "platform":platform.system(), "device":device, "cuda":torch.version.cuda,
            "cuda_device":torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "visible_cuda_devices":torch.cuda.device_count() if device.startswith("cuda") else 0}

def completed_valid(out, record):
    return bool(record.get("output_sha256")) and all(
        (out/name).is_file() and file_hash(out/name) == digest
        for name,digest in record["output_sha256"].items())

def execute(data, cfg, variant, seed, fold, train, test, out, resume, traits):
    protocol = cfg["outer_cv"]["strategy"] + "/" + cfg["graph_mode"]
    device = cfg["hardware"]["device"]
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or (torch.device(device).index or 0) >= torch.cuda.device_count():
            raise Blocked("Requested CUDA device is unavailable")
    elif device != "cpu":
        raise Blocked("Select cpu, cuda or cuda:N")
    runtime = environment_info(device)
    if device.startswith("cuda") and hasattr(torch.cuda, "mem_get_info"):
        available, total = torch.cuda.mem_get_info(torch.device(device))
        memory = {"cuda_free_bytes":available,"cuda_total_bytes":total}
    else:
        memory = {}
    commit, source = code_identity()
    identity = {"git_commit":commit,"code_hash":source,"config_hash":fingerprint(cfg),
                "data_hash":fingerprint(data.source),"dataset_version":provenance_version(data),
                "checkpoint_source":"trained_from_fold","train_indices_hash":id_hash(data,train),
                "test_indices_hash":id_hash(data,test),"runtime":runtime,
                "dataset":cfg["dataset"],"model_version":cfg["model_version"],
                "experiment_name":cfg["experiment"],"variant":variant,
                "evaluation_protocol":protocol,"random_seed":seed,"fold":fold,"output_traits":traits}
    run_id = fingerprint(identity)
    run = out / cfg["model_version"] / cfg["dataset"] / cfg["experiment"] / variant / f"seed_{seed}" / f"fold_{fold:02d}" / run_id[:16]
    run.mkdir(parents=True, exist_ok=True)
    path = run / "run_manifest.json"
    with task_lock(run / ".lock"):
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior["run_id"] != run_id:
                raise RuntimeError("Run identity collision")
            if prior["status"] == "COMPLETED":
                if not completed_valid(run, prior):
                    raise Blocked("Completed run artifacts changed. Preserve this run and use a fresh output root")
                print(f"COMPLETED (reused) {variant} seed {seed} fold {fold}")
                return "COMPLETED"
            if not resume:
                raise Blocked("Incomplete run exists. Supply --resume to continue its checkpoints")
        state = dict(identity, start_time=utc(), end_time=None, run_id=run_id,
                     hardware=dict(runtime, **memory), software_versions=runtime,
                     synthetic=cfg["dataset"]=="demo", scope=data.source.get("provenance", {}).get("purpose", "paper_candidate"), status="RUNNING")
        manifest(path,state)
        try:
            use_data = data
            if cfg["experiment"] in {"baselines","genomic_comparators"} or variant in {"STL","wo_MetaFusion"}:
                use_data = data.subset_traits(traits)
            atomic_text(run / "config_resolved.yaml", yaml.safe_dump(dict(cfg,variant=variant,seed=seed,
                                                          fold=fold,output_traits=traits), sort_keys=True))
            manifest(run / "environment_info.json", dict(runtime, **memory))
            if cfg["experiment"] == "baselines":
                pred, history, ledger, inner = run_baseline(use_data,train,test,cfg,variant,seed,device,run)
            elif cfg["experiment"] == "genomic_comparators":
                pred, history, ledger, inner = run_comparator(use_data,train,test,cfg,variant,seed,device,run,resume)
            else:
                branch_variant = "Full_Model" if variant.startswith("k") else variant
                pred, history, ledger, inner = run_deep(use_data,train,test,cfg,branch_variant,seed,device,run,resume)
            if not np.isfinite(pred).all():
                raise FloatingPointError("A frozen model produced nonfinite predictions")
            if cfg["experiment"] == "interpretability":
                from .interpretability import gradients
                gradients(use_data, test, train, cfg, device, run)
            manifest(run / "data_audit.json", data.audit())
            metadata = {"dataset":cfg["dataset"], "experiment":cfg["experiment"], "seed":seed,
                        "fold":fold,"model":variant,"model_version":cfg["model_version"],
                        "evaluation_protocol":protocol,"dataset_version":provenance_version(data),
                        "data_hash":identity["data_hash"],"test_indices_hash":identity["test_indices_hash"]}
            parts = []
            for trait in traits:
                i = use_data.traits.index(trait)
                obs = use_data.y[test,i]
                frame = use_data.ids.iloc[test].copy()
                frame = frame.assign(**metadata, trait=trait, true_value=obs, predicted_value=pred[:,i],
                                     label_valid=np.isfinite(obs))
                parts.append(frame)
            predictions = pd.concat(parts, ignore_index=True)
            predictions.to_csv(run / "predictions.csv",index=False,float_format="%.17g")
            stats = metrics(predictions)
            stats.to_csv(run / "metrics.csv",index=False)
            if cfg["experiment"]=="loeo":
                environment = data.ids.iloc[test].environment_id.unique().item()
                stats = stats.assign(environment_id=environment, records=len(test))
                loc = data.environment_locations
                if loc is not None and environment in loc.index:
                    stats = stats.assign(**loc.loc[environment].to_dict())
                stats.to_csv(run/"loeo_environment.csv",index=False)
            pd.DataFrame(history).to_csv(run / "training_log.csv",index=False)
            if ledger:
                pd.DataFrame(ledger).assign(outer_fold=fold, seed=seed, model_version=cfg["model_version"]).to_csv(
                    run/"oof_predictions.csv",index=False)
            rows = [(None,int(i),"outer_train") for i in train] + [(None,int(i),"outer_test") for i in test] + inner
            records = pd.DataFrame(rows,columns=["inner_fold","source_index","role"])
            records["sample_id"] = data.ids.iloc[records.source_index].sample_id.to_numpy()
            records.to_csv(run/"split_indices.tsv",sep="\t",index=False)
            state["output_sha256"] = {p.relative_to(run).as_posix():file_hash(p)
                                      for p in sorted(run.rglob("*")) if p.is_file()
                                      and p.name not in ("run_manifest.json",".lock") and p.suffix != ".tmp"}
            state.update(status="COMPLETED",end_time=utc())
        except (Exception, KeyboardInterrupt) as exc:
            state.update(status="BLOCKED" if isinstance(exc,Blocked) else "FAILED",
                         end_time=utc(),error=f"{type(exc).__name__}: {exc}")
            manifest(path,state)
            raise
        manifest(path,state)
    print(f"COMPLETED {variant} seed {seed} fold {fold}")
    return "COMPLETED"

def main(default="main", argv=None):
    args = arguments(default).parse_args(argv)
    try:
        cfg,variants = resolved(args)
        device = cfg["hardware"]["device"]
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.set_num_threads(cfg["hardware"]["cpu_threads"])
        traits = args.trait or cfg["trait_names"]
        output = Path(args.output_dir or cfg["output_dir"])
        if args.dataset == "demo" and "corrected" in output.resolve().parts:
            raise ValueError("Synthetic demos require a separate output root")
        plan = []
        data = demo_data() if args.dataset=="demo" else None
        data_error = None
        if data is None:
            try:
                data = load_data(cfg,args.data_dir)
            except Blocked as exc:
                data_error = str(exc)
        for seed in cfg["random_seeds"]:
            outer = cfg["outer_cv"]
            minimum = cfg["evaluation"]["loeo_minimum"] if outer["strategy"]=="loeo" else 1
            if data is not None and outer["strategy"]=="loeo" and not args.dry_run:
                report = data.ids.groupby("environment_id").size().rename("records").reset_index()
                report["eligible"] = report.records >= minimum
                report["reason"] = np.where(report.eligible, "", "below_predeclared_minimum")
                output.mkdir(parents=True,exist_ok=True)
                key = fingerprint([data.source, minimum])[:16]
                report.to_csv(output/f"loeo_environment_eligibility_{key}.csv",index=False)
            if data is None:
                folds = [(i,None,None) for i in range(outer["folds"])] if outer["strategy"]!="loeo" else [(None,None,None)]
            else:
                folds = list(splits(data,np.arange(len(data.ids)),outer["strategy"],outer["folds"],seed,minimum))
            for fold,train,test in folds:
                if args.fold is not None and fold is not None and fold != args.fold:
                    continue
                for variant in variants:
                    task = dict(dataset=cfg["dataset"], experiment=cfg["experiment"],variant=variant,
                                seed=seed, fold=fold, traits=traits, status="PENDING" if data is not None else "BLOCKED",
                                reason=data_error)
                    plan.append((task,train,test))
        if not plan:
            raise Blocked("Requested fold not in the eligible split plan")
        if args.dry_run:
            print(json.dumps([task for task,_,_ in plan],indent=2))
            return 0
        if args.dataset=="demo":
            print("SYNTHETIC DEMO — generated data")
        output.mkdir(parents=True,exist_ok=True)
        plan_id = fingerprint([cfg,variants,args.fold,traits])
        task_file = output / f"task_plan_{plan_id[:16]}.json"
        with task_lock(task_file.with_suffix(".lock")):
            for task,train,test in plan:
                if train is None:
                    continue
                task["status"] = "RUNNING"
                manifest(task_file,[r[0] for r in plan])
                task_cfg = cfg
                if re.fullmatch(r"k\d+",task["variant"]):
                    task_cfg = dict(cfg,model_parameters=dict(cfg["model_parameters"],
                                                               k_neighbors=int(task["variant"][1:])))
                try:
                    task["status"] = execute(data,task_cfg,task["variant"],task["seed"],task["fold"],train,test,
                                             output,args.resume,traits)
                except Exception as exc:
                    task.update(status="BLOCKED" if isinstance(exc,Blocked) else "FAILED",
                                reason=f"{type(exc).__name__}: {exc}")
                    print(task["reason"],file=sys.stderr)
            manifest(task_file,[r[0] for r in plan])
        if data_error:
            print(data_error,file=sys.stderr)
        return 0 if all(t["status"]=="COMPLETED" for t,_,_ in plan) else 2
    except (Blocked, ValueError, KeyError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}",file=sys.stderr)
        return 2
