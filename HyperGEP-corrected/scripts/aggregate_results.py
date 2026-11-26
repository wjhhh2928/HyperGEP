"""Verify run evidence and aggregate held-out predictions, matched transfer and plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml

from hypergep.cli import completed_valid
from hypergep.data import fingerprint
from hypergep.train import metrics

PAIR = ["dataset","dataset_version","data_hash","evaluation_protocol","model_version",
        "seed","fold","trait","test_indices_hash"]

def read_runs(root, include_demo):
    frames = []
    for path in sorted(root.rglob("run_manifest.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] != "COMPLETED" or record["synthetic"] != include_demo or record.get("scope") == "real_smoke":
            continue
        if not completed_valid(path.parent,record):
            raise ValueError("A completed result failed hash verification: "+path.parent.name)
        frame = pd.read_csv(path.parent/"predictions.csv",dtype={
            "sample_id":str,"genotype_id":str,"environment_id":str})
        config = yaml.safe_load((path.parent/"config_resolved.yaml").read_text(encoding="utf-8"))
        for key in ("variant", "seed", "fold", "output_traits"):
            config.pop(key, None)
        frames.append(frame.assign(run_id=record["run_id"], code_hash=record["code_hash"],
                                   fit_spec=fingerprint(config)))
    if not frames:
        raise ValueError("No completed runs for this result category")
    df = pd.concat(frames,ignore_index=True)
    key = PAIR+["experiment","model","sample_id"]
    if df.duplicated(key).any():
        raise ValueError("Multiple result candidates share a task slot. Select one frozen input set.")
    return df

def paired_transfer(df):
    source = df[df.experiment.eq("negative_transfer")]
    keys, variants = PAIR+["sample_id"], ["STL","MTL","Full_Model"]
    result = []
    for _,group in source.groupby(PAIR+["code_hash","fit_spec"],dropna=False):
        if set(group.model) != set(variants):
            continue
        reference = group[group.model.eq("STL")].set_index("sample_id").sort_index()
        for model in variants[1:]:
            target = group[group.model.eq(model)].set_index("sample_id").sort_index()
            if not reference.index.equals(target.index) or not np.array_equal(
                reference.label_valid.to_numpy(),target.label_valid.to_numpy()
            ) or not np.allclose(reference.true_value,target.true_value,equal_nan=True):
                raise ValueError("Transfer comparisons require identical held-out rows and labels")
        computed = metrics(group).set_index("model")
        row = {key:group.iloc[0][key] for key in keys if key!="sample_id"}
        row["code_hash"]=group.code_hash.iloc[0]
        row["fit_spec"]=group.fit_spec.iloc[0]
        row["valid_sample_count"] = int(computed.loc["STL","valid_sample_count"])
        for name,report in (("STL","STL"),("MTL","MTL"),("Full_Model","HyperGEP")):
            for metric in ("PCC","MAE","RMSE"):
                row[f"{report}_{metric}"] = computed.loc[name,metric]
            if report!="STL":
                row[f"Delta_PCC_{report}"] = row[f"{report}_PCC"]-row["STL_PCC"]
        result.append(row)
    return pd.DataFrame(result)

def residual_statistics(predictions):
    valid = predictions[predictions.label_valid]
    keys = PAIR + ["experiment","model"]
    rows = []
    rng = np.random.default_rng(4211)
    for index,group in valid.groupby(keys,dropna=False):
        dif = (group.predicted_value-group.true_value).to_numpy()
        value = float(wilcoxon(dif).pvalue) if len(dif)>1 and np.any(dif) else np.nan
        clusters = pd.DataFrame({"genotype":group.genotype_id.to_numpy(),"error":dif}).groupby("genotype")
        stats = clusters.error.agg(["sum","count"]).to_numpy()
        ci = [np.nan,np.nan]
        if len(stats)>=2:
            draw = rng.integers(len(stats),size=(1000,len(stats)))
            totals=stats[draw].sum(axis=1)
            ci=np.quantile(totals[:,0]/totals[:,1],[.025,.975])
        rows.append(dict(zip(keys,index), n=len(dif), genotype_clusters=len(stats), signed_mean_error=dif.mean(),
                         cluster_bootstrap_low=ci[0], cluster_bootstrap_high=ci[1], paired_wilcoxon_p=value,
                         inference="Conditional within-fold genotype bootstrap. Wilcoxon requires independent symmetric pairs."))
    return pd.DataFrame(rows)

def plots(folds, transfer, predictions, target, demo=False):
    for keys, group in folds.groupby(["dataset","data_hash","experiment","evaluation_protocol"]):
        dataset, checksum, experiment, protocol = keys
        fig, axes = plt.subplots(3,1,figsize=(max(8,group.trait.nunique()),10),constrained_layout=True)
        for metric, axis in zip(("PCC","MAE","RMSE"),axes):
            table = group.groupby(["trait","model"])[metric].mean().unstack()
            table.plot.bar(ax=axis,legend=metric=="PCC")
            axis.set_ylabel(metric)
        file_tag = f"{dataset}_{checksum[:8]}_{experiment}_{protocol.replace('/','-')}"
        if demo:
            fig.suptitle("SYNTHETIC DEMO")
        fig.savefig(target/f"{file_tag}.png",dpi=160)
        plt.close(fig)
    if not transfer.empty:
        for (dataset,checksum,protocol),group in transfer.groupby(["dataset","data_hash","evaluation_protocol"]):
            columns = ["Delta_PCC_MTL","Delta_PCC_HyperGEP"]
            summary = group.groupby("trait")[columns].agg(["mean","std","count"])
            tag = f"{dataset}_{checksum[:8]}_{protocol.replace('/','-')}"
            summary.to_csv(target/f"{tag}_transfer_summary.csv")
            means = summary.xs("mean",axis=1,level=1)
            means.apply(lambda col: pd.Series({"positive":(col>0).sum(),"negative":(col<0).sum(),
                                              "zero":(col==0).sum(),"undefined":col.isna().sum()})).to_csv(
                                                  target/f"{tag}_transfer_sign_counts.csv")
            ax = means.plot.bar(figsize=(10,5))
            ax.axhline(0,color="black",linewidth=.8)
            if demo:
                ax.set_title("SYNTHETIC DEMO")
            ax.figure.tight_layout()
            ax.figure.savefig(target/f"{tag}_transfer.png",dpi=160)
            plt.close(ax.figure)
    valid = predictions[predictions.label_valid & predictions.experiment.eq("main")]
    for (dataset,protocol,checksum),group in valid.groupby(["dataset","evaluation_protocol","data_hash"]):
        residuals = (group.predicted_value-group.true_value)
        fig,axis=plt.subplots(figsize=(8,4),constrained_layout=True)
        for trait,rows in group.groupby("trait").groups.items():
            axis.hist(residuals.loc[rows],bins=20,alpha=.4,label=trait)
        axis.legend()
        axis.set_xlabel("Predicted minus observed (trait units)")
        if demo:
            axis.set_title("SYNTHETIC DEMO")
        fig.savefig(target/f"{dataset}_{checksum[:8]}_{protocol.replace('/','-')}_residuals.png",dpi=160)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--demo",action="store_true")
    args = parser.parse_args()
    predictions = read_runs(args.root,args.demo)
    target = args.output_dir
    target.mkdir(parents=True,exist_ok=True)
    parts = []
    for (_,data_hash), group in predictions.groupby(["run_id","data_hash"]):
        parts.append(metrics(group).assign(data_hash=data_hash,dataset_version=group.dataset_version.iloc[0]))
    folds = pd.concat(parts,ignore_index=True)
    folds.to_csv(target/"fold_metrics.csv",index=False)
    keys = ["dataset","dataset_version","data_hash","evaluation_protocol","experiment","model","model_version","trait"]
    folds.groupby(keys)[["PCC","MAE","RMSE","valid_sample_count"]].agg(["mean","std","sem","count"]).to_csv(
        target/"trait_summary.csv")
    transfer = paired_transfer(predictions)
    if not transfer.empty:
        transfer.to_csv(target/"paired_transfer.csv",index=False)
    residual_statistics(predictions).to_csv(target/"residual_statistics.csv",index=False)
    predictions = predictions.assign(residual=np.where(predictions.label_valid,
                                      predictions.predicted_value-predictions.true_value,np.nan))
    predictions.to_csv(target/"heldout_predictions.csv",index=False)
    plots(folds,transfer,predictions,target,demo=args.demo)

if __name__=="__main__":
    main()
