"""Summarize frozen fold attribution archives for Figure 6 sensitivity analyses."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from hypergep.cli import completed_valid

def stable_rank(vector):
    return rankdata(vector,method="average")

def spearman(x,y):
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:
        return np.nan
    return float(np.corrcoef(stable_rank(x),stable_rank(y))[0,1])

def top_overlap(x,y,k):
    return len(set(np.argsort(-x,kind="stable")[:k]) & set(np.argsort(-y,kind="stable")[:k]))

def input_records(root, demo=False):
    outputs=[]
    for file in sorted(root.rglob("heldout_snp_gradients.npz")):
        record=json.loads((file.parent/"run_manifest.json").read_text(encoding="utf-8"))
        if record["status"]!="COMPLETED" or record["synthetic"]!=demo or record.get("scope")=="real_smoke":
            continue
        if not completed_valid(file.parent,record):
            raise ValueError("An attribution run has changed since completion")
        outputs.append((record,file))
    if not outputs:
        raise ValueError("No matches for the requested attribution scope")
    for column in ("dataset","data_hash","evaluation_protocol","random_seed","model_version"):
        if len({state[column] for state,_ in outputs})!=1:
            raise ValueError("Select one attribution dataset version, protocol and seed per analysis")
    return outputs

def summarize(outputs, out, markers_path, replicates, seed):
    arrays, refs, sections = [], [], []
    for state,file in outputs:
        with np.load(file,allow_pickle=False) as source:
            traits,markers=source["traits"],source["markers"]
            if arrays and (list(traits)!=list(refs[0][0]) or list(markers)!=list(refs[0][1])):
                raise ValueError("Marker and trait ordering differs between folds")
            arrays.append(source["gradients"])
            refs.append((traits,markers))
            section=pd.DataFrame({k:source[k] for k in ("sample_id","genotype_id","environment_id")})
            sections.append(section.assign(fold=state["fold"]))
    data=np.concatenate(arrays)
    rows=pd.concat(sections,ignore_index=True)
    if rows.sample_id.duplicated().any():
        raise ValueError("One out-of-fold attribution is required per sample")
    # Equal weight per genotype. Retain naturally observed environments within genotype.
    genotypes=list(rows.groupby("genotype_id",sort=True).indices.values())
    reference=np.mean([data[idx].mean(0,dtype=np.float64) for idx in genotypes],axis=0)
    matrix=pd.DataFrame(reference.T,index=pd.Index(markers,name="marker_id"),columns=traits)
    matrix.to_csv(out/"genotype_balanced_marker_attribution.csv")
    order=np.argsort(rows.sample_id.to_numpy(),kind="stable")[:500]
    composition=data[order].mean(0,dtype=np.float64)
    k=min(100,len(markers))
    pd.DataFrame([dict(trait=trait,n_balanced=len(rows),n_prefix=len(order),top_k=k,
                       rank_spearman=spearman(reference[i],composition[i]),
                       overlap=top_overlap(reference[i],composition[i],k))
                  for i,trait in enumerate(traits)]).to_csv(out/"fig6gh_composition.csv",index=False)
    matrix.corr(method="spearman").to_csv(out/"fig6a_marker_spearman.csv")
    alters=[]
    rng=np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled=np.mean([data[rng.choice(idx,len(idx),replace=True)].mean(0,dtype=np.float64)
                         for idx in genotypes],axis=0)
        for j,trait in enumerate(traits):
            alters.append(dict(replicate=replicate,trait=trait,
                               rank_spearman=spearman(reference[j],sampled[j]),top_k=k,
                               overlap_fraction=top_overlap(reference[j],sampled[j],k)/k))
    pd.DataFrame(alters).to_csv(out/"fig6de_conditional_bootstrap.csv",index=False)
    tv=[]
    for (fold,genotype),idx in rows.groupby(["fold","genotype_id"]).indices.items():
        frame=rows.iloc[idx]
        means=[data[idx[env_idx]].mean(0,dtype=np.float64)
               for env_idx in frame.groupby("environment_id").indices.values()]
        for j,trait in enumerate(traits):
            pairs=[np.abs(a[j]/a[j].sum()-b[j]/b[j].sum()).sum()/2
                   for a,b in itertools.combinations(means,2) if a[j].sum()>0 and b[j].sum()>0]
            if pairs:
                tv.append(dict(fold=fold,genotype_id=genotype,trait=trait,
                               env_pairs=len(pairs),mean_total_variation=np.mean(pairs)))
    pd.DataFrame(tv,columns=["fold","genotype_id","trait","env_pairs","mean_total_variation"]).to_csv(
        out/"fig6c_within_fold_environment_tv.csv",index=False)
    if markers_path:
        coords=pd.read_csv(markers_path,dtype={"marker_id":str,"chromosome":str,"reference_build":str})
        if coords.marker_id.duplicated().any() or coords.reference_build.isna().any() or coords.reference_build.nunique()!=1:
            raise ValueError("Coordinates require unique marker IDs and one declared reference build")
        coords=coords.set_index("marker_id").reindex(markers)
        if coords[["chromosome","position"]].isna().any().any() or (coords.position<1).any():
            raise ValueError("Missing coordinates or invalid 1-based marker positions")
        panel=matrix.join(coords)
        panel["window_1Mb"]=(panel.position.astype("int64")-1)//1_000_000
        groups=panel.groupby(["chromosome","window_1Mb"])
        regional=groups[list(traits)].mean()
        n=groups.size()
        regional.assign(marker_count=n,reference_build=coords.reference_build.iloc[0]).to_csv(out/"fig6b_regions.csv")
        regional.corr(method="spearman").to_csv(out/"fig6b_region_spearman.csv")
        sensitivity=[]
        for floor in (1,3,5,10,20):
            usable=regional[n>=floor]
            for a,b in itertools.combinations(traits,2):
                sensitivity.append(dict(min_snps=floor,windows=len(usable),trait_a=a,trait_b=b,
                                        spearman=spearman(usable[a].to_numpy(),usable[b].to_numpy())))
        pd.DataFrame(sensitivity).to_csv(out/"fig6f_window_density.csv",index=False)
    for path in sorted(out.glob("*_spearman.csv")):
        values=pd.read_csv(path,index_col=0)
        fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
        chart=ax.imshow(values,vmin=-1,vmax=1,cmap="coolwarm")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.columns,rotation=45,ha="right")
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index)
        fig.colorbar(chart,ax=ax,label="Spearman correlation")
        fig.savefig(path.with_suffix(".png"),dpi=160)
        plt.close(fig)
    rows.to_csv(out/"attribution_records.tsv",sep="\t",index=False)
    (out/"analysis_manifest.json").write_text(json.dumps({
        "runs":[state["run_id"] for state,_ in outputs],
        "seed":seed,"replicates":replicates,"top_k":k,"n_records":len(rows),"genotypes":len(genotypes),
        "bootstrap":"within-genotype records conditional on frozen folds",
        "composition_prefix":"first sample IDs in canonical order",
        "environment_tv":"within the same frozen outer fold and genotype",
        "regional_status":"executed" if markers_path else "BLOCKED: marker coordinates required"},indent=2),
        encoding="utf-8")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--markers",type=Path)
    parser.add_argument("--replicates",type=int,default=50)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--demo",action="store_true")
    args=parser.parse_args()
    if args.replicates<1:
        parser.error("--replicates must be positive")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summarize(input_records(args.root,args.demo),args.output_dir,args.markers,args.replicates,args.seed)

if __name__=="__main__":
    main()
