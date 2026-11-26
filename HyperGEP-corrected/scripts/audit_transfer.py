"""Recompute historical transfer summaries from the supplied legacy fold CSVs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    outputs=[]
    for file in sorted(args.input.rglob("Raw_Fold_Results.csv")):
        rows=pd.read_csv(file)
        dataset="g2f" if "g2f" in file.parent.name.lower() else "maizegep"
        observed=rows.pivot(index=["Fold","Trait"],columns="Architecture",values="PCC")
        expected=pd.read_csv(file.with_name("Quantitative_Transfer_Metrics.csv")).set_index("Trait")
        means=observed.groupby("Trait").mean()
        means["Negative_Transfer (Coupled-Ind)"]=means["1_Early_Coupled"]-means["2_Independent"]
        means["Synergy_Gain (HyperGEP-Ind)"]=means["3_HyperGEP"]-means["2_Independent"]
        means=means.reindex(index=expected.index,columns=expected.columns)
        delta=(means-expected).abs()
        for name in means.columns:
            outputs.append(dict(dataset=dataset,metric=name,traits=len(means),
                                folds=rows.Fold.nunique(),valid_fold_values=int(rows.PCC.notna().sum()),
                                max_abs_error=delta[name].max(),
                                status="VERIFIED" if np.allclose(means[name],expected[name],atol=1e-12,
                                                                 rtol=0,equal_nan=False) else "DIFFERENT"))
    if not outputs:
        raise ValueError("No historical transfer CSVs found")
    pd.DataFrame(outputs).to_csv(args.output,index=False,sep="\t")

if __name__=="__main__":
    main()
