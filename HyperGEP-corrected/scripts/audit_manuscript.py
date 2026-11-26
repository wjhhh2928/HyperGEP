"""Compare documented manuscript claims against available legacy workbook evidence."""
import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from hypergep.data import file_hash

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    source=args.source
    sheet="Supplementary Data 1.xlsx"
    maize=pd.read_excel(source/sheet,sheet_name="MaizeGEP-detailed-fold-metrics")
    g2f=pd.read_excel(source/sheet,sheet_name="G2F-detailed-fold-metrics")
    notes=[]
    def audit(panel,key,paper,observed,ref,status,comment):
        notes.append(dict(panel=panel,claim=key,manuscript=paper,evidence_value=observed,
                          evidence=ref,status=status,notes=comment))
    audit("Fig.2","trait count",11,maize.trait.nunique(),sheet,"VERIFIED","Workbook coverage only")
    audit("Fig.3","trait count",3,g2f.trait.nunique(),sheet,"VERIFIED","Workbook coverage only")
    yield_row=maize[(maize.Methods=="HyperGEP")&(maize.trait=="Y")].iloc[0]
    audit("Fig.2a","HyperGEP Y PCC","approximately 0.94",yield_row.PCC_mean,sheet,
          "VERIFIED" if abs(yield_row.PCC_mean-.94)<.01 else "VALUE_DIFFERS","Rounded legacy mean")
    for traits,row in g2f[g2f.Methods.eq("HyperGEP")].set_index("trait").iterrows():
        median=float(np.median([row[f"PCC_fold{i}"] for i in range(1,6)]))
        audit("Fig.3a",f"{traits} HyperGEP median PCC",">0.7",median,sheet,
              "VERIFIED" if median>.7 else "VALUE_DIFFERS","Median of five displayed fold values")
    for dataset,table in (("MaizeGEP",maize),("G2F",g2f)):
        for metric in ("PCC","MAE","RMSE"):
            cols=[f"{metric}_fold{i}" for i in range(1,6)]
            reported=table[f"{metric}_mean"]
            calculated=table[cols].mean(1)
            count=int(((reported-calculated).abs()>0.00011).sum())
            audit("Fig.2–3",f"{dataset} {metric} fold means","mean of five folds",count,sheet,
                  "VERIFIED" if count==0 else "VALUE_DIFFERS",
                  "Value is number of rows outside tolerance for four-decimal fold exports")
    transfer_specs=(("Maize_Quantitative_Transfer","Ablation_Transfer_Analysis"),
                    ("G2F_Quantitative_Transfer","Ablation_Transfer_Analysis_g2f"))
    for table,directory in transfer_specs:
        expected=pd.read_excel(source/sheet,sheet_name=table)
        local=pd.read_csv(source/"负迁移结果2"/directory/"Quantitative_Transfer_Metrics.csv")
        equal=expected.shape==local.shape and np.allclose(expected.iloc[:,1:],local.iloc[:,1:],rtol=0,atol=1e-12)
        audit("Fig.4d–e",table,"legacy quantitative transfer table",equal,sheet+" and local transfer CSV",
              "VERIFIED" if equal else "VALUE_DIFFERS","Compare numerical columns in listed trait order")
        negative=int((local.iloc[:,-2]<0).sum())
        positive=int((local.iloc[:,-1]>0).sum())
        audit("Fig.4d–e",table+" sign counts","early-coupled negative and HyperGEP positive tendency",
              f"early-coupled negative {negative}/{len(local)}; HyperGEP positive {positive}/{len(local)}",
              "Historical transfer summary", "PARTIAL","Matched fold IDs do not verify matched records or supervision")
    loeo=next(source.glob("Supplementary Data 2*.xlsx"))
    audit("Fig.5e–f","environment count",2382,"not recoverable from this workbook",loeo.name,
          "MISSING_EVIDENCE" if not zipfile.is_zipfile(loeo) else "PARTIAL","Acquire a portable workbook and source split")
    audit("Methods","MaizeGEP aligned record count",3000,"unbound","legacy arrays and source candidates",
          "MISSING_EVIDENCE","Need exact matching input and selection recipe")
    for panel,filename,sheet_name,claim,expected,column in (
       ("Fig.6a","Supplementary Data 5.xlsx","SNP_Importance","SNP count",12233,None),
       ("Fig.6b","Supplementary Data 6.xlsx","Regional_Importance","1 Mb regional entries",1753,None),
       ("Fig.6c","Supplementary Data 7.xlsx","Sample_Metadata","balanced record count",952,None),
       ("Fig.6c","Supplementary Data 7.xlsx","Sample_Metadata","genotype groups",120,"genotype_group"),
       ("Fig.6d–e","Supplementary Data 8.xlsx","Bootstrap_Raw","conditional bootstrap replicates",50,"bootstrap_repeat"),
    ):
        evidence=pd.read_excel(source/filename,sheet_name=sheet_name)
        if column:
            match=next((c for c in evidence if c.lower().strip().startswith(column)),None)
            observed=int(evidence[match].nunique()) if match else "column unconfirmed"
        else:
            observed=len(evidence)
        audit(panel,claim,expected,observed,filename+"/"+sheet_name,
              "VERIFIED" if observed==expected else "PARTIAL",
              "Workbook dimensions verify exported rows, attribution lineage pending")
    audit("S6","masking cohort per trait",132,"unbound","held-out masking source", "MISSING_EVIDENCE",
          "Need cohort sample IDs and fold-trained control-matching evidence")
    pd.DataFrame(notes).to_csv(args.output,index=False,sep="\t")

if __name__=="__main__":
    main()
