"""Stage one local, non-paper G2F sanity cohort from explicitly joined source tables."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hypergep.data import file_hash, fingerprint

TRAIT = "1_Training_Trait_Data_2014_2021.csv"
META = "2_Training_Meta_Data_2014_2021.csv"
GENOTYPE = "5_Genotype_Data_All_Years.csv"
LAT = "Weather_Station_Latitude (in decimal numbers NOT DMS)"
LON = "Weather_Station_Longitude (in decimal numbers NOT DMS)"
TARGETS = {"Yield_Mg_ha":"Y", "Plant_Height_cm":"PH", "Grain_Moisture":"GM"}

def extract_markers(path, count):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        lines = list(itertools.islice(stream,count+1))
    sha.update(b"".join(lines))
    rows = list(csv.reader(line.decode("utf-8-sig") for line in lines))
    header = rows.pop(0)
    if len(rows)!=count or any(len(row)!=len(header) for row in rows):
        raise ValueError("Genotype excerpt is truncated or malformed")
    return header, rows, {"prefix_sha256":sha.hexdigest(),"prefix_bytes":sum(map(len,lines)),
                           "source_size_bytes":path.stat().st_size,"marker_rows":count}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("target",type=Path)
    p.add_argument("--source-license",default="Unverified for redistributing this local excerpt")
    args=p.parse_args()
    raw, root = args.source, args.target
    target = root/"g2f"
    if target.exists():
        raise SystemExit("Choose a new smoke asset directory")
    header, markers, digest = extract_markers(raw/GENOTYPE,64)
    hybrid_columns = {name[:-3]:i for i,name in enumerate(header) if name.endswith("_GT")}
    marker_ids = [row[2] for row in markers]
    if len(set(marker_ids))!=len(marker_ids):
        raise ValueError("Marker IDs are not unique")
    meta = pd.read_csv(raw/META,encoding="latin-1",usecols=["Env","Year",LAT,LON]).dropna()
    meta = meta[["Env","Year",LAT,LON]].drop_duplicates()
    meta = meta[~meta.Env.duplicated(keep=False)].set_index("Env").sort_index()
    trait_header = pd.read_csv(raw/TRAIT, encoding="latin-1", nrows=0).columns
    moisture = next((k for k in trait_header if k.lower().startswith("grain_moisture")),None)
    yield_column = next((k for k in trait_header if k.lower()=="yield_mg_ha"),None)
    if not moisture or not yield_column:
        raise ValueError("Missing explicit G2F yield or grain moisture column")
    target_names = {yield_column:"Y","Plant_Height_cm":"PH",moisture:"GM"}
    record_keys = ["Env","Field_Location","Experiment","Replicate","Plot","Hybrid"]
    phenotype = pd.read_csv(raw/TRAIT,encoding="latin-1",usecols=record_keys+list(target_names),dtype={k:str for k in record_keys})
    eligible = phenotype[phenotype.Env.isin(meta.index) & phenotype.Hybrid.isin(hybrid_columns)]
    eligible = eligible.dropna(subset=record_keys).copy()
    eligible["sample_id"] = [fingerprint(tuple(row))[:24] for row in eligible[record_keys].itertuples(index=False,name=None)]
    eligible = eligible[~eligible.sample_id.duplicated(keep=False)]
    eligible = eligible[eligible[list(target_names)].notna().any(axis=1)]
    selected = []
    for env, group in eligible.groupby("Env",sort=True):
        if len(group)>=16:
            selected.append(group.sort_values("sample_id").head(16))
        if len(selected)==10:
            break
    if len(selected)<6:
        raise ValueError("Insufficient usable environments")
    selected = pd.concat(selected).rename(columns=dict(target_names,Env="environment_id",Hybrid="genotype_id",
                                                       Field_Location="site_id"))
    phenotype_out = selected[["sample_id","genotype_id","environment_id","site_id","Y","PH","GM"]]
    chosen = sorted(set(selected.genotype_id))
    values = {"0/0":0,"0/1":1,"1/0":1,"1/1":2,"./.":np.nan,".":np.nan}
    genotypes = pd.DataFrame([[values[marker[hybrid_columns[hybrid]].replace("|","/")] for marker in markers]
                              for hybrid in chosen],index=pd.Index(chosen,name="genotype_id"),columns=marker_ids)
    env = meta.loc[sorted(set(selected.environment_id))].rename(columns={LAT:"latitude",LON:"longitude"})
    env.index.name = "environment_id"
    target.mkdir(parents=True)
    phenotype_out.to_csv(target/"phenotypes.csv",index=False)
    genotypes.to_csv(target/"genotypes.csv")
    env.to_csv(target/"environments.csv")
    inputs = dict(phenotypes="phenotypes.csv",genotypes="genotypes.csv",environments="environments.csv")
    manifest = {"dataset_version":"g2f-2014-2021-local-excerpt-smoke-1",
                "purpose":"real_smoke","origin":"https://www.genomes2fields.org/resources/",
                "license":args.source_license,"preprocessing":"raw","row_alignment_verified":True,
                "label_provenance":f"Observed columns {list(target_names)}. Blank labels retained.",
                "environment_availability":"Station coordinates and trial year from environment metadata",
                "sha256":{k:file_hash(target/v) for k,v in inputs.items()},
                "sources":{TRAIT:file_hash(raw/TRAIT),META:file_hash(raw/META),GENOTYPE:digest},
                "selection":"First ten lexicographic eligible environments, sixteen sample IDs per environment, first 64 source marker rows",
                "samples":len(phenotype_out),"environments":len(env),"genotypes":len(genotypes)}
    (target/"provenance.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    config_dir=Path(__file__).resolve().parents[1]/"configs"
    paper=yaml.safe_load((config_dir/"paper.yaml").read_text())
    smoke=yaml.safe_load((config_dir/"demo.yaml").read_text())
    smoke["datasets"]={"g2f":paper["datasets"]["g2f"]}
    for field in ("sample_id_column","genotype_id_column","environment_id_column","site_id_column"):
        smoke[field]=paper[field]
    smoke["output_dir"]="artifacts/real_smoke/results"
    (root/"smoke.yaml").write_text(yaml.safe_dump(smoke),encoding="utf-8")
    print(f"Local real-data smoke cohort: {len(phenotype_out)} records, {len(env)} environments, "
          f"{len(genotypes)} genotypes and {len(marker_ids)} markers.")

if __name__=="__main__":
    main()
