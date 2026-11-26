"""Hash additional source candidates discovered after the snapshot."""
import argparse
import csv
from pathlib import Path

from hypergep.data import file_hash

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("research_root",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    base=args.research_root
    entries=set()
    for path in ("train_data/1_Training_Trait_Data_2014_2021.csv",
                 "train_data/2_Training_Meta_Data_2014_2021.csv",
                 "train_data/5_Genotype_Data_All_Years.csv","train_data/env_new2_3000.csv"):
        entries.add(base/path)
    entries.update((base/"baseline_cv_results").glob("*"))
    kit=base/"HyperGEP_四个阶段"
    for folder in kit.glob("*"):
        for p in folder.rglob("*"):
            if p.is_file() and (p.suffix.lower() in {".py",".json",".md",".log"} or "README" in p.name):
                entries.add(p)
    with args.output.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream,delimiter="\t")
        writer.writerow(("source_relative_path","size_bytes","sha256","preservation"))
        for p in sorted(entries):
            if p.is_file():
                writer.writerow((p.relative_to(base).as_posix(),p.stat().st_size,file_hash(p),"in_place"))

if __name__=="__main__":
    main()
