"""Check package structure and gate external publication on scientific evidence."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypergep.cli import EXPERIMENTS, ROOT, code_identity
from hypergep.data import file_hash

REQUIRED = ("README.md","configs/paper.yaml","configs/demo.yaml","run","environment/postInstall",
            "docs/DATA.md","docs/REPRODUCIBILITY.md","docs/CODE_OCEAN.md","reports/RELEASE_BLOCKERS.md",
            "reports/PAPER_EXPERIMENT_MAPPING.tsv","reports/TEST_REPORT.md","CITATION.cff",
            ".github/workflows/cpu.yml")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--structure-only",action="store_true")
    p.add_argument("--manifest",type=Path)
    args=p.parse_args()
    errors=[f"Missing {name}" for name in REQUIRED if not (ROOT/name).is_file()]
    for experiment in EXPERIMENTS:
        importlib.import_module("experiments.run_"+experiment)
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    for module in set(re.findall(r"python -m ((?:experiments|scripts)\.[a-z_]+)",readme)):
        result=subprocess.run([sys.executable,"-m",module,"--help"],cwd=ROOT,capture_output=True)
        if result.returncode:
            errors.append(f"Invalid documented CLI {module}")
    roots=[ROOT/name for name in ("hypergep","experiments","configs","docs","environment","scripts","tests")]
    files=sorted([ROOT/name for name in REQUIRED if (ROOT/name).is_file()] +
                 [file for top in roots for file in top.rglob("*") if file.is_file()
                  and "__pycache__" not in file.parts])
    for file in files:
        if file.suffix.lower() in (".pth",".pt",".npz",".xlsx",".pem",".key",".zip",".rar"):
            errors.append("Private binary in release candidate "+file.name)
        if file.suffix in (".py",".yaml",".md",".txt"):
            value=file.read_text(encoding="utf-8")
            if re.search(r'(?:[A-Za-z]:\\(?:Users|111科研)\\|/mnt/' r'mydisk/|sk-(?:proj-)?[A-Za-z0-9]{20,})',value):
                errors.append("Private absolute path or credential pattern in "+file.name)
    found=set()
    mapping=ROOT/"reports/PAPER_EXPERIMENT_MAPPING.tsv"
    if mapping.is_file():
        with mapping.open(encoding="utf-8",newline="") as stream:
            found={row["figure"] for row in csv.DictReader(stream,delimiter="\t")}
    if not {"2","3","4","5","6"}.issubset(found):
        errors.append("Missing paper figure mapping")
    blockers=[]
    if not args.structure_only:
        blockers=[line.strip()[2:] for line in (ROOT/"reports/RELEASE_BLOCKERS.md").read_text(
            encoding="utf-8").splitlines() if line.startswith("- [OPEN] ")]
        test_log = ROOT/"reports/TEST_REPORT.md"
        if not test_log.is_file() or "9 passed" not in test_log.read_text(encoding="utf-8"):
            blockers.append("Record evidence of the critical CPU test suite")
        if not (ROOT/"LICENSE").is_file():
            blockers.append("An approved license file is required")
    commit, code_hash = code_identity()
    state={"git_commit":commit,"code_hash":code_hash,"status":"BLOCKED" if errors or blockers else "PASS",
           "scope":"structure only" if args.structure_only else "formal release",
           "structure_errors":errors,"blockers":blockers,
           "sha256":{file.relative_to(ROOT).as_posix():file_hash(file) for file in files}}
    if args.manifest:
        args.manifest.write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({key:val for key,val in state.items() if key!="sha256"},ensure_ascii=False,indent=2))
    return 2 if state["status"]=="BLOCKED" else 0

if __name__=="__main__":
    raise SystemExit(main())
