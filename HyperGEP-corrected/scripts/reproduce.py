"""Recompute metrics and plots from a checked, frozen prediction bundle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from hypergep.data import file_hash

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("bundle",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    target=args.output_dir
    target.mkdir(parents=True,exist_ok=True)
    index=args.bundle/"bundle_manifest.json"
    status=target/"reproduction_status.json"
    state={"status":"BLOCKED","mode":"prediction-based metric reproduction and figure regeneration",
           "checkpoint_based_inference":"NOT_RUN","training_from_scratch":"NOT_RUN"}
    try:
        bundle=json.loads(index.read_text(encoding="utf-8"))
        if bundle["scope"]!="paper_candidate" or not bundle["files"]:
            raise ValueError("A paper-candidate bundle with checksums is required")
        root=args.bundle.resolve()
        for name,sha in bundle["files"].items():
            path=(root/name).resolve()
            if root not in path.parents or file_hash(path)!=sha:
                raise ValueError("Bundle file missing or changed: " + name)
        results=root/"results"
        for path in results.rglob("run_manifest.json"):
            name=path.relative_to(root).as_posix()
            if name not in bundle["files"]:
                raise ValueError("Unlisted run in reproduction bundle")
        subprocess.run([sys.executable,"-m","scripts.aggregate_results",str(results),
                        "--output-dir",str(target)],check=True)
        state["status"]="COMPLETED"
        state["bundle_sha256"]=file_hash(index)
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        state["reason"]=str(exc)
    status.write_text(json.dumps(state,indent=2),encoding="utf-8")
    print(state["status"],state.get("reason",""))
    return 0 if state["status"]=="COMPLETED" else 2

if __name__=="__main__":
    raise SystemExit(main())
