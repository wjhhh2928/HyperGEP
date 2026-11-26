"""Preserve a private source snapshot and hash adjacent research evidence."""
import csv
import hashlib
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
PARENT = ROOT.parent

def stream_digest(stream):
    checksum = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1048576), bytes()):
        checksum.update(chunk)
    return checksum.hexdigest()

def digest(path):
    with path.open("rb") as stream:
        return stream_digest(stream)

def sources():
    for p in ROOT.rglob("*"):
        if p.is_file() and PROJECT not in p.parents and not p.name.startswith("~$"):
            yield "input/" + p.relative_to(ROOT).as_posix(), p, True
    for name in ("source_materials", "负迁移结果2", "Ablation_Transfer_Analysis--负迁移结果",
                 "附件表", "split_traits-lolo的图", "Supplementary Information.pdf"):
        path = PARENT / name
        paths = [path] if path.is_file() else path.rglob("*")
        for p in paths:
            if p.is_file() and not p.name.startswith("~$"):
                yield "submission/" + p.relative_to(PARENT).as_posix(), p, p.suffix.lower() in {
                    ".py", ".csv", ".tsv", ".txt", ".log", ".xlsx", ".pdf"}
    base = PARENT.parent
    for pattern in ("Aggregated_Metrics_For_Plotting_run_k_*", "公开数据3000",
                    "生物可解释性", "train_3hgat*.py"):
        for path in base.glob(pattern):
            for p in ([path] if path.is_file() else path.rglob("*")):
                if p.is_file() and p.suffix.lower() in {
                    ".py", ".pth", ".pt", ".txt", ".csv", ".tsv", ".npy", ".xlsx", ".log", ".npz"}:
                    yield "research/" + p.relative_to(base).as_posix(), p, False

def main():
    if "--freeze" not in sys.argv:
        raise SystemExit("Explicit --freeze required")
    private = PROJECT / "artifacts" / "legacy"
    private.mkdir(parents=True, exist_ok=True)
    snapshot = private / "local_source_snapshot.zip"
    inventory = PROJECT / "reports" / "SOURCE_INVENTORY.tsv"
    if snapshot.exists() or inventory.exists():
        raise SystemExit("Existing snapshot preserved")
    records = []
    with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, file, copy in sorted(sources()):
            if copy:
                archive.write(file, arcname=name)
            records.append((name, file.stat().st_size, digest(file), "snapshot" if copy else "in_place"))
    with zipfile.ZipFile(snapshot) as archive:
        for name, _, checksum, mode in records:
            if mode == "snapshot":
                with archive.open(name) as stream:
                    if stream_digest(stream) != checksum:
                        raise RuntimeError(f"Source changed during snapshot: {name}")
    with inventory.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("source_relative_path", "size_bytes", "sha256", "preservation"))
        writer.writerows(records)
    (private / "SNAPSHOT_SHA256.txt").write_text(digest(snapshot) + "\n", encoding="utf-8")
    (private / "README.md").write_text(
        "# Private legacy evidence\n\nLocal originals and selected companion outputs. "
        "The inventory also hashes larger adjacent originals in place. Exclude from any public upload.\n",
        encoding="utf-8")
    print(len(records), "inventory entries", snapshot.stat().st_size, "archive bytes")

if __name__ == "__main__":
    main()
