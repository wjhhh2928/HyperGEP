"""Summarize the supplied manuscript and historical tables without changing sources."""
import csv
from pathlib import Path
import re
import sys
import fitz
from openpyxl import load_workbook
from zipfile import is_zipfile

root = Path(sys.argv[1])
paper = next(root.glob("Decoupled*.pdf"))
with fitz.open(paper) as pdf:
    for i, page in enumerate(pdf):
        for line in page.get_text().splitlines():
            line = line.strip()
            if re.match(r"^(Fig(?:ure)?[. ]+\s*[2-6]|Methods|Data availability|Code availability)", line):
                print(i + 1, line)
    text = "\n".join(page.get_text() for page in pdf)
private = root / "HyperGEP-corrected" / "artifacts" / "legacy"
(private / "manuscript_search_text.txt").write_text(text, encoding="utf-8")
with (private / "workbook_evidence.tsv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(("workbook", "sheet", "rows", "columns", "first_nonempty_rows"))
    for p in sorted(root.glob("Supplementary Data*.xlsx")):
        if not is_zipfile(p):
            writer.writerow((p.name, "BLOCKED: container is not ZIP/OOXML", "", "", ""))
            print(p.name, "BLOCKED: container is not ZIP/OOXML")
            continue
        book = load_workbook(p, read_only=True, data_only=True)
        for ws in book:
            rows = [[str(v) for v in row if v is not None][:12] for row in ws.iter_rows(
                min_row=1, max_row=min(5, ws.max_row), max_col=min(ws.max_column, 20), values_only=True)]
            writer.writerow((p.name, ws.title, ws.max_row, ws.max_column, rows))
        book.close()
        print(p.name, len(book.sheetnames), book.sheetnames[:16])
