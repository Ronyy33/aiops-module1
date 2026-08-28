import csv, hashlib, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
out = Path("data/file_index.csv")

rows = []
for p in sorted(root.rglob("*")):
    if not p.is_file():
        continue
    if p.name.startswith(".") or "__MACOSX" in p.parts:
        continue
    rows.append([p.name, p.parent.name, str(p.relative_to(root))])

rows.sort(key=lambda r: r[2])

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label", "relpath"])
    w.writerows(rows)

print(f"scanned    : {root}")
print(f"data rows  : {len(rows)}")
print(f"file lines : {len(rows) + 1}   (rows + 1 header line)")
print(f"md5        : {hashlib.md5(out.read_bytes()).hexdigest()}")
