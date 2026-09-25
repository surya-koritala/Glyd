#!/usr/bin/env python3
"""Fill docs/savings.html's rows from scripts/calc_rows.sh's output and
the terabyte gate's result: ratios (raw / stored) per codec and Glyd
level, decompress CPU-seconds, a note per row.
    scripts/calc_page.py rows.tsv gate.txt "measured-on sentence" > docs/savings.html
The page's template is the current docs/savings.html (its `/*ROWS*/`
object and `MEASURED_ON` sentence are replaced)."""
import json, re, sys

rows_tsv, gate_txt, measured_on = sys.argv[1], sys.argv[2], sys.argv[3]
raw, size, cpu = {}, {}, {}
for line in open(rows_tsv):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4 or not f[2].isdigit():
        continue
    row, codec = f[0], f[1]
    raw[row] = int(f[2])
    size[(row, codec)] = int(f[3])
    if len(f) > 4 and f[4].strip():
        cpu[(row, codec)] = float(f[4])

CODECS = ["lz4", "gzip6", "zstd3", "zstd19", "max", "maxrec", "ultra", "ultrarec"]

def entry(rows, note):
    r = sum(raw[x] for x in rows)
    ratio, secs = {"none": 1.0}, {}
    for c in CODECS:
        if all((x, c) in size for x in rows):
            ratio[c] = round(r / sum(size[(x, c)] for x in rows), 3)
        else:
            ratio[c] = None
        if all((x, c) in cpu for x in rows):
            secs[c] = round(sum(cpu[(x, c)] for x in rows), 2)
    return {"ratio": ratio, "cpu": secs or None, "note": note}

R = {}
R["corpus"] = entry(["logs", "sql", "json", "parquet"], "18 files, 8.7 GB: access and pageview logs 1.3 GB, GitHub events 2.6 GB, Wikipedia SQL dumps 3.9 GB, two NYC TLC Parquet files 0.9 GB.")
R["logs"] = entry(["logs"], "NASA and ClarkNet access logs, three hours of Wikipedia pageviews (1.3 GB).")
R["sql"] = entry(["sql"], "MySQL dumps of English and Simple English Wikipedia tables (3.9 GB).")
R["json"] = entry(["json"], "Three hours of GitHub Archive (2.6 GB): not record-shaped, so -r hands them to the plain level; hashes and free text set the floor.")
R["telemetry_csv"] = entry(["telemetry_csv"], "Three CSVs, 128 MiB each: Alibaba cluster machine usage, NOAA daily weather, NYC taxi trips.")
R["telemetry_json"] = entry(["telemetry_json"], "The cluster trace and the weather rows as JSON objects, one per line, 128 MiB each.")
R["parquet"] = entry(["parquet"], "NYC TLC's for-hire trips, two months as the city publishes them (zstd pages, 0.9 GB): every page is written back byte for byte by Glyd's port of zstd and its values modeled; zstd on the file gains almost nothing.")
R["parquet_snappy"] = entry(["parquet_snappy"], "A for-hire month and a yellow-taxi month written by pyarrow with snappy pages (0.6 GB): every page written back byte for byte by Glyd's port of snappy, its values modeled.")
v = entry(["versions"], "A Simple English Wikipedia page table a month on (108 MB), against the last: zstd's figures are its --patch-from, Glyd's are --base (-r levels as the plain ones); LZ4 and gzip have no delta mode, so each dump alone. Kernel and image pairs give 200-700x.")
v["ratio"]["maxrec"], v["ratio"]["ultrarec"], v["cpu"] = v["ratio"]["max"], v["ratio"]["ultra"], None
R["versions"] = v
# The store's row from the gate: raw bytes, zstd -3's bytes, the store's bytes.
g = open(gate_txt).read()
corpus = int(re.search(r"corpus: \d+ objects, (\d+) bytes", g).group(1))
zstd = int(re.search(r"zstd-3 put: (\d+) bytes stored", g).group(1))
store = int(re.search(r"B raw, (\d+) B on disk", g).group(1))
R["store"] = {"ratio": {"none": 1.0, "lz4": None, "gzip6": None, "zstd3": round(corpus / zstd, 3), "zstd19": None,
                        "max": round(corpus / store, 3), "maxrec": round(corpus / store, 3), "ultra": None, "ultrarec": None},
              "cpu": None,
              "note": "The gate: 1,192 public objects, 1.18 TB (400 Linux point releases, every hour of GitHub events in January 2024, five English Wikipedia dumps' tables, six Ubuntu images), put through glyd-store into S3 and every object read back byte-exact; zstd -3 is each object alone. The store keeps a version as a delta of the stored object it most resembles."}

page = open("docs/savings.html").read()
page = re.sub(r"/\*ROWS\*/\{.*?\};", "/*ROWS*/" + json.dumps(R, indent=None, separators=(",", ":")) + ";", page, flags=re.S)
page = page.replace("MEASURED_ON", measured_on)
sys.stdout.write(page)
