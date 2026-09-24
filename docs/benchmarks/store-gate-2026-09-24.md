# The store's gate: a terabyte bucket, 2026-09-24

One instance next to the bucket (`scripts/gate_aws.sh`: im4gn.4xlarge,
Neoverse-N1, 16 vCPU, 61 GB, 7.5 TB NVMe, us-east-1), glyd-store built
from commit 291a88b (the store as released in v0.14.7), objects in S3
over the store's own client. The corpus (`scripts/gate_corpus.sh`), as
on 2026-09-22: 1,192 public objects, 1,184,268,065,453 bytes, as object
storage holds them — 400 Linux point releases (6.6.1–150, 6.1.1–150,
5.15.1–100), every hour of GitHub events in January 2024 (744), five
English Wikipedia dumps' page, page_props and categorylinks tables,
nine Simple English dumps' three tables, six Ubuntu 24.04 cloud images.
Put in name order, every object read back by its own process and
compared byte for byte, then the whole bucket restored by one process
and compared again (`scripts/gate_run.sh`; raw results in
`benchmarks/gate/im4gn.4xlarge/`; the 2026-09-22 runs' at commit
b85eff7). The rebuild-from-the-bucket phase (`GATE_REBUILD=1`) was not
run this time; its code is unchanged since the 2026-09-22 run, where
the rebuilt index was identical and every object verified.

| | Bytes | Against raw | Against zstd -3 |
| :--- | ---: | ---: | ---: |
| Raw | 1,184.3 GB | | |
| zstd -3, each object alone (16 threads, 547 MB/s) | 153.5 GB | 7.7× | |
| **Glyd store** | **46.3 GB** | **25.6×** | **3.32× fewer bytes** |

| Family | Objects | Raw | Stored | Against raw | Deltas |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Linux 5.15 point releases | 100 | 113.8 GB | 0.40 GB | 286× | 99 |
| Linux 6.1 | 150 | 204.0 GB | 1.32 GB | 155× | 150 |
| Linux 6.6 | 150 | 213.2 GB | 1.84 GB | 116× | 150 |
| English Wikipedia tables | 15 | 165.6 GB | 8.51 GB | 19.5× | 9 |
| Simple English Wikipedia tables | 27 | 3.2 GB | 0.08 GB | 43× | 24 |
| Ubuntu cloud images | 6 | 6.6 GB | 0.34 GB | 19× | 5 |
| GitHub events, hourly | 744 | 477.8 GB | 33.81 GB | 14.1× | 0 (record mode alone; no hour is a version of another) |

Rates, wall clock on the one instance, S3 included:

| Phase | Time | Rate |
| :--- | ---: | ---: |
| zstd -3 put (16 threads; compress, upload) | 36 min | 547 MB/s of raw |
| store put (read, fingerprint, find the base, delta, upload) | 53 min | 372 MB/s of raw |
| zstd -3 get, every object one at a time, compared | 58 min | 341 MB/s of raw; 0 failed |
| store get, every object by its own process, compared | 43 min | 461 MB/s of raw; 1,192 byte-exact, 0 failed |
| store restore, the bucket by one process, compared | 33 min | 603 MB/s of raw; 1,192 byte-exact, 0 failed |

Per object, the put ran at 474 MB/s on average over the kernels, 330
over the hourly events, 230 over the Wikipedia tables and 113 over the
Ubuntu images (gzip inside: the deflate emulation's speed).

Against the 2026-09-22 second run (glyd-store 0.12.0): put 243 → 372
MB/s, get 166 → 461 MB/s, stored 49.56 → 46.29 GB (3.10× → 3.32× fewer
bytes than zstd -3). Where the bytes moved: the hourly events 37.13 →
33.81 GB (record mode alone; the parse's changes since v0.12.0), the
Ubuntu images 0.36 → 0.34 GB, the kernels within 2%; the English
Wikipedia tables 7.90 → 8.51 GB. There the fifth dump's three tables
(2026-09-01) were stored alone, where before one of them was a delta
against the first dump's, four months back, the root of a chain that
had reached depth four: 0.6 GB, to be looked at (v0.14.7 tightened
the region of the base a unit searches).

In money, S3 Standard at $23 per TB-month: this terabyte costs $42.4 a
year under zstd -3 and $12.8 under the store; per petabyte of such
data, $35.8K against $10.8K.

Also in this run, the machine's GNU gzip 1.12 on 512 MB of three
objects, then `glyd --max` on the gzip, decoded and compared: an hour
of GitHub events, gzip 75.3 MB → **38.3 MB** (`-r`); an English
Wikipedia categorylinks table, 27.1 MB → **16.7 MB** (`-r`); a Linux
tree, 72.2 MB → **57.9 MB**; all three byte-exact. These are 3–15%
larger than the 2026-09-22 figures (33.2, 16.2, 55.1 MB) because
`--max` has run without the 128 MB matcher since v0.14.4; `--max
--long` is what `--max` was.
