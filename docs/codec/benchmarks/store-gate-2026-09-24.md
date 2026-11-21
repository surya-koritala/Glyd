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

## v0.14.8: the same gate, the store keeping versions within their kind

The same corpus and instance type (glyd-store 0.14.8 at commit
de0ab9f; raw results in `benchmarks/gate/im4gn.4xlarge/`). What
changed in the store since the first run above: a version's base stays
in its own family past the depth cap (an object whose base holds under
98% of its fingerprints starts one: a new kernel major, a monthly
table), and a family's first object takes a shallow base; the delta
trial samples four windows spread over the object.

| | v0.14.7 (above) | v0.14.8 |
| :--- | ---: | ---: |
| stored | 46.29 GB (25.6×; 3.32× fewer bytes than zstd -3) | **44.35 GB (26.7×; 3.46×)** |
| put | 372 MB/s | 386 MB/s |
| zstd -3's own put, 16 threads | 547 MB/s | 535 MB/s |
| get, every object by its own process, compared | 461 MB/s, 1,192 byte-exact | 464 MB/s, 1,192 byte-exact |
| zstd -3's own get | 341 MB/s | 348 MB/s |
| restore, the bucket by one process, compared | 603 MB/s, 1,192 byte-exact | 571 MB/s, 1,192 byte-exact |

| Family | Raw | v0.14.7 | v0.14.8 |
| :--- | ---: | ---: | ---: |
| Linux 5.15 point releases (100) | 113.8 GB | 0.40 GB (286×) | 0.40 GB (286×) |
| Linux 6.1 (150) | 204.0 GB | 1.32 GB (155×) | **0.50 GB (406×)** |
| Linux 6.6 (150) | 213.2 GB | 1.84 GB (116×) | **0.57 GB (377×)** |
| English Wikipedia tables (15) | 165.6 GB | 8.51 GB (19.5×) | 8.67 GB (19.1×) |
| Simple English Wikipedia tables (27) | 3.2 GB | 0.08 GB | 0.09 GB |
| Ubuntu cloud images (6) | 6.6 GB | 0.34 GB | 0.33 GB |
| GitHub events, hourly (744) | 477.8 GB | 33.81 GB | 33.81 GB |

The kernels gained 2.1 GB: every fifth 6.1 and 6.6 release had been a
26–42 MB delta of 5.15.1, the root of the one chain all 400 releases
formed, and is now a delta of its own major's first release. The
English Wikipedia tables would have come to 7.94 GB (the new delta
trial takes the September page table as a 1,205 MB delta again;
measured in the put of a run stopped before its checks), and the
shallow rule gave back 0.73 GB of that: each month's page table holds
0.50–0.69 of the last, so each started a family and took a base two
months back, the September one going back to alone.
The next release lifts only a family's first object that sits at the
depth cap, when its family first needs it; measured on the Ryzen box it
keeps the kernels as here and the Wikipedia tables as in v0.14.7.

Two runs between v0.14.7 and this one are not reported as results: in
one the read-back compared each object with the wrong file (the index's
new family field had moved the name; its restore, which takes names
from the store, found all 1,192 byte-exact), and the other shared a
results directory with it and was stopped after its put when the first
finished. Both harness faults are fixed (`scripts/gate_run.sh`,
`scripts/gate_aws.sh`).

## v0.14.9: a family's first object lifted when its family needs it

The same corpus and instance type (glyd-store at commit 3856a03, the
store of v0.14.8 with the lift; raw results in
`benchmarks/gate/im4gn.4xlarge/`). In place of v0.14.8's rule (a
family's first object takes a shallow base), a family's first object
may sit at the depth cap until a later version of its family needs a
base there; then it is stored again against a shallower one.

| | v0.14.8 | v0.14.9 |
| :--- | ---: | ---: |
| stored | 44.35 GB (26.7×; 3.46× fewer bytes than zstd -3) | **43.61 GB (27.2×; 3.52×)** |
| put | 386 MB/s | 374 MB/s |
| get, every object by its own process, compared | 464 MB/s, 1,192 byte-exact | 486 MB/s, 1,192 byte-exact |
| zstd -3's own get | 348 MB/s | 346 MB/s |
| restore, the bucket by one process, compared | 571 MB/s, 1,192 byte-exact | 559 MB/s, 1,192 byte-exact |

| Family | Raw | v0.14.8 | v0.14.9, as put |
| :--- | ---: | ---: | ---: |
| Linux 5.15 point releases (100) | 113.8 GB | 0.40 GB (286×) | 0.40 GB (286×) |
| Linux 6.1 (150) | 204.0 GB | 0.50 GB (406×) | 0.48 GB (425×) |
| Linux 6.6 (150) | 213.2 GB | 0.57 GB (377×) | 0.55 GB (388×) |
| English Wikipedia tables (15) | 165.6 GB | 8.67 GB (19.1×) | **7.94 GB (20.9×)** |
| Simple English Wikipedia tables (27) | 3.2 GB | 0.09 GB | 0.07 GB |
| Ubuntu cloud images (6) | 6.6 GB | 0.33 GB | 0.32 GB |
| GitHub events, hourly (744) | 477.8 GB | 33.81 GB | 33.81 GB |

"As put" sums the put log, 43.56 GB; the lifts, which store a family
head again after its line in the log, account for the 0.05 GB more
that the index holds (43.61 GB). The English Wikipedia tables are where the delta trial of
v0.14.8 put them before its shallow rule gave 0.73 GB back, and 0.57
GB under v0.14.7.
