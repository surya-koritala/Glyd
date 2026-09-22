# The store's gate: a terabyte bucket, 2026-09-22

One instance next to the bucket (`scripts/gate_aws.sh`: im4gn.4xlarge,
Neoverse-N1, 16 vCPU, 61 GB, 7.5 TB NVMe, us-east-1), glyd-store
v0.11.2 built from commit 660d873, objects in S3 over the store's own
client. The corpus (`scripts/gate_corpus.sh`): 1,192 public objects,
1,184,268,065,453 bytes, as object storage holds them — 400 Linux
point releases (6.6.1–150, 6.1.1–150, 5.15.1–100), every hour of
GitHub events in January 2024 (744), five English Wikipedia dumps'
page, page_props and categorylinks tables, nine Simple English dumps'
three tables, six Ubuntu 24.04 cloud images. Put in name order, every
object read back and compared byte for byte, the metadata directory
deleted and rebuilt from the bucket, then verified
(`scripts/gate_run.sh`; raw results in `benchmarks/gate/im4gn.4xlarge/`).

| | Bytes | Against raw | Against zstd -3 |
| :--- | ---: | ---: | ---: |
| Raw | 1,184.3 GB | | |
| zstd -3, each object alone (16 threads, 817 MB/s) | 153.5 GB | 7.7× | |
| **Glyd store** | **49.0 GB** | **24.2×** | **3.13× fewer bytes** |

| Family | Objects | Raw | Stored | Against raw | Deltas |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Linux 5.15 point releases | 100 | 113.8 GB | 0.40 GB | 285× | 99 |
| Linux 6.1 | 150 | 204.0 GB | 1.29 GB | 158× | 150 |
| Linux 6.6 | 150 | 213.2 GB | 1.86 GB | 115× | 150 |
| English Wikipedia tables | 15 | 165.6 GB | 7.90 GB | 21× | 10 |
| Simple English Wikipedia tables | 27 | 3.2 GB | 0.08 GB | 43× | 24 |
| Ubuntu cloud images | 6 | 6.6 GB | 0.36 GB | 18× | 5 |
| GitHub events, hourly | 744 | 477.8 GB | 37.13 GB | 12.9× | 0 (record mode alone; no hour is a version of another) |

Rates, wall clock on the one instance, S3 included:

| Phase | Time | Rate |
| :--- | ---: | ---: |
| put (read, fingerprint, find the base, delta, upload) | 2 h 12 min | 150 MB/s of raw |
| get, every object, compared with its original | 1 h 46 min | 186 MB/s of raw; 1,192 byte-exact, 0 failed |
| rebuild the metadata directory from the bucket | 1 h 26 min | index identical to the one before |
| verify after the rebuild | 1 h 02 min | 1,192 ok, 0 failed |

In money, S3 Standard at $23 per TB-month: this terabyte costs $42.4 a
year under zstd -3 and $13.5 under the store; per petabyte of such
data, $35.8K against $11.4K.

What the run found besides the numbers: the first attempt died of
memory on the 20 GB Wikipedia tables (three copies of one object in
memory: the file read, the base, the last object kept as the next
base); v0.11.2's successor maps the file and caps the kept object at
1 GB, and the run above is the second attempt. The put rate is the
instance's, single-process, including the 20 GB objects; the store
does not yet overlap one object's upload with the next one's
compression. zstd -3 per family was not recorded, only its total.
