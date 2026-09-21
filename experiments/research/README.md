# Sizing the levers (2026-09-20)

Four directions measured at once with cheap proxies on real data, so the
next investment is chosen from one table. Scripts in this directory;
data in `corpus/research/` (kernel point releases from kernel.org,
loghub 2.0 logs from Zenodo record 8196385, the taxi Parquet from
`corpus/bench`). Every rebuild in the Glyd rows is byte-exact.

## A. Version chains (`chain.sh`)

Linux 6.10 -> 6.10.1 -> ... -> 6.10.5, each version against the
previous one, 10 threads:

| Version | Alone (zstd -19) | Chain, Glyd `--max --base` | Chain, zstd -3 patch | Chain, zstd -19 patch | vs 6.10 fixed, Glyd |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 6.10.1 | 149.5 MB | 3.04 MB · 0.28 s | 3.26 MB | 2.58 MB · 46 s | 3.04 MB |
| 6.10.2 | 149.5 MB | **1.75 MB · 0.13 s** | 3.13 MB | 2.22 MB · 48 s | 3.03 MB |
| 6.10.3 | 149.5 MB | **1.87 MB · 0.14 s** | 3.24 MB | 2.28 MB · 52 s | 3.16 MB |
| 6.10.4 | 149.5 MB | **1.79 MB · 0.14 s** | 3.16 MB | 2.26 MB · 52 s | 3.20 MB |
| 6.10.5 | 149.5 MB | **1.79 MB · 0.18 s** | 3.17 MB | 2.26 MB · 58 s | 3.20 MB |

Six versions stored as a chain: 149 + 5 x 1.8 = 158 MB against 897 MB
alone (5.7x); a chain beats a fixed base by 1.7x after four steps, and
Glyd's step beats zstd -19's patch on size at 300x its speed.

## B. System and application logs of other shapes (`logs.sh`)

128 MB of each (loghub 2.0). None is detected by record mode today
(their field counts vary line to line); `--max` lands at zstd -3 and
`--ultra -r` at zstd -19. The template prototype
(`experiments/structure/generic_templates.py`: tokens with digits are
variables, the rest a template; 32 MB slices, zstd -19 on the streams):

| Log | zstd -3 | zstd -19 | Glyd `--max` | Glyd `--ultra -r` | Templates vs zstd -19 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| HDFS (Hadoop file system) | 10.5x | 16.0x | 10.8x | 15.6x | **1.62x smaller** |
| Spark (application logs) | 14.5x | 24.4x | — | 24.2x | **1.61x** |
| Android (system log) | 12.9x | 23.0x | 14.6x | 20.8x | **1.35x** |
| Linux (syslog) | 12.3x | 19.6x | 12.2x | 18.7x | 1.25x |
| BGL (supercomputer RAS log) | 11.0x | 22.4x | 11.1x | 21.7x | 1.23x |

## C. The cold tier (`coldtier.sh`)

64 MB slices, one thread. xz -9 and brotli -q 11 land within 5% of
zstd -19; context mixing (zpaq -m5) is the only thing that moves the
JSON floor:

| Data | zstd -19 | Glyd `--ultra` | Glyd `--ultra -r` | zpaq -m5 (context mixing) | zpaq vs Glyd's best |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GitHub Archive JSON events | 14.6x · 13 s | 15.9x · 26 s | 13.2x | **22.8x · 176 s** | 1.43x smaller, 7x slower |
| NASA access log | 15.7x · 31 s | 15.6x | 26.4x · 3.9 s | **31.7x · 206 s** | 1.20x, 53x slower |
| enwiki page_props dump | 6.2x · 33 s | 6.1x | 8.6x · 11 s | **11.1x · 160 s** | 1.29x, 15x slower |
| webster (text) | 4.8x · 22 s | 4.8x · 20 s | 4.6x | **7.3x · 119 s** | 1.54x, 6x slower |

Shipped as `glyd --cold` (v0.7.0): 64 MB slices, 32 MB units on two
threads, every decode byte-checked — JSON events 22.5x, NASA `-r` 31.1x,
page_props `-r` 11.6x, webster 7.1x; zpaq -m5 within 3% either way at
3-4x its speed per core (1.2-1.3 MB/s each way).

## F. Binary float columns (`floats.py`)

Double columns of the taxi Parquet as raw f64 against decimal-scaled
integers (what ALP-style codecs do), both through zstd -19: 1.05-1.07x
(passenger_count 1.30x). zstd already sees the decimal doubles' repeated
bytes. Not a lever.

## What it says

1. **Template log mode**: the largest untouched class (application and
   system logs with variable line shapes) gains 1.2-1.6x over zstd -19
   with a tokenizer that separates templates from variables — about a
   week of work, on top of record mode's column types.
2. **Version chains**: the codec side is done and beats zstd -19's patch
   at 300x its speed; what is left is the pack (chains, re-materialised
   reads, cross-object matching).
3. **Context mixing for cold data**: 1.2-1.5x over every LZ codec on
   every text-like class, JSON events included, at 1 MB/s per core —
   the only lever left for hash-and-text JSON; cold archives only.
   Built as `--cold` (section C).
4. Float columns and Parquet pages: dead ends, measured.

## G. Model checkpoints (`tensors.py`)

Public checkpoints from Hugging Face, every transform lossless: byte
planes (the k-th byte of every element together), and the XOR with the
previous checkpoint of the same run, under zstd -3 / -19 and Glyd
`--max` (10 threads).

| File | as it is: zstd -3 · zstd -19 · Glyd --max | byte planes: zstd -3 · zstd -19 · Glyd --max |
| :--- | ---: | ---: |
| Pythia-70m step 142000, fp32 (282 MB) | 1.74x · 2.06x · 1.79x | 2.20x · 2.33x · **2.22x at 970 MB/s** |
| Qwen2.5-0.5B, bf16 (988 MB) | 1.28x · 1.32x · 1.28x | 1.41x · 1.49x · 1.41x |

Between consecutive Pythia checkpoints (1,000 steps apart, end of
training) 0.3% of the elements are identical, so byte-level history
(Glyd `--base`) gains nothing: 1.79x. The XOR of the two, in byte
planes: **2.94x (zstd -3), 3.18x (zstd -19), 2.96x (Glyd --max)** — a
checkpoint after the first costs 1.7x less than compressed alone.

What it says: a tensor-aware transform is worth 1.26x on fp32 files and
1.10x on bf16 over zstd -3 (the mantissa bits are noise and stay), and
1.7x per checkpoint of a run against its predecessor. Real at petabyte
scale, not a leap; ZipNN-class results (17-33% on bf16) agree.

## H. A bucket, compressed across its objects (`examples/bucket.rs`)

The redundancy of object storage is between objects, not inside them.
A realistic bucket (`scripts/download_bucket.sh`, 39 objects, 39.2 GB:
six builds of the Ubuntu 24.04 cloud root filesystem, the fifteen
Linux 6.10 point releases, two months of three Wikipedia tables, twelve
hours of GitHub events), each object arriving in order. For each, the
stored object sharing the most fingerprints (one sparse anchor in 4 KB,
the last eight holders of each kept) is its base; it is stored as a
delta (`--max --base`) when that saves a fifth or more of its own size,
chains at most four deep (past that, the chain's root is the base).

| Family | raw | zstd -3, each object alone | Glyd, across the bucket | gain |
| :--- | ---: | ---: | ---: | ---: |
| Linux releases (15) | 22.5 GB | 3,237 MB | **232 MB** | **13.9x** |
| Ubuntu images (6 builds) | 6.6 GB | 1,880 MB | **361 MB** | **5.2x** |
| Wikipedia dumps (2 months, 3 tables) | 0.7 GB | 140 MB | **71 MB** | **2.0x** |
| GitHub events (12 hours) | 9.4 GB | 875 MB | 670 MB | 1.3x (no base pays) |
| **The bucket** | **39.2 GB** | **6,132 MB (6.4x)** | **1,334 MB (29.4x)** | **4.6x** |

1,362 MB/s on ten M1 cores, every delta byte-exact by construction of
base mode. An event stream gains nothing across hours (its repeats are
short and LZ already has them); everything that is a version of
something gains 2-14x. Read cost: an object at depth d is d + 1 decodes
at 6-10 GB/s.
