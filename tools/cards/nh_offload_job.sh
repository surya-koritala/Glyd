#!/usr/bin/env bash
# Nemotron 3 Super's expert cache on ONE PCIe card (Linear GLY-130; rp_run.sh contract: results in ~/results, ~/results/DONE at the end): a real RTX 4090 (24 GB, PCIe 4) or RTX 5090 (32 GB, PCIe 5), or a 48 GB L40/L40S
# (PCIe 4 x16, as a 4090 has) with the cache capped to what a 24 GB and a 32 GB card would have. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 at 2dc98e2a (247 GB) packed is 70.5 GB at wren and 83.5 GB at linnet: no consumer card
# holds either, so every expert (63.4 GB at wren, 77.5 GB at linnet) stays in the pod's memory, pinned, and a cache of them is on the GPU (src/offload.rs). An expert that is not on the GPU is copied in (the cache's first
# version, `copy`: the records' 11.95 tokens a second on the 4090 and 43.59 on the 5090) or its pairs are computed by the CPU from the pod's memory while the GPU runs the others (src/cpu_experts.rs; modes `cpu`, `split`,
# `auto`: what `plan` says of each miss), which takes a second copy of the experts' first matrices in the row layout besides the experts: half their bytes again, 29.5 GiB at wren and 36.1 at linnet. The level follows the card,
# as nh_run_check_job.sh's does:
#   23 to 40 GB (RTX 4090: 24,564 MiB, RTX 5090: 32,607 MiB)   wren (--layout i4: the experts four-bit, every other matrix six-bit), the cache `--expert-cache auto`: what `run` gives that card (by the plan's figures, 122 experts
#                                                              a layer of 512 at 23 GiB free, 191 at 31 GiB); on the 4090 also the budget of a 16 GB card (RTX 4080: 16,376 MiB, about 15 GiB free): `--expert-cache auto:15`, 52 experts a layer
#                                                              (6.00 GiB of cache; `auto:14` is the 44 of the least cache, and the plan refuses a card with less than 14.0 GiB free), labelled "4090 capped to a 16 GB card's budget"
#   45 GB or more (L40, L40S: 46,068 MiB)                      linnet (--layout i5), three caches: `auto:23` (a 24 GB card's: 106 experts a layer), `auto:31` (a 32 GB card's: 163) and `auto` (this card's)
# Each step is a line of steps.txt with its log; a step that fails does not stop the ones that can still run:
#   0. guards     the CPUs (12 or more: the pack is bound by them), the pod's memory (the experts are pinned in it: 72 GiB at wren, 88 at linnet; `free -g` is recorded and a pod that has less stops here, at its first lines,
#                 with the line that says so; a pod with less than the experts and the CPU's half (102 GiB, 124) runs the mode `copy` alone and says so), the card's memory, nvcc's compute_N for the card (the RTX 5090's
#                 sm_120 wants CUDA 12.8), the download's speed (8 minutes in, it must project to be done in 45)
#   1. tiny       for each level (6, 5, 4) a random tiny model with mixture-of-experts layers (nemotron_ref.py moetoy: no Hub read), then glyd-engine nhoffload on it: run with every expert on the GPU (eager and as a graph)
#                 and through the cache with 3, 5 and 8 places a layer of 16 (a call of every layer is two waves, then LRU churn), a prompt in chunks and in one, on the first pass, on the warm cache, with hints and with
#                 a predictor: every logit bit for bit. Twice: with the experts' second matrices in the row layout (the default: the sparse product, a place holds an expert's first matrix and its rows) and in the block
#                 layout (GLYD_ENGINE_DENSE_DOWN=1, a development build's variable: the library's dense product); and with the CPU computing the pairs of the experts that are not in places (--offload-mode cpu, split and auto
#                 on 4 threads): every logit within two places of bf16 of the largest, the token it picks within that of the best (`logits_within`; the tiny model's weights are on the grids, the CPU's weights the GPU's); then
#                 moecheck relu2-22of512 relu2-rows-22of512 (the dense and the sparse product over Int6, Int5 and Int4 experts against the CPU reference), and cpuexperts (the CPU kernels' own speed on the pod's processor).
#                 A run of nhoffload that fails is a fault of the code found in seconds: the job stops before the pack, its longest step, and says so in steps.txt
#   2. download   the checkpoint (the pinned revision), in the background beside the builds
#   3. pack       glyd-engine pack MODEL ~/v/packed --layout i4 (wren) or i5 (linnet), the experts' second matrices in the row layout: the record's pod (22 CPUs) took 18 minutes, 12 CPUs about 60
#   4. generate   glyd-engine generate with --expert-cache on the packed model, one run for each cache above, on ONE load of the model for all the variants of --offload-mode (the cache cold at the start of each): 256 greedy
#                 tokens of a prompt, the prefill and the decode steps timed apart (decode tokens a second, and of the second half of the steps, the first filling a cold cache), and for each variant the cache's counts a token:
#                 expert reads, the share found on the GPU (the record's replay at 32% of the experts says 74.2% of a chat's decode reads), the misses (by the CPU, copied in), the CPU's time and the copy stream's, the host's wait
#                 for the GPU. The variants: copy (the baseline), cpu (the CPU's threads: the pod's CPUs less two, then 8, 16, 32 where the pod has them, and the default kept to cores, `:pin`), split and auto (also on 16 threads).
#                 The table of summary.txt has a row for each
#   4b. 16 GB     on the 4090 alone: step 4 again with `--expert-cache auto:15` (a 16 GB card's budget; the plan line of the log says the places, or that the weights and the least cache do not fit) for copy, cpu, split and auto
#   4c. logits    the whole model's logits through the cache in the fastest variant of step 4 (nhtoy PACKED OUT --expert-cache auto --offload-mode ...: 64 tokens of a text prefilled, 96 steps teacher-forced, eager) against the
#                 bf16 checkpoint run layer by layer on the GPU (nemotron_ref.py full, as nh_run_check_job.sh step 5): top-1 token and KL divergence at each of 97 positions. The real gate of the CPU's sums on the model
#   4d. 32 GB RAM Hummingbird (HB=1) only: the SSD tier (GLY-130; src/offload.rs `Tier`): a computer of 32 GB cannot pin the 39.2 GiB of experts, so `--host-memory 20` gives them 20 GiB and the rest are read from the packed
#                 directory on the pod's disk as the router asks for them (O_DIRECT, 16 reads at a time). First `glyd-engine diskcheck` (random whole experts of the directory, one at a time and 16 at a time, the page cache
#                 bypassed, MB/s: a consumer NVMe on PCIe 4 reads about 5 to 7 GB/s, which the pod's disk is to be compared with), then step 4's run on one load each for the card's own cache and (on the 4090) the 16 GB
#                 budget's `--expert-cache auto:15`, in the modes cpu and auto, with the experts read from the disk a token (disk_reads, disk_mb, disk_ms), the peak resident memory (getrusage) and the plan line
#   5. run        glyd-engine run on the packed directory: one answer (--prompt), the way a user starts it (the mode `auto`)
#   6. dense      the same model packed again with the experts' second matrices in the block layout (GLYD_ENGINE_DENSE_DOWN=1), and step 4's runs on it in the mode copy: the dense product, the same caches on the same card, so that
#                 the summary puts the decode tokens a second of the sparse and the dense product side by side (the records' runs are the dense one's: benchmarks/gpu/nh-offload-4090-2026-10-10, 11.95 tokens a second at 123 of 512
#                 experts a layer, 67.7% of the reads on the GPU, 83% of the decode time the copy stream's; nh-offload-5090-2026-10-10, 43.59 at 190, 77.9%, 52%). Left out where the time left is short of another pack and
#                 the runs (it is, after steps 4 to 4c)
# A pod of more GPUs has more of the host's memory, and Runpod's per-GPU memory on these cards is not known here: the launch lines below ask for RP_GPUS=2 for the 4090 and the 5090 (3 if the line "ram:" in steps.txt says
# the pod is short). The job uses GPU 0 alone (CUDA_VISIBLE_DEVICES=0) and says so in steps.txt; the other GPUs idle and are paid for.
# Uploaded beside it: src.tar (git archive of the tree, prefix src/). Results: summary.txt (made after every step), steps.txt, one .log a step (generate-CARD.log the sparse product's with the variants, generate-16g.log the
# 16 GB budget's, generate-dense-CARD.log the dense one's, logits-run.log and logits-check.log the whole model's), tiny6/, tiny5/, tiny4/ (toy.txt), smi.csv (nvidia-smi each second), free.txt (free -g), machine.txt, gpu.txt (rp_run.sh).
#
#   cd <the tree at the commit to test> && W=$(mktemp -d) && git archive --format=tar.gz --prefix=src/ HEAD > $W/src.tar
#   RTX 4090, two GPUs for the memory:   NAME=nh-offload-4090 MAX_MIN=150 RP_GPUS=2 RP_DISK=520 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh 4090 tools/cards/nh_offload_job.sh   # 2 x $0.89 = $1.78/h; about 65 minutes, $1.9 (the cap, 150 minutes: $4.45)
#   RTX 5090, two GPUs for the memory:   NAME=nh-offload-5090 MAX_MIN=150 RP_GPUS=2 RP_DISK=520 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh 5090 tools/cards/nh_offload_job.sh   # 2 x $1.19 = $2.38/h; about 40 minutes, $1.6 (the cap: $5.95)
#   L40 (then L40S), one GPU:            NAME=nh-offload-l40  MAX_MIN=150 RP_DISK=520 UPLOAD="$W/src.tar" bash tools/runpod/rp_class.sh l40 tools/cards/nh_offload_job.sh              # $0.82/h ($1.09 on an L40S); about 100 minutes, $1.4 ($1.8) (the cap: $2.05, $2.73)
#   (with RP_GPUS=3: $2.67/h on 4090s, $3.57/h on 5090s.) The minutes are the records' (nh-offload-4090-2026-10-10: a pod of 22 CPUs, 34 minutes in all, the pack 18 and a run of generate 6, of which 290 s the load; nh-offload-5090-2026-10-10:
#   55 CPUs, 16 minutes, the pack 7 and a run of generate 2; nh-run-check-2026-10-10: the whole model's logits 2 minutes of the engine and 2 of the reference) and on top of them the CPU's half made at each load (about 15 s), a run of
#   generate of one load and nine variants (about 10 minutes), the 4090's second (the 16 GB budget, about 8) and the logits (a load, 160 tokens, the reference: about 10); the L40's three caches three runs of each, of three variants; on a pod
#   of 12 CPUs the pack takes about 60 minutes and the later steps are left out as the time runs short (steps.txt says so). None of this was launched.
# (rp_guard.sh deletes glyd-* pods older than its MAX_AGE: MAX_AGE=160 bash tools/runpod/rp_guard.sh & first.) Results come to $GLYD_RUNS/results-$NAME. RP_DISK: the checkpoint 247 GB, the packed model 84 GB (71 at wren)
# and, in step 6, the dense one (as much again), the builds and the environment 30 GB.
# Hummingbird (HB=1 in an uploaded hb.env; Linear GLY-98, GLY-134): Nemotron 3 Super at the Hummingbird level (src/moe_hb.rs: the routed experts as codebook rows of 3 and 2 bits, 42.1 GB, 39.2 GiB; every other matrix six-bit, 7.8 GB), for the cards
# of 16 to 32 GB with system RAM; the same steps with these changes: the card of 15 to 40 GB runs the cache `--expert-cache auto` (one of 45 GB or more the three caches of a 24 GB card, a 32 GB card and its own); the experts pinned are 39.2 GiB, so the pod
# needs 52 GiB of memory; there are no wren or linnet packs and no dense steps; step 1 is the tiny model packed at this level from a random export (glyd-engine hbtoy) and run through the cache against every expert there (nhoffload) and glyd-engine hbcheck (the
# kernels against float64 at the real shape, and their speed); step 3 packs from an export, or takes a finished directory; glyd-gpu verify decodes every pack of the directory on the CPU. hb.env (all optional but one of the first three):
#   PACKED_S3=s3://BUCKET/PREFIX/   a finished glyd-v7 Hummingbird directory, downloaded in the background beside the builds (no Hugging Face download and no pack), then verified
#   EXPORT_S3=s3://BUCKET/PREFIX/   the calibration export (meta.json and layer-LLL.safetensors): downloaded into EXPORT_DIR beside the bf16 checkpoint's download, then packed here (glyd-engine pack MODEL DIR --layout hummingbird --codes EXPORT_DIR)
#   EXPORT_DIR=/path                an export already on the pod (default ~/v/export: where one is if a calibration ran on this pod): packed as above
#   PACKED_UP=s3://BUCKET/PREFIX/   where the directory packed here goes once glyd-gpu verify passes (in the background beside the runs, waited for at the end): the next pods take it as PACKED_S3
# The S3 prefixes are read with the keys of an uploaded s3.env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION: short-lived, for the bucket alone): sourced in the download step's own shell and nowhere else, never echoed, and removed when
# the job ends. `aws s3 cp --recursive` runs 64 requests at once in chunks of 64 MB (aws configure set default.s3.max_concurrent_requests 64, multipart_chunksize 64MB), and the MB/s of each download is a line of steps.txt. RP_DISK: 150 for a finished directory (50 GB),
# 420 with the export and the checkpoint (247 GB + 42 GB + 50 GB). None of this was launched.
# The CPU half at this level (GLY-130): both matrices of an expert are neuron rows in the host tier, so the CPU computes the pairs of the experts that are not in places from the pinned experts where they are: no second copy (the
# pod needs the 52 GiB alone, and the sweep of modes runs on any pod that passes the memory guard). One load of the directory serves the variants of step 4 (copy, cpu on the default threads, 8, 16 and 32, cpu kept to cores, split,
# auto, and split and auto on 16 threads), the cache cold at the start of each; a 24 GB card's `--expert-cache auto` is the plan's 188 experts a layer of 512 at 23 GiB free, a 32 GB card's 293 at 31 (the mean expert 2,056,068
# bytes; the figures of cli.rs's test, a directory's own bytes may move them a little). The 4090 also runs step 4b, "4090 capped to a 16 GB card's budget": `--expert-cache auto:15` (a 16 GB RTX 4080 leaves about 15 GiB free),
# 84 experts a layer of 512 (6.43 GiB of cache) by the plan's figures, for copy, cpu, split and auto: the owner's floor. The SSD tier (steps tinyhb-ssd-MODE and 4d): step 1 runs nhoffload on the tiny model with `--host-memory` (three budgets, 0.0005, 0.0003 and 0.00012 GiB of
# its 0.00065: most of its experts stay in the packed directory) in each of copy, cpu, split and auto: every logit and token of the run with the experts on the disk the same to the bit as the run with all of them in memory in that
# mode (the planner's times held so that two runs plan alike), and experts read from the directory. Step 1 also runs nhoffload on the tiny model in the modes cpu, split and auto (every logit within two places
# of bf16 of the largest, the places counted at the mean of the widths and split between the two); step 4c runs the whole model's logits through the cache twice, with the experts copied and in the fastest variant, and compares
# the two (nemotron_ref.py compare: the level is lossy against bf16 whatever serves the misses), and, where the bf16 checkpoint was downloaded (no PACKED_S3), against the checkpoint (nemotron_ref.py full).
#   W=$(mktemp -d) && git archive --format=tar.gz --prefix=src/ HEAD > $W/src.tar && printf 'HB=1\nPACKED_S3=s3://BUCKET/PREFIX/\n' > $W/hb.env      # (s3.env: the keys, a file of the owner's, not made here)
#   NAME=nh-hb-5090 MAX_MIN=90 RP_GPUS=2 RP_DISK=150 UPLOAD="$W/src.tar $W/hb.env $S3_ENV" bash tools/runpod/rp_class.sh 5090 tools/cards/nh_offload_job.sh
#   NAME=nh-hb-4090 MAX_MIN=90 RP_GPUS=2 RP_DISK=150 UPLOAD="$W/src.tar $W/hb.env $S3_ENV" bash tools/runpod/rp_class.sh 4090 tools/cards/nh_offload_job.sh   # 2 x $0.89 = $1.78/h; about 50 minutes, $1.5 (the cap, 90 minutes: $2.7)
# (the minutes are an estimate from the records' loads and variants: the build 10 minutes beside the download, the tiny model 5 (and 2 for the runs on the disk), the 24 GB card's run 15 (a load of 3 minutes and ten variants), the 16 GB budget's 7,
# the 32 GB RAM runs (4d) 8 each (a load reads 20 GiB from the disk) and the disk check 1, the logits 12.)
set -u
R=~/results; mkdir -p $R ~/v; cd ~/v; tar -xf ~/src.tar
P=$R/steps.txt; : > $P
[ -f ~/hb.env ] && . ~/hb.env   # (HB, PACKED_S3, EXPORT_S3, EXPORT_DIR: no keys in it)
HB=${HB:-0}; PACKED_S3=${PACKED_S3:-}; EXPORT_S3=${EXPORT_S3:-}; EXPORT_DIR=${EXPORT_DIR:-$HOME/v/export}; PACKED_UP=${PACKED_UP:-}
T0=$(date +%s)
MAX_MIN=${JOB_MAX_MIN:-135}   # (the launcher's cap, MAX_MIN=150 in the launch lines above, counts the pod's start too, up to 15 minutes of it: the job starts no step in its last 8 minutes and cuts a step's limit to the time it has left)
left() { echo $(( MAX_MIN * 60 - ($(date +%s) - T0) )); }
late() { [ "$(left)" -le 480 ]; }
tmo() { local l=$(( $(left) - 240 )); [ "$l" -lt 60 ] && l=60; [ "$l" -lt "$1" ] && echo "$l" || echo "$1"; }   # (a step's limit: its own, or the time left less 4 minutes)
step() { local n=$1; shift; local t=$(date +%s); "$@" > $R/$n.log 2>&1; local rc=$?; LAST_S=$(( $(date +%s) - t )); echo "$n: rc=$rc $LAST_S s (job minute $(( ($(date +%s) - T0) / 60 )))" >> $P; return $rc; }
RAM=0; CPUS=0; NGPU=0; GPU_NAME=; LEVEL=; CAPS=; RAMCAPS=; SRC=~/v/src; SMI=; HOSTHALF=0; SWEEP=copy; SWEEP16=copy; CAP16=; TH=1; BEST=
tps() { sed -n 's/.*(\([0-9.]*\) tokens a second).*/\1/p' "$1" 2> /dev/null | tail -1; }   # (the decode speed of a generate log's last "prompt tokens" line)
vtps() { sed -n "s/^\[$2\] result: .*decode_tps=\([0-9.]*\).*/\1/p" "$1" 2> /dev/null | tail -1; }   # (the decode speed of the variant $2 of a generate log of variants)
# the variant of a generate log of variants with the most decode tokens a second (MODE[:THREADS][:pin], as --offload-mode names it)
best() { sed -n 's/^\[\([^]]*\)\] result: .*decode_tps=\([0-9.]*\).*/\2 \1/p' "$1" 2> /dev/null | sort -g -r | head -1 | cut -d' ' -f2; }
# a generate log of variants: the plan and the load, a row for each variant (decode tokens a second, of the second half of the steps, of the prefill; the share of expert reads found on the GPU, the misses a token and how
# they were served, the CPU's and the copy stream's milliseconds a token, the host's wait for the GPU), and where each variant's tokens first differ from the first's
sweep_report() {
    grep -E "^expert cache plan|^loaded in|^peak resident memory|^the computer's memory holds|^glyd-engine:|does not make the scalar" "$1" 2> /dev/null | head -8
    echo "variant        decode tok/s  2nd half  prefill tok/s  found%  missed/token  by CPU  copied  later  CPU ms/token  copy ms/token  wait ms/token  disk reads/token  disk MB/token  disk ms/token"
    sed -n 's/^\[\([^]]*\)\] result: \(.*\)$/\1 \2/p' "$1" 2> /dev/null | awk '{ for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
        printf "%-14s %12s %9s %14s %7s %13s %7s %7s %6s %13s %14s %14s %17s %14s %14s\n", $1, v["decode_tps"], v["second_half_tps"], v["prefill_tps"], v["found_pct"], v["missed"], v["host"], v["copied"], v["promoted"], v["host_ms"], v["copy_ms"], v["wait_ms"], v["disk_reads"], v["disk_mb"], v["disk_ms"] }'
    grep -E "tokens against the first variant" "$1" 2> /dev/null
    grep -q "result:" "$1" 2> /dev/null || tail -3 "$1" 2> /dev/null
}
summary() {
    echo "Nemotron 3 Super at ${LEVEL:-?} through the expert cache on ${GPU_NAME:-?} (a pod of $NGPU GPUs, GPU 0 used), $((RAM / 1024)) GiB of memory, $CPUS CPUs, tree $(cat $SRC/COMMIT 2> /dev/null)"
    echo "--- steps"; cat $P
    echo "--- free -g at the start"; cat $R/free.txt 2> /dev/null
    if [ "$HB" = 1 ]; then
        echo "--- tiny model at the Hummingbird level (a random export, 3 and 2 bits): every logit against the run with every expert on the GPU (nhoffload), places split between the two widths"; grep -E "^(ok|FAIL|nhoffload|the experts)|expert cache:" $R/tinyhb-offload.log 2> /dev/null
        for m in cpu split auto; do
            echo "--- the same with the CPU computing the pairs of the experts that are not in places from the host tier in place (--offload-mode $m, 4 threads): within two places of bf16 of the largest logit"; grep -E "^(ok|FAIL|nhoffload|the experts)|expert cache:" $R/tinyhb-offload-$m.log 2> /dev/null
        done
        for m in copy cpu split auto; do
            echo "--- the same with the experts on the disk (--host-memory 0.0005,0.0003,0.00012 GiB of the tiny model's 0.00065, --offload-mode $m, 4 threads): every logit and token the same to the bit as the run with every expert in memory in that mode, and experts read from the packed directory"; grep -E "^(ok|FAIL|nhoffload|the experts)|experts read from the packed directory" $R/tinyhb-ssd-$m.log 2> /dev/null
        done
        echo "--- hbcheck (the kernels against float64, and their speed at the real shape)"; grep -E "^(ok|FAIL|all|FAILED|speed)" $R/hbcheck.log 2> /dev/null
        echo "--- the directory: glyd-gpu verify (every pack decoded on the CPU to glyd.json's sha256)"; tail -3 $R/verify.log 2> /dev/null
    fi
    for lv in 6 5 4; do
        [ "$HB" = 1 ] && break
        echo "--- tiny model, level $lv: every logit against the run with every expert on the GPU (nhoffload), the experts' second matrices in the row layout (the sparse product)"; grep -E "^(ok|FAIL|nhoffload|the experts)|expert cache:" $R/tiny$lv-offload.log 2> /dev/null
        echo "--- the same in the block layout (the dense product)"; grep -E "^(ok|FAIL|nhoffload|the experts)|expert cache:" $R/tiny$lv-offload-dense.log 2> /dev/null
        for m in cpu split auto; do
            echo "--- the same with the CPU computing the pairs of the experts that are not in places (--offload-mode $m, 4 threads): within two places of bf16 of the largest logit"; grep -E "^(ok|FAIL|nhoffload|the experts)|expert cache:" $R/tiny$lv-offload-$m.log 2> /dev/null
        done
    done
    [ "$HB" = 1 ] || { echo "--- moecheck (the dense and the sparse second product over Int6, Int5 and Int4 experts against the CPU reference)"; grep -E "^(ok|FAIL|all|FAILED)" $R/moecheck.log 2> /dev/null; }
    echo "--- cpuexperts (the CPU kernels on this pod's processor: pairs a second, a decode token's time for 100, 230 and 400 misses)"; grep -v "^$" $R/cpuexperts.log 2> /dev/null | head -14
    echo "--- the pack"; cat $R/pack.log $R/pack.txt 2> /dev/null
    case "$GPU_NAME" in
        *4090*) record="; the record's 4090 run (dense, 123 of 512 experts a layer): 11.95";;
        *5090*) record="; the record's 5090 run (dense, 190 of 512 experts a layer): 43.59";;
        *) record="";;
    esac
    [ "$HB" = 1 ] && record=""   # (the records are wren's)
    for c in $CAPS; do
        echo "--- generate with the cache of ${c%%:*} (--expert-cache ${c#*:}), the experts' second matrices in the row layout (the sparse product), one load, a cold cache for each variant of --offload-mode (copy is the cache's first version)"
        sweep_report $R/generate-${c%%:*}.log
        echo "--- the fastest variant: $(best $R/generate-${c%%:*}.log) at $(vtps $R/generate-${c%%:*}.log "$(best $R/generate-${c%%:*}.log)") tokens a second$record"
        [ -f $R/generate-dense-${c%%:*}.log ] || continue
        echo "--- the same with the block layout (the dense product), copy"
        grep -E "loaded in|prompt tokens|^prefill:|^decode:" $R/generate-dense-${c%%:*}.log 2> /dev/null
        grep -q "^decode:" $R/generate-dense-${c%%:*}.log 2> /dev/null || tail -3 $R/generate-dense-${c%%:*}.log 2> /dev/null
        sp=$(vtps $R/generate-${c%%:*}.log copy); dn=$(tps $R/generate-dense-${c%%:*}.log)
        [ -n "$sp" ] && [ -n "$dn" ] && echo "--- decode tokens a second with the cache of ${c%%:*}, copy: sparse $sp, dense $dn (sparse / dense $(awk -v s=$sp -v d=$dn 'BEGIN { printf "%.3f", s / d }'))$record"
    done
    if [ -n "$CAP16" ]; then
        echo "--- 4090 capped to a 16 GB card's budget (--expert-cache ${CAP16#*:}: the plan sees about 15 GiB free, as a 16 GB RTX 4080 leaves; the places, or the refusal, in the plan line): decode tokens a second by variant, the share of reads found on the GPU and the misses a token"
        sweep_report $R/generate-${CAP16%%:*}.log
        grep -E "GiB in use" $R/generate-${CAP16%%:*}.log 2> /dev/null | head -1
    fi
    echo "--- the whole model's logits through the cache in the fastest variant (${BEST:-?}) against the bf16 checkpoint: top-1 and KL (nhtoy, then nemotron_ref.py full)"
    grep -E "^eager:|expert cache plan|does not make the scalar" $R/logits-run.log 2> /dev/null | cut -c1-600
    grep -E "^(ok|FAIL|full)|position|toy|prefill|candidates" $R/logits-check.log 2> /dev/null
    if [ "$HB" = 1 ]; then
        echo "--- the same run with the experts copied (the baseline), and the logits of the fastest variant against it (nemotron_ref.py compare: the level is lossy against bf16 whatever serves the misses)"
        grep -E "^eager:|expert cache plan" $R/logits-run-copy.log 2> /dev/null | cut -c1-600
        grep -E "^(ok|FAIL)" $R/logits-compare.log 2> /dev/null
    fi
    if [ "$HB" = 1 ]; then
        echo "--- the disk of this pod, as the SSD tier reads it (diskcheck: random whole experts of the packed directory, page cache bypassed; a consumer NVMe on PCIe 4 reads about 5 to 7 GB/s)"; grep -E "^diskcheck" $R/diskcheck.log 2> /dev/null; grep -q "^diskcheck" $R/diskcheck.log 2> /dev/null || tail -3 $R/diskcheck.log 2> /dev/null
        for c in $RAMCAPS; do
            echo "--- 32 GB RAM: --host-memory 20 with the cache of ${c%%:*} (--expert-cache ${c#*:}), one load, cpu and auto: decode tokens a second, the share of reads found on the GPU, the misses a token, the experts read from the disk a token, and the peak resident memory (getrusage)"
            sweep_report $R/generate-ram20-${c%%:*}.log
            grep -E "^\[[^]]*\] decode a token, the computer's memory" $R/generate-ram20-${c%%:*}.log 2> /dev/null
        done
    fi
    echo "--- run: one answer"; grep -v "^$" $R/run-prompt.log 2> /dev/null | tail -6
    echo "--- GPU memory used at most in the job (MiB)"; awk -F', ' '$2 + 0 > m { m = $2 + 0 } END { print m }' $R/smi.csv 2> /dev/null
}
save() { ( summary > $R/summary.tmp 2>&1; mv -f $R/summary.tmp $R/summary.txt ) 2> /dev/null; }   # (a subshell: a variable not yet set ends it, not the job)
fin() { [ -z "${SMI:-}" ] || kill $SMI 2> /dev/null; rm -f ~/s3.env; save; touch $R/DONE; }   # (not `kill 0`, which is the whole process group; the keys of s3.env go with the job)
trap fin EXIT
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # (the repo is ungated: nothing here reads or sends a token)
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_HIGH_PERFORMANCE=1 HF_HOME=$HOME/v/hf PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 RUST_BACKTRACE=1
export CUDA_VISIBLE_DEVICES=0   # (one GPU whatever the pod has: a pod of two or three is for its memory; nvidia-smi below asks for GPU 0 with -i)
REPO=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16; REV=2dc98e2afe4face0e4ce40972a915c45368bd34a
E=$SRC/glyd-engine; REF=$E/tools/nemotron_ref.py; M=~/v/m; PK=~/v/packed
CC=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | head -1); ARCH=$([ "$CC" = "9.0" ] && echo 90a || echo "${CC/./}")
# the CPUs this container really has (a cgroup quota, not the host's cores): the packer's threads and torch's pools are sized by it
cpus() { local q p c=${CGROOT:-/sys/fs/cgroup}; { read -r q p < $c/cpu.max; } 2> /dev/null && [ "$q" != max ] && { echo $(( (q + p - 1) / p )); return; }
         { read -r q < $c/cpu/cpu.cfs_quota_us; read -r p < $c/cpu/cpu.cfs_period_us; } 2> /dev/null && [ "${q:--1}" -gt 0 ] && { echo $(( (q + p - 1) / p )); return; }; nproc; }
# the memory this container really has, in MiB: the host's MemTotal, or the control group's limit (v2, then v1) where that is lower
ram_mib() { local t l c=${CGROOT:-/sys/fs/cgroup}; t=$(awk '/^MemTotal:/ { print int($2 / 1024) }' ${MEMINFO:-/proc/meminfo} 2> /dev/null); t=${t:-0}
            for f in $c/memory.max $c/memory/memory.limit_in_bytes; do l=$(cat $f 2> /dev/null); case $l in ''|max|*[!0-9]*) ;; *) l=$(( l / 1048576 )); [ "$l" -lt "$t" ] && t=$l; break;; esac; done; echo $t; }
CPUS=$(cpus); export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS
RAM=$(ram_mib); NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
GPU_NAME=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | head -1); GPU_MIB=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
free -g > $R/free.txt 2>&1
{ TZ=EST5EDT date; nvidia-smi -L; nvidia-smi -i 0; nvcc --version | tail -2; echo "arch sm_$ARCH; a pod of $NGPU GPUs, the job uses GPU 0; $CPUS CPUs of $(nproc); $((RAM / 1024)) GiB of memory (host or control group)"; cat $R/free.txt; df -h ~ | tail -1; cat $SRC/COMMIT 2> /dev/null; } > $R/machine.txt 2>&1
echo "pod: $NGPU GPU(s) ($GPU_NAME, ${GPU_MIB:-?} MiB); the job uses GPU 0 alone (CUDA_VISIBLE_DEVICES=0)" >> $P
# the level follows the card: wren on a 24 or 32 GB one (the cache it has), linnet on a 48 GB one (the caches of a 24 and a 32 GB card); Hummingbird (HB=1) on any card of 15 to 40 GB with the cache it has, on a 48 GB one with three;
# NEED_CPU: the memory with the CPU's half (wren and linnet: the experts' first matrices again; Hummingbird: nothing, the CPU reads the places of the host tier where they are)
if [ "$HB" = 1 ]; then
    LAYOUT=hummingbird; LEVEL=hummingbird; EXPERTS="39.2"; NEED=52; NEED_CPU=52
    if [ "${GPU_MIB:-0}" -ge 45000 ]; then CAPS="24g:auto:23 32g:auto:31 48g:auto"; RAMCAPS="24g:auto:23"   # (a 48 GB card holds the model whole: the 32 GB computer's card is a 24 GB one)
    elif [ "${GPU_MIB:-0}" -ge 15000 ] && [ "${GPU_MIB:-0}" -lt 40000 ]; then CAPS="card:auto"; RAMCAPS="card:auto"
    else echo "gpu: $GPU_NAME, ${GPU_MIB:-?} MiB: the Hummingbird job runs on a card of 15 to 40 GB (the cache it has) or of 48 GB (the caches of a 24 and a 32 GB card): stopped" >> $P; exit 1; fi
    [ -n "$PACKED_S3$EXPORT_S3" ] || [ -f "$EXPORT_DIR/meta.json" ] || { echo "hb: HB=1 and none of PACKED_S3, EXPORT_S3, or an export in EXPORT_DIR ($EXPORT_DIR/meta.json): stopped" >> $P; exit 1; }
elif [ "${GPU_MIB:-0}" -ge 45000 ]; then
    LAYOUT=i5; LEVEL=linnet; NEED=88; NEED_CPU=124; EXPERTS="72.2"; CAPS="24g:auto:23 32g:auto:31 48g:auto"
elif [ "${GPU_MIB:-0}" -ge 23000 ] && [ "${GPU_MIB:-0}" -lt 40000 ]; then
    LAYOUT=i4; LEVEL=wren; NEED=72; NEED_CPU=102; EXPERTS="59.1"; CAPS="card:auto"
else
    echo "gpu: $GPU_NAME, ${GPU_MIB:-?} MiB: this job runs on a card of 24 or 32 GB (wren, the cache the card has) or of 48 GB (linnet, the caches of a 24 and a 32 GB card): stopped" >> $P; exit 1
fi
# the guards, before anything is downloaded or built: the pack is bound by the CPUs (about 5.7 MB/s of bf16 a CPU: 241 GB takes about 59 min on 12), the experts are pinned in the pod's memory with the engine and the
# system beside them, and the nvcc has to know the card (compute_120 is CUDA 12.8's)
MINCPUS=12; [ "$HB" = 1 ] && MINCPUS=8   # (the Hummingbird pack fits nothing: it reads the six-bit matrices' 16 GB of bf16 and writes the experts' codes as they are)
[ "$CPUS" -ge $MINCPUS ] || { echo "cpus: $CPUS, too few to pack in time: stopped (launch again for another host)" >> $P; exit 1; }
[ "$RAM" -ge $((NEED * 1024)) ] || { echo "ram: $((RAM / 1024)) GiB ($(free -g | awk '/^Mem:/ { printf "free -g: total %s, used %s, free %s, available %s", $2, $3, $4, $7 }')), and the experts at $LEVEL are $EXPERTS GiB pinned ($NEED GiB with the engine and the system): stopped (launch again with RP_GPUS=2 or 3, a pod of more GPUs having more of the host's memory, or for another host)" >> $P; exit 1; }
command -v nvcc > /dev/null && nvcc --list-gpu-arch | grep -qx "compute_$ARCH" || { echo "nvcc: none, or none that knows compute_$ARCH (the RTX 5090's sm_120 wants CUDA 12.8 or later): stopped" >> $P; exit 1; }
# the CPU's half takes the experts' first matrices again (not pinned): a pod that holds the experts and not that runs the mode copy alone
if [ "$RAM" -ge $((NEED_CPU * 1024)) ]; then HOSTHALF=1; else echo "ram: $((RAM / 1024)) GiB, short of the $NEED_CPU GiB that the experts and the CPU's half take: only the mode copy is run (launch again with RP_GPUS=2 or 3 for the others)" >> $P; fi
# the CPU's crew: the pod's CPUs less two (the engine's default); the variants of --offload-mode one load of the model measures: copy (the cache's first version), cpu on the default threads, on 8, 16 and 32 where the pod has
# them and kept to cores (:pin), split and auto (also on 16 threads); a 48 GB card's three caches each take copy, cpu and auto
TH=$(( CPUS > 3 ? CPUS - 2 : 1 ))
variant() { case ",$1," in *",$2,"*) echo "$1";; *) echo "$1,$2";; esac; }
if [ $HOSTHALF = 1 ]; then
    if [ "${GPU_MIB:-0}" -ge 45000 ]; then
        SWEEP=$(variant $(variant $SWEEP cpu) auto)
    else
        SWEEP=$(variant $SWEEP cpu); for n in 8 16 32; do [ $n -le $TH ] && SWEEP=$(variant $SWEEP cpu:$n); done
        SWEEP=$(variant $(variant $(variant $SWEEP cpu:$TH:pin) split) auto); [ 16 -le $TH ] && SWEEP=$(variant $(variant $SWEEP split:16) auto:16)
    fi
    SWEEP16=copy,cpu,split,auto
fi
# a card of 24 GB: also the budget of a 16 GB one (RTX 4080: 16,376 MiB, about 15 GiB free once a context is made)
case "$GPU_NAME" in *4090*) CAP16="16g:auto:15"; [ "$HB" = 1 ] && RAMCAPS="$RAMCAPS $CAP16";; esac
echo "variants: $SWEEP (the CPU's threads: $TH)${CAP16:+; the 16 GB budget: $SWEEP16}" >> $P
nvidia-smi -i 0 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > $R/smi.csv 2>&1 &
SMI=$!

# the hub client first: the checkpoint downloads (the pinned revision, 247 GB) while torch installs and the library and the engine build, at the same time (every step has a time limit)
# (three tries 30 s apart: a pod whose first reads of PyPI time out stops here otherwise, as one 5090 pod's did on 2026-10-10)
EXTRA=; [ -n "$PACKED_S3$EXPORT_S3" ] && EXTRA=awscli   # (the S3 prefixes are read with the AWS CLI)
step venv env EXTRA="$EXTRA" bash -c 'python3 -m venv --system-site-packages ~/v/venv && for i in 1 2 3; do ~/v/venv/bin/pip install -q --timeout 60 "huggingface_hub>=1.5,<2" hf_xet $EXTRA && exit 0; sleep 30; done; exit 1' || exit 1
dl() { for i in 1 2 3; do timeout 1800 ~/v/venv/bin/hf download "$REPO" --revision "$REV" --local-dir $M && return 0; done; return 1; }   # (a stalled attempt is cut after 30 min and resumes)
# s3get SRC DST (a prefix to a directory, or a directory to a prefix): `aws s3 cp --recursive`, 64 requests at once in chunks of 64 MB; the keys of ~/s3.env are in the shell of the parentheses alone (not traced, never printed), and the MB/s is a line of steps.txt
s3get() {
    local src=$1 dst=$2 t=$(date +%s) rc bytes secs aws; aws=$(command -v aws || echo ~/v/venv/bin/aws)
    [ -f ~/s3.env ] || { echo "s3get: no ~/s3.env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION): stopped"; return 1; }
    local here=$dst; case $dst in s3://*) here=$src ;; *) mkdir -p "$dst" ;; esac
    $aws configure set default.s3.max_concurrent_requests 64; $aws configure set default.s3.multipart_chunksize 64MB
    ( set +x; set -a; . ~/s3.env 2> /dev/null; set +a; $aws s3 cp --recursive --only-show-errors "$src" "$dst" ); rc=$?   # (the source's own errors are not shown: a line of the keys could be in one)
    secs=$(( $(date +%s) - t )); bytes=$(du -sb "$here" 2> /dev/null | cut -f1)
    echo "s3: $src -> $dst: $(( ${bytes:-0} / 1000000 )) MB in $secs s = $(( ${bytes:-0} / 1000000 / (secs + 1) )) MB/s (rc $rc)" >> $P
    return $rc
}
WATCH_DIR=$M; WATCH_WANT=247000000000; WATCH_SECS=2700   # (what the slow-download guard below measures: the checkpoint on disk 45 minutes in)
EXP=
if [ "$HB" = 1 ] && [ -n "$PACKED_S3" ]; then
    WATCH_DIR=$PK; WATCH_WANT=50000000000; WATCH_SECS=1500   # (a finished directory is about 50 GB: on disk 25 minutes in)
    step fetch-packed s3get "$PACKED_S3" $PK &
    DL=$!
else
    step download dl &
    DL=$!
    if [ "$HB" = 1 ] && [ -n "$EXPORT_S3" ] && [ ! -f "$EXPORT_DIR/meta.json" ]; then
        step fetch-export s3get "$EXPORT_S3" "$EXPORT_DIR" &
        EXP=$!
    fi
fi
( step pyenv timeout -k 10 1500 bash -c '( ~/v/venv/bin/python -c "import torch" 2> /dev/null || ~/v/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 ) && ~/v/venv/bin/pip install -q numpy safetensors tokenizers' ) &
PYENV=$!
( step lib timeout -k 10 2400 bash -c "cd $SRC && bash gpu/build_lib.sh ~/v/lib" ) &
LIB=$!
( step rustup timeout -k 10 600 bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.1'
  . ~/.cargo/env
  export GLYD_ENGINE_ARCHS=$ARCH   # (this GPU alone)
  step engine timeout -k 10 2400 bash -c "cd $SRC && cargo build --release -p glyd-engine $([ "$HB" = 1 ] && echo -p glyd-gpu)" ) &   # (glyd-gpu's program verifies a Hummingbird directory)
ENG=$!
# a slow host stops here, not at the cap: 8 minutes in, the weights must project to be on disk 45 minutes after the start (the builds go on beside it; the size is what the disk has allocated in the model's directory, not what a preallocated file claims)
while kill -0 $DL 2> /dev/null && [ $(( $(date +%s) - T0 )) -lt 480 ]; do sleep 15; done
if kill -0 $DL 2> /dev/null; then
    t=$(( $(date +%s) - T0 )); kb=$(du -sk $WATCH_DIR 2> /dev/null | cut -f1); b=$(( ${kb:-0} * 1024 ))
    [ $(( b / (t + 1) * WATCH_SECS )) -ge $WATCH_WANT ] || { echo "download: $(( b / 1000000 / (t + 1) )) MB/s after $(( t / 60 )) min, too slow for $(( WATCH_WANT / 1000000000 )) GB in $(( WATCH_SECS / 60 )) min: stopped" >> $P; exit 1; }
fi
wait $PYENV
wait $LIB
wait $ENG
B=$SRC/target/release/glyd-engine; PY=~/v/venv/bin/python
export GLYD_GPU_LIB=$(ls ~/v/lib/libglyd_gpu_cuda*.so 2> /dev/null | head -1)
[ -x $B ] || { echo "no engine: see engine.log" >> $P; exit 1; }
[ -n "$GLYD_GPU_LIB" ] || { echo "no library: see lib.log" >> $P; exit 1; }
step versions bash -c "$PY -c 'import torch; print(\"torch\", torch.__version__, torch.version.cuda)'; $B --version"

# 1. the random tiny model at each level, all its experts on the GPU and through the cache, bit for bit (the download goes on beside it), with the CPU computing the pairs of the experts that are not in places (within
# two places of bf16 of the largest logit), and the grouped products over each layout's experts
# (Hummingbird: the tiny model packed at the level from a random export, its experts at 3 and 2 bits, run all there and through the cache with 3, 5 and 8 places a layer of 16, split between the two widths, the misses copied
# and then computed by the CPU from the host tier in place (cpu, split, auto on 4 threads: every logit within two places of bf16 of the largest); then hbcheck)
TINYFAIL=0
if [ "$HB" = 1 ]; then
    T=~/v/tinyhb
    if late; then echo "tinyhb: skipped, late" >> $P; else
        step tinyhb-make timeout -k 10 300 $PY -I $REF moetoy $T --level 6
        step tinyhb-pack timeout -k 10 $(tmo 600) $B hbtoy $T/toy $T/hb --share 40
        step tinyhb-offload timeout -k 10 $(tmo 900) $B nhoffload $T/hb/packed $T/toy --slots 3,5,8 || TINYFAIL=$((TINYFAIL + 1))
        for m in cpu split auto; do
            late || step tinyhb-offload-$m timeout -k 10 $(tmo 900) $B nhoffload $T/hb/packed $T/toy --slots 3,5,8 --offload-mode $m --cpu-threads 4 || TINYFAIL=$((TINYFAIL + 1))
        done
        # (the SSD tier: with 0.0005, 0.0003 and 0.00012 GiB of the tiny model's 0.00065 for the experts, the rest read from the packed directory, in each mode: every logit and token the same to the bit as the run with every
        # expert in memory in that mode)
        for m in copy cpu split auto; do
            late || step tinyhb-ssd-$m timeout -k 10 $(tmo 900) $B nhoffload $T/hb/packed $T/toy --slots 3,5,8 --offload-mode $m --cpu-threads 4 --host-memory 0.0005,0.0003,0.00012 || TINYFAIL=$((TINYFAIL + 1))
        done
        mkdir -p $R/tinyhb; cp $T/toy/toy.txt $R/tinyhb/ 2> /dev/null
    fi
    late || step hbcheck timeout -k 10 $(tmo 900) $B hbcheck --speed
    save
fi
for lv in 6 5 4; do
    [ "$HB" = 1 ] && break
    late && { echo "tiny$lv: skipped, late" >> $P; continue; }
    T=~/v/tiny$lv
    step tiny$lv-make timeout -k 10 300 $PY -I $REF moetoy $T --level $lv
    step tiny$lv-offload timeout -k 10 $(tmo 900) $B nhoffload $T/toy $T/toy --slots 3,5,8 || TINYFAIL=$((TINYFAIL + 1))
    # (the same with the experts' second matrices in the block layout: the dense product, as the cache ran before the sparse one)
    late || step tiny$lv-offload-dense env GLYD_ENGINE_DENSE_DOWN=1 timeout -k 10 $(tmo 900) $B nhoffload $T/toy $T/toy --slots 3,5,8 || TINYFAIL=$((TINYFAIL + 1))
    # (and with the CPU's half: the pairs of a miss computed on 4 threads, or copied as the planner says)
    for m in cpu split auto; do
        late || step tiny$lv-offload-$m timeout -k 10 $(tmo 900) $B nhoffload $T/toy $T/toy --slots 3,5,8 --offload-mode $m --cpu-threads 4 || TINYFAIL=$((TINYFAIL + 1))
    done
    mkdir -p $R/tiny$lv; cp $T/toy/toy.txt $R/tiny$lv/ 2> /dev/null
    save
done
[ "$HB" = 1 ] || late || step moecheck timeout -k 10 $(tmo 900) $B moecheck relu2-22of512 relu2-rows-22of512
late || step cpuexperts timeout -k 10 $(tmo 300) $B cpuexperts --bits 4 --threads $TH
# a check of the tiny model that failed is a fault of the code, found in seconds: the pack (the longest step) and the model's runs are not paid for (the logs and summary.txt say which)
[ "$TINYFAIL" -eq 0 ] || { echo "tiny: $TINYFAIL of the nhoffload runs failed (the experts through the cache, copied or computed by the CPU, against every expert on the GPU): stopped before the pack" >> $P; exit 1; }

wait $DL || { echo "download: failed" >> $P; exit 1; }
[ -z "$EXP" ] || wait $EXP || echo "fetch-export: failed (see fetch-export.log)" >> $P
df -h ~ > $R/disk-after-download.txt 2>&1

# 3. the pack: the whole checkpoint at the card's level ($LEVEL, --layout $LAYOUT; at wren the experts in four-bit codes and every other matrix in six-bit ones)
# (Hummingbird: the experts' codes are the export's, packed as they are, the rest six-bit from the checkpoint; with PACKED_S3 the directory came whole, and glyd-gpu verify decodes every pack of it)
if [ "$HB" = 1 ]; then
    if [ -z "$PACKED_S3" ]; then
        if [ -f "$EXPORT_DIR/meta.json" ]; then
            late || step pack timeout -k 10 $(tmo 4200) $B pack $M $PK --layout hummingbird --codes "$EXPORT_DIR" --threads $CPUS
        else
            echo "pack: no export ($EXPORT_DIR/meta.json is not there, and EXPORT_S3 gave none): the steps that need the packed directory are skipped" >> $P
        fi
    fi
    VERIFIED=0; [ -f $PK/glyd.json ] && { late || { step verify timeout -k 10 $(tmo 2400) $SRC/target/release/glyd-gpu verify $PK --threads $CPUS && VERIFIED=1; }; }
    if [ "$VERIFIED" = 1 ] && [ -z "$PACKED_S3" ] && [ -n "$PACKED_UP" ]; then step push-packed s3get $PK "$PACKED_UP" & UP=$!; fi   # (only a directory verified here; the runs go on beside the upload)
else
    late || step pack timeout -k 10 $(tmo 4200) $B pack $M $PK --layout $LAYOUT --threads $CPUS
fi
PACK_S=${LAST_S:-4200}
if [ -f $PK/glyd.json ]; then
    { du -sb $PK; ls $PK | head -40; } > $R/pack.txt 2>&1
else
    echo "pack: no glyd.json: the steps that need the packed directory are skipped" >> $P
fi
save

# 4. generate with each cache, 256 greedy tokens of a prompt, on one load of the model for all the variants of --offload-mode: the speed of the decode steps and the cache's counts
PROMPT="Explain, step by step, how a hash table handles collisions, then compare open addressing with chaining, and say when each one is the better choice for a program that reads far more than it writes."
if [ -f $PK/glyd.json ]; then
    for c in $CAPS; do
        late && { echo "generate-${c%%:*}: skipped, late" >> $P; continue; }
        # (a load of the model is 5 minutes and a variant about a minute; the 48 GB card's caches carry three variants each)
        step generate-${c%%:*} timeout -k 10 $(tmo 2700) $B generate $PK "$PROMPT" --new 256 --ignore-eos --ctx 4096 --expert-cache ${c#*:} --offload-mode $SWEEP --cpu-threads $TH
        save
    done
    LASTCAP=${CAPS##* }
    BEST=$(best $R/generate-${LASTCAP%%:*}.log); BEST=${BEST:-auto}
    # 4b. a 16 GB card's budget on the 4090: the plan sees about 15 GiB free (the places, or the refusal of a card too small for the rest of the model and the least cache, are in the plan line of the log)
    if [ -n "$CAP16" ]; then
        late && echo "generate-${CAP16%%:*}: skipped, late" >> $P || { step generate-${CAP16%%:*} timeout -k 10 $(tmo 2400) $B generate $PK "$PROMPT" --new 256 --ignore-eos --ctx 4096 --expert-cache ${CAP16#*:} --offload-mode $SWEEP16 --cpu-threads $TH; save; }
    fi
    # 4c. the whole model's logits through the cache in the fastest variant of step 4, against the bf16 checkpoint (as nh_run_check_job.sh step 5: 64 tokens prefilled, 96 teacher-forced steps)
    IFS=: read -r BM BT BP <<< "$BEST"
    LV=$([ "$LAYOUT" = i5 ] && echo 5 || echo 4)
    TOKDIR=$M; [ -f $M/tokenizer.json ] || TOKDIR=$PK   # (a finished Hummingbird directory came without the checkpoint: the tokenizer is in it)
    late || {
        step ids timeout -k 10 300 $PY -I $REF ids $TOKDIR ~/v/logits --prefill 64 --total 160 --level $LV
        if [ "$HB" = 1 ]; then
            # (the Hummingbird level is lossy against bf16 whatever serves the misses: the CPU's sums are held to the same run with the experts copied, the cache's first version)
            mkdir -p ~/v/logits-copy; cp ~/v/logits/toy.txt ~/v/logits/ids.txt ~/v/logits-copy/ 2> /dev/null
            late || step logits-run-copy timeout -k 10 $(tmo 3000) $B nhtoy $PK ~/v/logits-copy --expert-cache auto --offload-mode copy
        fi
        late || step logits-run timeout -k 10 $(tmo 3000) $B nhtoy $PK ~/v/logits --expert-cache auto --offload-mode ${BM:-auto} ${BT:+--cpu-threads $BT} ${BP:+--cpu-pin}
        if [ "$HB" = 1 ]; then late || step logits-compare timeout -k 10 300 $PY -I $REF compare ~/v/logits ~/v/logits-copy; fi
        if [ -f $M/model.safetensors.index.json ]; then late || step logits-check timeout -k 10 $(tmo 3300) $PY -I $REF full $M ~/v/logits; fi
        mkdir -p $R/logits; cp ~/v/logits/toy.txt ~/v/logits/ids.txt ~/v/logits/toy_*_tokens.txt $R/logits/ 2> /dev/null
        save
    }
    # 4d. 32 GB RAM (Hummingbird): the pod's disk as the SSD tier reads it (diskcheck), then step 4's run on one load with --host-memory 20 (20 GiB of the experts in the computer's memory, the other 19 GiB read from the packed
    # directory as the router asks), in the modes cpu and auto, for the card's own cache and (the 4090) the 16 GB budget's; the experts read from the disk a token, the peak resident memory and the plan line are in the log
    if [ "$HB" = 1 ]; then
        late || step diskcheck timeout -k 10 $(tmo 600) $B diskcheck $PK --reads 600 --depth 16
        for c in $RAMCAPS; do
            late && { echo "generate-ram20-${c%%:*}: skipped, late" >> $P; continue; }
            step generate-ram20-${c%%:*} timeout -k 10 $(tmo 2400) $B generate $PK "$PROMPT" --new 256 --ignore-eos --ctx 4096 --expert-cache ${c#*:} --host-memory 20 --offload-mode cpu,auto --cpu-threads $TH
            save
        done
    fi
    # 5. the way a user starts it (the cache the plan gives this card)
    late || step run-prompt timeout -k 10 $(tmo 1500) $B run $PK --prompt "What is the capital of France? Answer in one short sentence." --no-think
    # 6. the same caches with the experts' second matrices dense: the model packed again in the block layout (GLYD_ENGINE_DENSE_DOWN=1) and step 4's runs on it, in the mode copy, if the time left holds another pack (its time
    # and a fifth more), a run of generate for each cache (10 minutes) and the 8 minutes the job keeps
    PKD=~/v/packed-dense
    NCAP=$(set -- $CAPS; echo $#)
    if [ "$HB" = 1 ]; then
        echo "dense: the Hummingbird level has no dense second product (its experts' down matrices are rows): not run" >> $P
    elif [ "$(left)" -gt $(( PACK_S * 12 / 10 + NCAP * 600 + 480 )) ]; then
        step pack-dense env GLYD_ENGINE_DENSE_DOWN=1 timeout -k 10 $(tmo 4200) $B pack $M $PKD --layout $LAYOUT --threads $CPUS
        if [ -f $PKD/glyd.json ]; then
            for c in $CAPS; do
                late && { echo "generate-dense-${c%%:*}: skipped, late" >> $P; continue; }
                step generate-dense-${c%%:*} timeout -k 10 $(tmo 1500) $B generate $PKD "$PROMPT" --new 256 --ignore-eos --ctx 4096 --expert-cache ${c#*:} --offload-mode copy
                save
            done
        else
            echo "pack-dense: no glyd.json: the dense runs are skipped" >> $P
        fi
    else
        echo "dense: not run, the $(( $(left) / 60 )) minutes left are short of another pack ($(( PACK_S / 60 )) min) and the runs" >> $P
    fi
fi
[ -z "${UP:-}" ] || wait $UP || echo "push-packed: failed (see push-packed.log)" >> $P   # (before fin removes the keys)
save
cat $R/summary.txt 2> /dev/null | head -260
nvidia-smi -i 0 > $R/gpu-final.txt 2>&1
