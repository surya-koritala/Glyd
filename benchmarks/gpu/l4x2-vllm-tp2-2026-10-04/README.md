# Tensor parallel 2 with the embedding and the output layer packed: two L4s (2026-10-04)

`vllm serve --quantization glyd` over two GPUs (vLLM's tensor parallel, 2 ranks) with the embedding and the LM head (the output layer) packed, checked on two NVIDIA L4s of an AWS g6.12xlarge (4 x L4, 48 vCPUs; driver 595.91.07), vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.18.0. Under tensor parallel vLLM splits the embedding's rows and the output layer's rows over the ranks (here 16,032 rows each of the tiny models' table, which vLLM pads from 32,001 to 32,064 rows, and 75,968 of Qwen3-4B's 151,936): each rank packs its own share.
The code is the whole-model packing as of the run's day, before the release's last merges; the check was not repeated on the final tree.
This is a check, not a benchmark: it says the packed tables and the packed layers are right over two GPUs, and makes no claim about speed or memory (the weights `check_vllm` prints for its Glyd runs are of a run with `verify` on).

## Two tiny random models over 2 GPUs (`tables_check.py --tp 2`)

A tied model (`to`, one table) and an untied one (`uo`, two), a vocabulary of 32,001 rows (vLLM pads it to 32,064): **all passed (8 checks)** (`tables/report.txt`):

- the embedding and the LM head packed (1 pack for the tied model, 2 for the untied; shard rows 16,032), every lookup the checkpoint's row bit for bit on 2 ranks, each product within 1e-2 (2.8e-03 tied, 3.0e-03 untied) and the same bits twice;
- the first token of every prompt bf16's where bf16's margin is over 0.1 (3 of 4 tied, 1 of 4 untied);
- prompt logprobs within 0.0065 (tied) and 0.0071 (untied) on average of bf16's (at most 0.05; the largest 0.024 and 0.027);
- `exact`: bf16 eager's tokens, logprobs and prompt logprobs bit for bit, the tied model's table as vLLM runs it and the untied model's embedding packed with the head as vLLM runs it.

## Qwen3-4B (tied) over 2 GPUs (`check_vllm.py --brief --tp 2`)

**All passed (7 checks)** (`real/report.txt`):

- the embedding and the LM head packed (1 pack, 250 MiB), every lookup of random ids and of the shard's ends the checkpoint's own row bit for bit, each product within 1e-2 of its matrix (3.04e-03) and the same bits twice;
- every pack (288) decoded to its weights bit for bit; each layer's product within 1e-2 of `F.linear` (5.61e-03) and the same bits every run;
- top-1 agreement with bf16 on its continuation 0.9929 (at least 0.97; bf16 eager's own 0.9942), the mean |logprob difference| 7.29e-03, at most 2.0x bf16 eager's against its graphs (8.50e-03);
- `exact`, eager: bf16 eager's tokens, logprobs and prompt logprobs bit for bit (8 of 8; continuation True), the tied model's table as vLLM runs it.

## Files

`tables/report.txt` and `real/report.txt` (the checks above), `steps.txt` (the steps of the job with their times: 756 s in all), `env.txt`, `machine-short.txt`.
