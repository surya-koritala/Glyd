# Open models, 2026-09-27: Lambda Cloud A10 (24 GB)

`sizes.txt`: every projection's matrix packed in both layouts and unpacked, compared bit for bit
(`gpu/sizes.py`), for the nine models tracked since this date. Four come from the first pass
(`run.sh`, `sizes-first-pass.txt`), where their count is the same under both rules; five were
measured again in a second pass (`run2.sh`) with the rule of c914521 (a layer's experts kept as one
tensor counted a matrix an expert; `in_proj_qkvz` and the like counted), which the first pass had
missed for Gemma 4 and Llama 4 Scout (experts) and Qwen3.8 and Qwen3-Next (linear-attention inputs).

`loaders.txt`: the class transformers 5.17 builds for each checkpoint. `quality-*.txt`: bf16 against
Glyd end to end (`e2e.py --format auto --fused --baseline --merge --batch 1,8,32 --ppl enwik8
--mmlu 300`) for the two that fit this GPU in bf16. `fixtest-gemma-3-4b-it.txt`: the same for Gemma
3 4B with e2e.py at c914521 (Gemma's decoder and scaled embedding); its perplexity is near 720 for
both because the windows start without the BOS token Gemma needs.

Llama and Gemma 3 from `unsloth/` (the same weights, ungated).
