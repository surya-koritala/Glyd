# Kimi-K3 on real weights, 1x A100 40 GB (Lambda us-west-2), 2026-10-21

The first 4 layers of moonshotai/Kimi-K3 (f831ab66), sliced from the Hub (`hubslice --layers 4`, 57.6 GB in 271 s with 64 threads; 8 threads ran at 18 MB/s, GLY-356). Layers 0-2 are delta attention, layer 3 is latent attention, layer 0 is dense, layers 1-3 have the 896 MXFP4 experts. The checks compare the dynamic engine to the CPU reference (src/kimi_ref.rs). The expert cache holds 128 or 16 places a layer, since 40 GB can't hold 3 x 15.7 GB of experts.

out-1 (13:52-14:45 EDT): every logit NaN. Cause: Kimi-K3 releases the short-conv weights and o_norm as F32, and the engine read their bytes as bf16. kimilayers reported a difference of 0 because `f32::max` skipped the NaN. Fixed in a3ec2cbd: fp32-weight kernels, and NaN counted as the largest difference.

out-2 (14:58-15:38 EDT), after the fix:

| check | result |
|---|---|
| kimilayers, X after layer L (6 tokens) | layer 0: 0.0002-0.0010 of 0.21; layer 1: 0.0005-0.0020, one token 0.0106 of 0.153; layer 2: 0.0020-0.0039 of 1.4-2.1; layer 3 (MLA): 0.0020-0.0039 |
| kimicheck, 128 places, 16 steps | 12 of 16 greedy tokens the reference's; 16-token prompt 0.094 largest logit difference, same top token |
| kimicheck, 16 places, 8 steps | 7 of 8 |

The decode steps that differ are the ones with an expert near-tie (median closest choice 0.00008, against 0.00018 for the others). Cost about $3.50 for both sessions.
