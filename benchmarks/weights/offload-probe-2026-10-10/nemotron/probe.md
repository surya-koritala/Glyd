## Expert-offload probe: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 @ 2dc98e2a

The bf16 model as released, nothing rounded. Held-out text: 24 sequences of 2048 tokens (49,152 tokens, 6 domains) through 88 blocks; 1,081,344 routed (token, expert) pairs in each of the 40 MoE layers measured (40 in the model). The neuron profile of each expert and the routing counts that rank the experts come from 72 calibration sequences of the same domains, taken in the same pass. Decode: 1260 one-token steps of 20 greedy decodes of 64 new tokens (thinking off).

How to read it:
- **Pair.** One token routed to one expert of one MoE layer (22 a token a layer). Sparsity and predictor numbers are over pairs, look-ahead numbers over tokens. p10, p50, p90 are percentiles: exact for counts (neurons, columns, experts named), the upper edge of a bin 0.5% wide for a ratio (the predictor's recall, the weighted look-ahead).
- **Zero.** Neurons of the expert's 2,688 that are exactly 0 after relu2 for that pair, in the bf16 the model computes. A zero neuron needs no column of down.
- **Kept at 1% / 0.1%.** Columns of down kept, the largest activation first, for the expert's output to be within that relative L2 error of the whole one (float32 sums); zero neurons are not counted. The activations are known when it is counted: it is what the best choice of columns needs, not what a rule that finds them without computing up first keeps. 0.1% is the size of bf16's own rounding of an output (0.11% rms).
- **Bytes.** At 4 bits a weight an expert is 2,752,512 B, down 1,376,256 B. Down read = the mean kept share of down; 'both' = the same neurons' rows of up too, which needs the active set before up is read.
- **Predicted.** The B most frequent active neurons of the expert in the calibration text. Recall = share of the pair's active neurons among them; error = relative L2 error of the output with only those B neurons computed; oracle = the same for the pair's B largest activations (no predictor does better); within = share of pairs with error at most 1%.
- **Look-ahead.** The router of MoE layer N (its pre-norm, gate, correction bias and top-K choice: the model's own modules) on the stream at the input of block N - d; d = 0 is the model's own choice and must name every expert. 'At previous' = the stream at the input of the previous MoE layer, 'two earlier' = the input of the block two before that. Recall = share of the K experts the model chose that it names, weighted = share of their routing weight; x1.5, x2, x3 = the first 33, 44, 66 experts of its ranking. Random = m / 512; popular = the m most used experts of the layer in the calibration text, one fixed set.
- **Importance.** Weight = the routing weight (sigmoid scores renormalised over the 22, times the routed scale). Output = L2 norm of the expert's 1,024-wide output before the weight. Weight x output = the mean over the expert's pairs of the two. Contribution = L2 norm of what it adds to the block's 4,096-wide output, fc2_latent_proj(weight x output). Share = contribution over the sum of the token's 22 contributions. Hot = a layer's 13% most used experts by calibration counts, cold = its 50% least used (the quality job's mix13 and cold50 sets). Spearman = rank correlation, across a layer's experts, of calibration frequency with the mean over the pairs the expert got.

### Summary, held-out text

- Zero after relu2: 91.8% of an expert's neurons on average (p10 83.7%, p90 98.5%).
- Columns of down kept for 1.0% error: 4.4% on average (p10 0.9%, p90 9.2%); at 4 bits that reads 60,792 B of an expert's 1,376,256 B of down and saves 47.8% of its 2,752,512 B, or 2,630,927 B if the rows of up of the same neurons could be skipped too.
- Columns of down kept for 0.1% error: 5.6% on average (p10 1.2%, p90 11.6%); at 4 bits that reads 77,318 B of an expert's 1,376,256 B of down and saves 47.2% of its 2,752,512 B, or 2,597,876 B if the rows of up of the same neurons could be skipped too.
- Naming the active neurons from the calibration profile: 168 neurons (6.25%): recall 22.4%, output error 82.13% (oracle 1.34%); 336 neurons (12.5%): recall 34.7%, output error 73.30% (oracle 0.10%); 672 neurons (25%): recall 52.7%, output error 59.76% (oracle 0.00%); 1344 neurons (50%): recall 76.7%, output error 37.99% (oracle 0.00%); 2016 neurons (75%): recall 91.9%, output error 19.16% (oracle 0.00%).
- Look-ahead, recall of the 22 experts chosen: at the previous MoE layer's input 68.7% (p10 50.0%, weighted 74.5%); two blocks earlier 59.2% (p10 36.4%, weighted 65.8%); the router at d = 0 names 100.0% of tokens whole.
- Importance: a cold expert's mean weight x output is 1.00 of a hot one's, its mean contribution 1.00; Spearman of calibration frequency with the mean weight x output -0.12, with the mean contribution -0.12.

### Held-out text: relu2 sparsity inside the experts (43,253,760 pairs, 40 layers)

| share of the expert's neurons, % | mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| zero (exactly 0 after relu2) | 91.8 | 83.7 | 94.1 | 98.5 |
| columns of down kept for 1.0% error | 4.4 | 0.9 | 3.5 | 9.2 |
| columns of down kept for 0.1% error | 5.6 | 1.2 | 4.5 | 11.6 |

| bytes of an expert read at 4 bits (mean over pairs) | kept for 1.0% | kept for 0.1% |
|---|---:|---:|
| down, columns kept (B) | 60,792 | 77,318 |
| down, saved (B) | 1,315,464 | 1,298,938 |
| saved, share of the expert's bytes (%) | 47.8 | 47.2 |
| up and down, if the neurons were named first, saved (B) | 2,630,927 | 2,597,876 |

By layer (%):

| MoE layer (block) | zero mean | zero p10 | zero p50 | zero p90 | kept 1% mean | kept 1% p10 | kept 1% p50 | kept 1% p90 | kept 0.1% mean | kept 0.1% p10 | kept 0.1% p50 | kept 0.1% p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (1) | 99.4 | 98.2 | 99.8 | 100.0 | 0.3 | 0.0 | 0.0 | 0.6 | 0.3 | 0.0 | 0.0 | 0.8 |
| 1 (3) | 50.8 | 42.8 | 50.8 | 58.7 | 2.0 | 1.0 | 1.8 | 3.1 | 2.9 | 1.7 | 2.7 | 4.4 |
| 2 (5) | 78.0 | 71.0 | 78.0 | 85.3 | 2.3 | 1.3 | 2.2 | 3.5 | 3.3 | 1.9 | 3.2 | 4.8 |
| 3 (8) | 96.3 | 93.8 | 96.7 | 98.3 | 2.1 | 0.9 | 1.9 | 3.6 | 2.9 | 1.3 | 2.6 | 4.7 |
| 4 (10) | 96.6 | 94.1 | 97.0 | 98.7 | 2.1 | 0.9 | 1.9 | 3.8 | 2.8 | 1.2 | 2.5 | 4.9 |
| 5 (12) | 97.2 | 95.0 | 97.6 | 99.0 | 1.8 | 0.6 | 1.6 | 3.3 | 2.4 | 0.9 | 2.0 | 4.3 |
| 6 (14) | 87.9 | 83.3 | 88.8 | 92.2 | 5.7 | 3.6 | 5.5 | 8.1 | 7.7 | 5.1 | 7.4 | 10.7 |
| 7 (17) | 96.2 | 93.7 | 96.8 | 98.5 | 2.3 | 1.0 | 2.0 | 4.0 | 3.0 | 1.3 | 2.6 | 5.2 |
| 8 (19) | 92.7 | 88.8 | 93.3 | 96.1 | 4.4 | 2.4 | 4.1 | 6.8 | 5.7 | 3.2 | 5.4 | 8.9 |
| 9 (21) | 99.1 | 98.3 | 99.3 | 99.8 | 0.5 | 0.1 | 0.4 | 1.0 | 0.6 | 0.1 | 0.5 | 1.3 |
| 10 (23) | 97.0 | 94.2 | 97.5 | 99.1 | 2.0 | 0.6 | 1.6 | 3.9 | 2.6 | 0.7 | 2.1 | 4.9 |
| 11 (26) | 85.4 | 78.8 | 86.1 | 91.3 | 8.7 | 5.3 | 8.4 | 12.6 | 11.2 | 6.9 | 10.8 | 16.0 |
| 12 (28) | 92.5 | 88.4 | 93.1 | 95.9 | 5.0 | 2.7 | 4.6 | 7.9 | 6.4 | 3.5 | 5.9 | 10.0 |
| 13 (30) | 90.1 | 85.4 | 90.5 | 94.2 | 6.7 | 3.9 | 6.4 | 9.9 | 8.5 | 5.0 | 8.1 | 12.5 |
| 14 (32) | 90.3 | 84.4 | 90.6 | 95.9 | 6.7 | 2.8 | 6.4 | 10.8 | 8.4 | 3.5 | 8.1 | 13.6 |
| 15 (34) | 95.9 | 92.5 | 96.5 | 98.8 | 2.8 | 0.8 | 2.4 | 5.2 | 3.5 | 1.1 | 3.1 | 6.5 |
| 16 (37) | 87.1 | 80.5 | 87.6 | 93.3 | 9.0 | 4.5 | 8.6 | 13.8 | 11.3 | 5.8 | 10.8 | 17.2 |
| 17 (39) | 85.0 | 79.2 | 85.4 | 90.4 | 10.5 | 6.5 | 10.2 | 14.7 | 13.1 | 8.3 | 12.8 | 18.3 |
| 18 (41) | 93.9 | 89.6 | 94.5 | 97.2 | 4.2 | 1.9 | 3.8 | 7.2 | 5.3 | 2.4 | 4.8 | 9.0 |
| 19 (43) | 89.8 | 84.0 | 90.4 | 94.6 | 7.1 | 3.7 | 6.6 | 11.2 | 8.9 | 4.7 | 8.4 | 14.0 |
| 20 (45) | 84.2 | 78.2 | 84.6 | 89.4 | 11.1 | 7.2 | 10.8 | 15.5 | 13.9 | 9.2 | 13.4 | 19.2 |
| 21 (48) | 93.8 | 88.7 | 94.6 | 97.7 | 4.2 | 1.5 | 3.6 | 7.8 | 5.4 | 2.0 | 4.6 | 9.8 |
| 22 (50) | 89.3 | 84.2 | 89.8 | 93.8 | 7.4 | 4.2 | 7.0 | 11.1 | 9.3 | 5.4 | 8.8 | 13.8 |
| 23 (52) | 93.0 | 89.1 | 93.5 | 96.4 | 4.8 | 2.5 | 4.5 | 7.6 | 6.1 | 3.1 | 5.6 | 9.5 |
| 24 (54) | 91.7 | 87.6 | 92.1 | 95.4 | 5.7 | 3.1 | 5.4 | 8.6 | 7.2 | 4.0 | 6.8 | 10.8 |
| 25 (56) | 91.9 | 87.5 | 92.3 | 95.8 | 5.6 | 2.9 | 5.2 | 8.7 | 7.1 | 3.7 | 6.7 | 10.9 |
| 26 (59) | 95.0 | 90.8 | 95.6 | 98.4 | 3.5 | 1.1 | 3.0 | 6.5 | 4.4 | 1.4 | 3.8 | 8.1 |
| 27 (61) | 93.3 | 88.2 | 94.0 | 97.4 | 4.7 | 1.8 | 4.1 | 8.3 | 5.9 | 2.3 | 5.2 | 10.3 |
| 28 (63) | 93.6 | 88.8 | 94.1 | 97.6 | 4.5 | 1.6 | 4.0 | 7.9 | 5.6 | 2.1 | 5.1 | 9.9 |
| 29 (65) | 94.7 | 90.0 | 95.3 | 98.6 | 3.7 | 1.0 | 3.2 | 7.1 | 4.6 | 1.2 | 4.0 | 8.8 |
| 30 (67) | 95.3 | 91.0 | 95.9 | 98.9 | 3.3 | 0.7 | 2.8 | 6.3 | 4.1 | 0.9 | 3.6 | 7.8 |
| 31 (70) | 95.1 | 90.9 | 95.6 | 98.8 | 3.4 | 0.8 | 3.0 | 6.4 | 4.3 | 1.0 | 3.8 | 8.0 |
| 32 (72) | 94.9 | 91.1 | 95.3 | 98.5 | 3.5 | 1.0 | 3.3 | 6.2 | 4.4 | 1.3 | 4.1 | 7.8 |
| 33 (74) | 95.2 | 91.7 | 95.5 | 98.7 | 3.3 | 0.9 | 3.1 | 5.8 | 4.2 | 1.1 | 3.9 | 7.3 |
| 34 (76) | 96.8 | 94.3 | 97.1 | 98.9 | 2.2 | 0.8 | 2.0 | 3.9 | 2.8 | 1.0 | 2.5 | 5.0 |
| 35 (79) | 96.3 | 93.5 | 96.7 | 98.8 | 2.6 | 0.9 | 2.3 | 4.5 | 3.2 | 1.1 | 2.9 | 5.7 |
| 36 (81) | 96.4 | 93.4 | 96.9 | 98.9 | 2.5 | 0.8 | 2.1 | 4.6 | 3.1 | 1.0 | 2.7 | 5.8 |
| 37 (83) | 96.6 | 93.5 | 97.1 | 99.0 | 2.4 | 0.7 | 2.0 | 4.5 | 3.0 | 0.9 | 2.5 | 5.7 |
| 38 (85) | 95.1 | 91.6 | 95.6 | 97.8 | 3.4 | 1.5 | 3.0 | 5.8 | 4.3 | 1.9 | 3.8 | 7.3 |
| 39 (87) | 87.3 | 81.0 | 87.8 | 93.0 | 8.8 | 4.7 | 8.3 | 13.3 | 11.0 | 6.0 | 10.5 | 16.6 |

#### Held-out text: naming the active neurons before up is computed

| neurons named | of the expert's | recall mean, % | recall p10, % | error mean, % | within 1%, % | oracle error mean, % | oracle within 1%, % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 168 | 6.25 | 22.4 | 10.5 | 82.13 | 1.7 | 1.34 | 75.8 |
| 336 | 12.5 | 34.7 | 19.5 | 73.30 | 1.9 | 0.10 | 96.9 |
| 672 | 25 | 52.7 | 36.0 | 59.76 | 2.4 | 0.00 | 100.0 |
| 1344 | 50 | 76.7 | 63.0 | 37.99 | 4.6 | 0.00 | 100.0 |
| 2016 | 75 | 91.9 | 84.5 | 19.16 | 13.9 | 0.00 | 100.0 |

#### Held-out text: look-ahead routing

| d (blocks) | tokens | recall x1, mean % | recall x1.5, mean % | recall x2, mean % | recall x3, mean % | recall x1, p10 % | weighted x1, mean % | weighted x1, p10 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1966080 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1 | 1966080 | 74.5 | 86.2 | 91.1 | 95.2 | 54.5 | 79.1 | 59.0 |
| 2 | 1916928 | 69.6 | 81.7 | 87.7 | 93.3 | 50.0 | 75.3 | 55.0 |
| 3 | 1916928 | 62.7 | 74.2 | 80.8 | 87.8 | 40.9 | 68.9 | 44.5 |
| 4 | 1867776 | 59.6 | 71.0 | 77.8 | 85.6 | 36.4 | 66.3 | 42.0 |
| 5 | 1867776 | 55.5 | 66.2 | 73.1 | 81.3 | 31.8 | 62.2 | 36.0 |
| 6 | 1818624 | 52.8 | 63.3 | 70.3 | 78.9 | 31.8 | 59.7 | 33.0 |
| at previous MoE layer (d = 2, 3) | 1916928 | 68.7 |  |  |  | 50.0 | 74.5 | 54.0 |
| two blocks earlier (d = 4, 5) | 1867776 | 59.2 |  |  |  | 36.4 | 65.8 | 41.5 |
| random |  | 4.3 | 6.4 | 8.6 | 12.9 |  |  |  |
| popular |  | 9.5 | 13.2 | 16.7 | 23.1 |  |  |  |

By layer, for the 22 experts chosen (recall and weighted recall, %; L = the previous MoE layer, d = blocks back from the predicted layer N):

| predicted MoE layer N (block) | L (block) | d | at previous: recall mean | p10 | weighted | d | two earlier: recall mean | p10 | weighted | popular 22 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (1) | - | - | - | - | - | - | - | - | - | 6.0 |
| 1 (3) | 1 | 2 | 49.5 | 27.3 | 52.5 | - | - | - | - | 7.7 |
| 2 (5) | 3 | 2 | 58.5 | 40.9 | 65.0 | 4 | 45.4 | 22.7 | 52.2 | 6.9 |
| 3 (8) | 5 | 3 | 57.5 | 40.9 | 67.0 | 5 | 45.3 | 27.3 | 55.4 | 6.5 |
| 4 (10) | 8 | 2 | 65.3 | 50.0 | 74.1 | 4 | 57.0 | 40.9 | 67.0 | 6.4 |
| 5 (12) | 10 | 2 | 68.0 | 50.0 | 75.1 | 4 | 52.5 | 31.8 | 61.7 | 6.4 |
| 6 (14) | 12 | 2 | 73.4 | 59.1 | 78.9 | 4 | 62.5 | 45.5 | 69.3 | 8.1 |
| 7 (17) | 14 | 3 | 59.2 | 40.9 | 65.5 | 5 | 52.0 | 31.8 | 58.7 | 7.5 |
| 8 (19) | 17 | 2 | 69.3 | 54.5 | 75.2 | 4 | 56.9 | 36.4 | 63.9 | 8.3 |
| 9 (21) | 19 | 2 | 59.9 | 45.5 | 65.0 | 4 | 47.0 | 31.8 | 52.2 | 7.6 |
| 10 (23) | 21 | 2 | 68.4 | 54.5 | 73.0 | 4 | 52.9 | 36.4 | 57.9 | 9.5 |
| 11 (26) | 23 | 3 | 66.2 | 50.0 | 72.1 | 5 | 61.1 | 45.5 | 67.5 | 9.9 |
| 12 (28) | 26 | 2 | 73.9 | 59.1 | 80.1 | 4 | 65.9 | 50.0 | 73.0 | 9.0 |
| 13 (30) | 28 | 2 | 71.9 | 59.1 | 77.3 | 4 | 60.8 | 45.5 | 67.1 | 10.7 |
| 14 (32) | 30 | 2 | 69.9 | 54.5 | 75.4 | 4 | 60.3 | 45.5 | 66.3 | 13.2 |
| 15 (34) | 32 | 2 | 61.6 | 45.5 | 66.7 | 4 | 54.6 | 36.4 | 60.2 | 11.2 |
| 16 (37) | 34 | 3 | 65.3 | 50.0 | 70.8 | 5 | 58.3 | 40.9 | 64.2 | 13.4 |
| 17 (39) | 37 | 2 | 70.6 | 54.5 | 75.3 | 4 | 61.8 | 45.5 | 66.9 | 12.7 |
| 18 (41) | 39 | 2 | 68.7 | 50.0 | 73.9 | 4 | 63.7 | 45.5 | 69.0 | 12.4 |
| 19 (43) | 41 | 2 | 73.4 | 59.1 | 78.6 | 4 | 63.5 | 45.5 | 69.8 | 12.2 |
| 20 (45) | 43 | 2 | 71.5 | 54.5 | 76.3 | 4 | 63.2 | 45.5 | 68.8 | 12.8 |
| 21 (48) | 45 | 3 | 65.9 | 50.0 | 71.3 | 5 | 58.5 | 40.9 | 64.3 | 9.6 |
| 22 (50) | 48 | 2 | 71.5 | 54.5 | 76.6 | 4 | 64.6 | 45.5 | 70.3 | 10.6 |
| 23 (52) | 50 | 2 | 70.3 | 54.5 | 77.0 | 4 | 57.6 | 40.9 | 64.4 | 9.8 |
| 24 (54) | 52 | 2 | 77.7 | 63.6 | 83.3 | 4 | 63.2 | 45.5 | 70.8 | 9.8 |
| 25 (56) | 54 | 2 | 75.7 | 59.1 | 82.4 | 4 | 67.0 | 45.5 | 75.0 | 9.9 |
| 26 (59) | 56 | 3 | 70.2 | 50.0 | 77.3 | 5 | 64.7 | 45.5 | 72.5 | 9.2 |
| 27 (61) | 59 | 2 | 77.4 | 59.1 | 82.8 | 4 | 66.7 | 45.5 | 73.7 | 10.3 |
| 28 (63) | 61 | 2 | 75.1 | 59.1 | 81.3 | 4 | 67.5 | 45.5 | 75.0 | 10.4 |
| 29 (65) | 63 | 2 | 75.5 | 54.5 | 81.3 | 4 | 65.7 | 40.9 | 72.9 | 8.1 |
| 30 (67) | 65 | 2 | 78.2 | 63.6 | 83.2 | 4 | 68.6 | 45.5 | 75.0 | 8.7 |
| 31 (70) | 67 | 3 | 69.1 | 45.5 | 73.3 | 5 | 64.0 | 36.4 | 69.5 | 8.2 |
| 32 (72) | 70 | 2 | 76.3 | 59.1 | 81.2 | 4 | 64.0 | 36.4 | 69.0 | 8.9 |
| 33 (74) | 72 | 2 | 75.3 | 59.1 | 80.6 | 4 | 64.8 | 45.5 | 71.5 | 7.6 |
| 34 (76) | 74 | 2 | 69.5 | 50.0 | 74.9 | 4 | 58.2 | 36.4 | 64.9 | 8.8 |
| 35 (79) | 76 | 3 | 65.2 | 45.5 | 70.5 | 5 | 54.8 | 31.8 | 61.1 | 9.6 |
| 36 (81) | 79 | 2 | 63.8 | 40.9 | 68.6 | 4 | 55.0 | 31.8 | 60.0 | 10.1 |
| 37 (83) | 81 | 2 | 63.9 | 45.5 | 68.9 | 4 | 48.1 | 22.7 | 53.4 | 10.5 |
| 38 (85) | 83 | 2 | 68.1 | 50.0 | 74.9 | 4 | 54.1 | 36.4 | 61.4 | 9.5 |
| 39 (87) | 85 | 2 | 67.4 | 50.0 | 76.9 | 4 | 56.1 | 36.4 | 66.0 | 10.7 |

#### Held-out text: per-expert importance

| experts | of all experts, % | of calibration pairs, % | of these pairs, % | mean weight | mean output norm | mean weight x output | mean contribution | mean share of the token's, % | of all contribution, % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hot | 13.1 | 23.7 | 23.4 | 0.2246 | 22.3397 | 5.9237 | 10.7070 | 4.8 | 27.0 |
| mid | 36.9 | 42.3 | 42.4 | 0.2239 | 19.2606 | 4.9268 | 8.8483 | 4.4 | 40.5 |
| cold | 50.0 | 34.0 | 34.1 | 0.2333 | 18.8368 | 4.9480 | 8.8131 | 4.6 | 32.5 |

Mean over layers: Spearman of calibration frequency with the mean weight -0.25, with the mean output norm -0.04, with the mean weight x output -0.12, with the mean contribution -0.12; a cold expert's mean weight x output over a hot one's 1.00, its mean contribution 1.00.

Experts of a layer by calibration frequency in tenths, the most used first (pooled over the layers):

| tenth | of all experts, % | of calibration pairs, % | of these pairs, % | mean weight | mean output norm | mean weight x output | mean contribution | mean share of the token's, % | of all contribution, % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.2 | 19.3 | 19.1 | 0.2253 | 23.0290 | 6.1650 | 11.1929 | 4.9 | 23.1 |
| 2 | 10.2 | 14.2 | 14.1 | 0.2223 | 19.8047 | 5.0600 | 9.0057 | 4.4 | 13.7 |
| 3 | 10.0 | 12.1 | 12.1 | 0.2233 | 19.4418 | 4.9702 | 8.9326 | 4.4 | 11.7 |
| 4 | 10.0 | 10.8 | 10.9 | 0.2245 | 18.7112 | 4.7576 | 8.5501 | 4.3 | 10.0 |
| 5 | 10.0 | 9.8 | 9.8 | 0.2251 | 18.8697 | 4.8341 | 8.7108 | 4.3 | 9.2 |
| 6 | 10.0 | 8.9 | 8.9 | 0.2280 | 18.8952 | 4.9298 | 8.8700 | 4.4 | 8.5 |
| 7 | 10.0 | 8.0 | 8.0 | 0.2299 | 18.7543 | 4.8971 | 8.7361 | 4.4 | 7.6 |
| 8 | 10.0 | 7.0 | 7.0 | 0.2322 | 18.4714 | 4.7975 | 8.5198 | 4.5 | 6.5 |
| 9 | 10.0 | 5.9 | 6.0 | 0.2372 | 18.9174 | 4.9803 | 8.8176 | 4.7 | 5.7 |
| 10 | 10.0 | 4.1 | 4.0 | 0.2483 | 19.3901 | 5.3161 | 9.3722 | 5.3 | 4.1 |

| MoE layer (block) | Spearman weight | output | weight x output | contribution | cold / hot weight x output | cold / hot contribution |
|---|---:|---:|---:|---:|---:|---:|
| 0 (1) | -0.34 | -0.21 | -0.20 | -0.20 | 1.41 | 1.41 |
| 1 (3) | -0.32 | -0.08 | -0.13 | -0.13 | 0.98 | 0.98 |
| 2 (5) | -0.40 | -0.31 | -0.34 | -0.34 | 1.12 | 1.12 |
| 3 (8) | -0.38 | -0.15 | -0.24 | -0.24 | 1.13 | 1.12 |
| 4 (10) | -0.38 | -0.01 | -0.17 | -0.17 | 1.08 | 1.08 |
| 5 (12) | -0.36 | -0.05 | -0.19 | -0.18 | 1.12 | 1.12 |
| 6 (14) | -0.36 | -0.18 | -0.25 | -0.25 | 1.07 | 1.08 |
| 7 (17) | -0.30 | -0.08 | -0.16 | -0.16 | 0.99 | 0.99 |
| 8 (19) | -0.40 | -0.14 | -0.26 | -0.26 | 1.10 | 1.10 |
| 9 (21) | -0.28 | -0.06 | -0.12 | -0.12 | 1.05 | 1.04 |
| 10 (23) | -0.28 | -0.13 | -0.17 | -0.17 | 1.11 | 1.11 |
| 11 (26) | -0.36 | -0.19 | -0.27 | -0.27 | 1.23 | 1.22 |
| 12 (28) | -0.37 | -0.19 | -0.27 | -0.28 | 1.10 | 1.11 |
| 13 (30) | -0.39 | -0.15 | -0.24 | -0.24 | 1.08 | 1.09 |
| 14 (32) | -0.41 | -0.24 | -0.34 | -0.34 | 1.23 | 1.22 |
| 15 (34) | -0.25 | 0.01 | -0.05 | -0.06 | 1.04 | 1.04 |
| 16 (37) | -0.39 | -0.22 | -0.30 | -0.30 | 1.18 | 1.18 |
| 17 (39) | -0.30 | -0.11 | -0.16 | -0.16 | 1.06 | 1.06 |
| 18 (41) | -0.29 | -0.02 | -0.09 | -0.09 | 0.99 | 0.99 |
| 19 (43) | -0.26 | -0.15 | -0.16 | -0.17 | 1.03 | 1.03 |
| 20 (45) | -0.31 | -0.07 | -0.14 | -0.14 | 1.02 | 1.02 |
| 21 (48) | -0.27 | -0.19 | -0.21 | -0.21 | 1.11 | 1.12 |
| 22 (50) | -0.24 | -0.07 | -0.11 | -0.11 | 1.05 | 1.04 |
| 23 (52) | -0.30 | 0.14 | -0.04 | -0.04 | 0.89 | 0.89 |
| 24 (54) | -0.27 | 0.12 | -0.02 | -0.02 | 0.83 | 0.83 |
| 25 (56) | -0.27 | 0.05 | -0.08 | -0.08 | 0.93 | 0.93 |
| 26 (59) | -0.32 | 0.11 | -0.13 | -0.13 | 0.87 | 0.87 |
| 27 (61) | -0.33 | 0.17 | -0.05 | -0.05 | 1.02 | 1.02 |
| 28 (63) | -0.29 | 0.04 | -0.11 | -0.11 | 0.96 | 0.96 |
| 29 (65) | -0.20 | 0.09 | -0.04 | -0.04 | 0.86 | 0.85 |
| 30 (67) | -0.16 | 0.08 | -0.02 | -0.02 | 0.86 | 0.86 |
| 31 (70) | -0.22 | 0.12 | -0.02 | -0.02 | 0.90 | 0.89 |
| 32 (72) | -0.20 | -0.02 | -0.07 | -0.07 | 0.87 | 0.87 |
| 33 (74) | -0.10 | 0.03 | 0.00 | 0.00 | 0.89 | 0.89 |
| 34 (76) | -0.01 | 0.02 | 0.02 | 0.02 | 0.91 | 0.91 |
| 35 (79) | -0.00 | 0.04 | 0.03 | 0.03 | 0.94 | 0.95 |
| 36 (81) | 0.10 | 0.06 | 0.10 | 0.10 | 0.89 | 0.88 |
| 37 (83) | 0.02 | -0.04 | -0.02 | -0.02 | 0.85 | 0.85 |
| 38 (85) | 0.16 | 0.12 | 0.13 | 0.13 | 0.71 | 0.72 |
| 39 (87) | 0.10 | 0.11 | 0.14 | 0.15 | 0.72 | 0.70 |

### Decode steps: relu2 sparsity inside the experts (1,108,800 pairs, 40 layers)

| share of the expert's neurons, % | mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| zero (exactly 0 after relu2) | 91.6 | 83.4 | 94.0 | 98.5 |
| columns of down kept for 1.0% error | 4.5 | 1.0 | 3.5 | 9.4 |
| columns of down kept for 0.1% error | 5.7 | 1.3 | 4.6 | 11.8 |

| bytes of an expert read at 4 bits (mean over pairs) | kept for 1.0% | kept for 0.1% |
|---|---:|---:|
| down, columns kept (B) | 61,579 | 78,545 |
| down, saved (B) | 1,314,677 | 1,297,711 |
| saved, share of the expert's bytes (%) | 47.8 | 47.1 |
| up and down, if the neurons were named first, saved (B) | 2,629,354 | 2,595,422 |

#### Decode steps: naming the active neurons before up is computed

| neurons named | of the expert's | recall mean, % | recall p10, % | error mean, % | within 1%, % | oracle error mean, % | oracle within 1%, % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 168 | 6.25 | 20.4 | 9.5 | 85.60 | 1.6 | 1.40 | 75.9 |
| 336 | 12.5 | 31.9 | 18.0 | 77.67 | 1.8 | 0.13 | 96.4 |
| 672 | 25 | 49.4 | 34.0 | 64.89 | 2.2 | 0.00 | 99.9 |
| 1344 | 50 | 73.9 | 60.5 | 43.01 | 3.8 | 0.00 | 100.0 |
| 2016 | 75 | 90.4 | 83.0 | 22.84 | 10.7 | 0.00 | 100.0 |

#### Decode steps: look-ahead routing

| d (blocks) | tokens | recall x1, mean % | recall x1.5, mean % | recall x2, mean % | recall x3, mean % | recall x1, p10 % | weighted x1, mean % | weighted x1, p10 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50400 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1 | 50400 | 79.3 | 90.3 | 94.1 | 96.8 | 63.6 | 84.5 | 69.5 |
| 2 | 49140 | 73.9 | 85.9 | 91.1 | 95.4 | 59.1 | 80.8 | 65.0 |
| 3 | 49140 | 67.8 | 79.5 | 85.5 | 91.3 | 50.0 | 75.5 | 55.5 |
| 4 | 47880 | 64.8 | 76.4 | 82.9 | 89.6 | 45.5 | 73.1 | 53.5 |
| 5 | 47880 | 61.0 | 72.2 | 78.8 | 86.0 | 40.9 | 69.4 | 47.0 |
| 6 | 46620 | 58.5 | 69.5 | 76.3 | 84.2 | 36.4 | 67.2 | 45.0 |
| at previous MoE layer (d = 2, 3) | 49140 | 72.7 |  |  |  | 54.5 | 79.8 | 63.5 |
| two blocks earlier (d = 4, 5) | 47880 | 64.3 |  |  |  | 45.5 | 72.6 | 52.5 |
| random |  | 4.3 | 6.4 | 8.6 | 12.9 |  |  |  |
| popular |  | 6.5 | 9.3 | 12.1 | 17.5 |  |  |  |

#### Decode steps: per-expert importance

| experts | of all experts, % | of calibration pairs, % | of these pairs, % | mean weight | mean output norm | mean weight x output | mean contribution | mean share of the token's, % | of all contribution, % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hot | 13.1 | 23.7 | 17.7 | 0.2168 | 25.8323 | 6.8782 | 12.3466 | 4.4 | 18.3 |
| mid | 36.9 | 42.3 | 40.4 | 0.2211 | 24.1439 | 6.3711 | 11.3443 | 4.3 | 38.2 |
| cold | 50.0 | 34.0 | 41.9 | 0.2377 | 25.7140 | 7.1105 | 12.4671 | 4.8 | 43.5 |

Mean over layers: Spearman of calibration frequency with the mean weight -0.24, with the mean output norm -0.09, with the mean weight x output -0.13, with the mean contribution -0.13; a cold expert's mean weight x output over a hot one's 1.17, its mean contribution 1.17.

### Checks

- wall clock: prefill 443 s, decode 251 s
- the layer-by-layer pass against the model's own forward on 2 sequences: KL 0.0e+00, top-1 agreement 1.0000
- the bisection against scanning every count, on 1280 counts of the first pairs of one expert in each of 40 layers: 1270 equal, largest difference 6 columns
- prefill: the look-ahead router at d = 0 (the same router on the same stream) names all 22 experts the model chose for 100.0% of tokens
- decode: the look-ahead router at d = 0 (the same router on the same stream) names all 22 experts the model chose for 100.0% of tokens
- decode: the experts the router chooses again from the kept inputs are the ones it chose while decoding for 100.0% of the steps of a layer on average (99.9% in the worst layer)

