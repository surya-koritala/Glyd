# h86cal: KLD and warm decode speed (2026-10-15, RTX 4080 SUPER, 60 GiB RAM)

KLD: wikitext-2 test 32 x 512 vs bf16, as in table-step2.md. Speed: qwenserve at ctx 32768 as users run it, compare.py's speed part
twice per server start, the second (warm) pass counted; ms a token from the server's log (greedy, default drafts) for the 33-token chat
prompt, the 4,016-token and the 29,663-token prompt. Single runs vary about ±0.5 ms; today's precise and h86cal i6 were run twice
(alternating) and averaged. The first measurements (13.2 vs 10.5 ms at 4K) were with a partly cold page cache: the two packs (38 + 42 GiB)
evict each other in 60 GiB, so every server start reads experts from NVMe at first.

| config | mean KLD | median | PPL ratio | top-1 | ms/token short / 4K / 30K | vs today | places a layer | VRAM in use | experts on disk |
|---|--:|--:|--:|--:|---|---|--:|--:|---|
| today's precise (h4 + cb3 copies, always-on i4, attn/head i5) | 0.406 | 0.060 | 1.265 | 82.0% | 9.25 / 9.0 / 8.35 (2 runs) | - | 116 | 15.11 GiB | 38.1 GiB + 4.3 GB copies |
| h86cal, always-on i4 (attn/head i5) | 0.296 | 0.035 | 1.161 | 85.1% | 9.5 / 9.2 / 8.5 | +2-3% | 116 | 15.11 GiB | 41.8 GiB, no copies |
| h86cal, attention + head i6, rest i4 | 0.283 | 0.032 | 1.143 | 85.7% | 9.1 / 8.8 / 9.3 | -2..+11% (noisy) | 116 | 15.26 GiB | same |
| h86cal, always-on i5 | 0.251 | 0.022 | 1.114 | 86.5% | 9.4 / 9.6 / 8.8 | +2-7% | 115 | 15.39 GiB | same |
| h86cal, always-on i6 | 0.219 | 0.019 | 1.076 | 87.7% | 10.15 / 9.85 / 9.2 (2 runs) | +9-10% | 111 | 15.38 GiB | same |

Where the time goes: not the GPU hit rate (69.7-75.0% for all, today 71.4-72.8%) and not the places lost to i6 (116 -> 111). The
always-on matrices are read every pass: i6 reads ~1 GB more a pass than i4 (~+7%); the bigger cb3 share of h86 adds ~2-3% (host misses
now land on cb3 experts more often). Tokens a pass (drafts) range 1.9-2.7 with the text written, with no clear difference by config.
