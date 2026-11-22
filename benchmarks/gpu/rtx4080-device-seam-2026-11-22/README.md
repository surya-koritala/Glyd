# The device seam before and after, RTX 4080, gemma-4-12B at wren (2026-11-22)

R1a (#134) and R15 (#236) on this branch: every call to the GPU driver is in `src/dev.rs` (the CUDA and HIP device layer), a buffer belongs to the context it was made in
(read, written, zeroed and freed in it whichever context is current), and the device functions that could cause undefined behaviour are `unsafe` with their contracts.
#134 asks for `glyd bench` before and after on the reference GPU: this is that.

**Result: no change. Every figure of the gate suite is within 0.2% of main's in both sessions, and `bench --compare` says "within the noise band" for both.**

Machine: a rented RTX 4080 (16 GB, driver 595.99.02, Threadripper PRO 3955WX), nothing else on the GPU; the machine of `../rtx4080-shared-starts-2026-11-22`. Main is a604fbb5;
this branch is main with the three commits of M2 batch 13. Both built on the machine by the GPU job's recipe (kernels for sm_89), main in a target directory of its own.
`glyd-engine bench google/gemma-4-12B-it:wren --suite gate --runs 3 --no-draft`, main then this branch, twice (`main-gate-1.json`, `head-gate-1.json`, then `-2`).

| load | figure | main, 1 | branch, 1 | main, 2 | branch, 2 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| short prompt, 1 user | decode tokens/s | 86.57 | 86.60 | 86.60 | 86.61 |
| | first token | 0.024 s | 0.024 | 0.024 | 0.024 |
| short prompt, 8 users | decode tokens/s a user | 61.95 | 62.11 | 62.10 | 62.10 |
| | first token | 0.107 s | 0.103 | 0.123 | 0.123 |
| 1k prompt, 1 user | decode tokens/s | 82.10 | 82.12 | 82.11 | 82.11 |
| | prefill tokens/s | 3,555 | 3,553 | 3,546 | 3,545 |
| 4k prompt, 1 user | decode tokens/s | 78.99 | 78.99 | 78.99 | 78.99 |
| | prefill tokens/s | 3,444 | 3,437 | 3,436 | 3,435 |

- The eight users' first token moves between sessions (0.103 to 0.123 s) and moves the same way for both builds: it is when the eight requests reach the round, not the build.
- Checked on the same machine before the bench: `bufcheck` (a buffer of either of two contexts on the GPU written, read, zeroed and freed while the other is current, its
  256 MB back), `keepcheck`, `roundcheck` and `usercheck` on every family's random model, `twomodels`, Nemotron-H's `nhtoy` and `nhoffload`, and `tiercheck` on random
  DeepSeek-V4, GLM-5.3, Muse Spark and Kimi-K3 models (0 logits differ against every expert resident).
