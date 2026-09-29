# How fast it responds: time to first token and tokens a second, 2026-09-29

What a user of `generate()` waits for, bf16 against Glyd, on one GPU at a time: `gpu/respond.py` in three modes, each
in a process of its own, the same prompts in each (a fixed English text's first N tokens; B copies for B sequences),
greedy decoding, every reply forced to its length (`min_new_tokens`: no early end of sequence):

- **bf16**: transformers' own model (`AutoModelForCausalLM`, bf16), its `generate()` as it runs by default (eager, a
  dynamic cache).
- **Glyd (default)**: `glyd.from_pretrained(MODEL)` as it loads by default: the layout the GPU's (`"auto"`: tiered on
  Ada, 12-bit on an A10, A100 and H100, where it fits), `generate()` compiled (a static cache, CUDA graphs) where a
  call's cache holds at most 2048 positions in all (1280 on a GeForce card), else eager.
- **Glyd exact**: `glyd.from_pretrained(MODEL, exact=True)`: every matrix decoded whole, then `F.linear`, the logits
  bf16's bit for bit; eager.

Each call is timed by a streamer (transformers hands it the prompt, then each step's tokens, syncing the GPU a step):
**time to first token** from the call to the first new token, **tokens a second** after it (new tokens less the first,
times the sequences, over the first token's put to the last), and the call's **total**. The measurements, in this
order in each process: time to first token for prompts of 128, 512, 2048 and 8192 tokens (16 new tokens each); the
chat mix (a 200-token prompt, a 300-token reply); tokens a second at 1, 8 and 32 sequences (a 128-token prompt, 256 new
tokens); the long-document mix (2000 and 200). Each configuration's first call is a warm-up, kept apart (in the default
mode the first one compiles); then the median of its repeats (5 for the time to first token, 3 for the rest). The
GPU's clocks and power are sampled while each mode runs (`nvidia-smi`, once a second).

| file | what |
| :--- | :--- |
| `gpu/respond.py` | one model in one mode: `python respond.py MODEL --mode bf16|glyd|exact --out RESULT.json` |
| `resp_job.sh` | the modes in turn, the library built for the GPU, the models downloaded: unattended, 20 minutes at most on a cloud GPU (an even share of the time left a mode; what does not fit its share is recorded so) |
| `resp_summary.py` | the tables (summary.txt) and every result in one JSON (respond.json) |
