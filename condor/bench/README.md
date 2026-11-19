# bench/

- `run_bench.sh`: GSM8K 150 + MMLU-Pro 150 against this project's server (the numbers in `results/`).
- `compare.py`: the same questions plus a speed test against any OpenAI-compatible server, for comparing on your own machine.
- `eval_slice.py`, `merge_slices.py`: what both use to run Inspect AI's tasks in slices of 25 and add them up.

## compare.py

```sh
python3 bench/compare.py --url http://127.0.0.1:8080 --model qwen3.8-flash-next --note "precise"
```

What it does, in order:

1. One warm-up request (not counted).
2. Three streamed requests, temperature 0, thinking off (`chat_template_kwargs.enable_thinking: false`): a short story prompt
   (up to 512 tokens out), then a ~4K-token and a ~30K-token prompt (it fits a server started with a 32,768-token
   context) (numbered inventory notes, made the same way every time, up to
   256 tokens out). Each long prompt starts with its own label, so a server's prompt cache can't reuse the 4K one for the 30K one.
   - "First token" is the time from sending the request to the first streamed token (thinking or answer).
   - "Prompt read" is the prompt's token count (from the server's `usage`) divided by that time.
   - "Writing" is the tokens after the first divided by the time from the first to the last.
   - Token counts come from the server's `usage` (`stream_options.include_usage`). If a server doesn't send it, the
     script counts stream chunks instead, which can be off where a server sends several tokens in one chunk.
3. GSM8K and MMLU-Pro with Inspect AI, the same questions and settings as `results/v0.1.0`: every second question, the first 150
   (or `--questions N`, a multiple of 25), greedy, thinking off, at most 3,072 tokens an answer, 8 requests at a time. The first
   run makes a Python environment for Inspect AI in `~/.cache/compare-venv-eval` (set `COMPARE_VENV` to put it elsewhere).
4. Prints the machine (GPU, driver, PCIe link, CPU, RAM, OS) and everything above as one block, also saved to
   `compare-out/compare.md`.

Things to keep the same between servers: the computer, nothing else running on the GPU, the context size the server was started
with, and the model. Say in `--note` which file or quant you ran and the flags, so the block makes sense later. The prompt set is
named in the block (`compare-v1`); numbers from different prompt sets don't compare.

If a server needs an API key, set `OPENAI_API_KEY`.
