# Changelog

## Unreleased

- Ibis moved into the Glyd repository (`ibis/`), and the engine is open source: Glyd's engine and GPU library are Apache-2.0 and
  their source is in the same repository. Nothing about the model, the modes or the commands changes.
- `tools/install.sh` installs the engine bundle of a Glyd release (built by Glyd's release workflow from the public source); until a
  Glyd release carries one, it installs Ibis's v0.2.4 engine as before. `tools/install.sh --source` builds the engine and the GPU
  library on your machine (Rust and the CUDA toolkit), for the GPUs it finds there.
- `./setup.sh` reinstalls the engine when Glyd's latest release is newer than the installed one (an engine built with `--source` stays).

## 0.2.4

- Precise mode is closer to the original at the same size: most experts are at 3 bits a weight now (by how much each is used), the
  rest at 2. KL divergence vs the original bf16 (wikitext-2, 32 x 512): 0.22 -> 0.15, perplexity 1.076x -> 1.037x the original's. On
  the mixed text set (code, docs, five languages, an agent transcript), over the original's 64 most likely tokens: 0.180 -> 0.136,
  top-1 87% -> 88%. Long documents: 0.073 -> 0.042. It writes about 10% slower than v0.2.3's precise (same build, warm chat on the 4080 SUPER: 100 vs 112 tok/s; after a 4K prompt 75 vs 91). Still 82 GB.
- Best mode too: 4 bits for the experts that carry about 80% of the work, 3 for the rest. wikitext-2 KL 0.165 -> 0.12 (perplexity
  1.052x -> 1.041x); mixed text set 0.146 -> 0.097, top-1 87.6% -> 89.9%; long documents 0.039 -> 0.028. About 5% slower than v0.2.3's
  best (warm chat 98 vs 104 tok/s). 90 GB in all (was 89), 64 GB of RAM.
- The new experts are new downloads, `precise-v2/` (45 GB) and `best-v2/` (53 GB): `./setup.sh` fetches them, or `tools/download.sh`
  (`--best`). The old `precise/` and `best/` folders can be deleted after. Engines before v0.2.4 keep using the old folders. On
  Windows, run `tools\install.ps1` after `git pull` first (setup.ps1 doesn't update the engine yet).
- Long prompts are read faster: a ~4K-token prompt's first token 1.70 -> 1.30 s in fast mode on the 4080 SUPER, 2.16 -> 1.84 s in
  precise. The experts are copied to the GPU through buffers of the copier threads' own instead of the driver's slow path.
- A named `tool_choice` gets one call to that tool (fast mode sometimes added a second one). Fast passes 18 of the 19 tool-call tests;
  the 12-round loop still overruns now and then with fast's experts.
- docs/CLIENTS.md: Hermes Agent needs a 64K context (`GLYD_CTX=65536`).
- Tested on Windows Server 2022 on an AWS L4 with only NVIDIA's driver installed (no CUDA toolkit): install, download, every test in
  `tests/`, the bench and the MCP server.

## 0.2.3

- Images: PNG and JPEG in OpenAI `image_url` parts (base64, or a URL on a public host: the server won't fetch from this computer or
  its network on a client's behalf) and Anthropic image blocks, up to 1,024 tokens an image by default (`--image-pixels`). Our own decoder, no image library; the image encoder runs on the GPU and matches the reference model
  within bf16 noise. On a 4080 SUPER an image adds 0.3-2 s to the prompt; text speed is unchanged. `tests/test_vision.py`. Not yet:
  video, a camera's orientation tag, uploads on the chat page.
- Up to 4 requests at once (`GLYD_USERS`, 1 to 8), each streaming from the start; past that they wait in order. One user's speed is
  unchanged; total speed stays near one user's, since every pass is still one user's. A new long prompt pauses the other streams
  while it is read. `tests/test_parallel.py`.
- Two cards: `GLYD_GPUS=0,1` splits the layers over them. On 2x4090: the same output as one card, byte for byte; 189 vs 186 tok/s.
- The MCP server reports its name as `ibis`.

## 0.2.2

- Fast mode is closer to the original at the same size: its experts are packed a new way, still 2 bits a weight. KL divergence vs the
  original bf16 (wikitext-2, 32 x 512): 0.53 -> 0.39, top-1 79% -> 83%. On a mixed text set (code, docs, five languages, an agent
  transcript), over the original's 64 most likely tokens: 0.397 -> 0.295, top-1 78% -> 81%. GSM8K 149/150, MMLU-Pro 121/150 (the v0.2.1
  experts on the same engine, the same night: 147 and 124). Writing after a short / 4K / 30K prompt: 124 / 116 / 125 tok/s, vs
  120 / 117 / 141 with the v0.2.1 experts on the same night; prompts are read at the same speed.
- The fast experts are a new download, `fast/` (29 GB): `./setup.sh` fetches them, or `tools/download.sh --fast`; the old `pack/` folder
  can be deleted after. Engines before v0.2.2 keep using `pack/`. 66 GB in all for fast mode (was 68).
- Known: with the new fast experts two of the 19 tool-call tests in `tests/test_agents.py` fail (a forced tool choice gets a second call;
  the 12-round tool loop overruns), 19 of 19 with precise and best. On the list for v0.2.3.
- The project is Ibis and the repository is surya-koritala/ibis (old links redirect). More Flash-class models will come here.
- Precise and best are unchanged.

## 0.2.1

- Best mode (`--best`, `-Best` on Windows; setup asks): about 30% of the experts at 4 bits, the rest at 3, calibrated like precise.
  KL divergence vs bf16 (wikitext-2, 32 x 512) 0.165 vs precise's 0.22, top-1 89% vs 87%; GSM8K 149/150, MMLU-Pro 128/150. About 10%
  slower than precise on short prompts, about the same at 30K. 52 GB of experts (89 GB in all), 64 GB of RAM. Precise stays the default.
- The MCP server's glyd_start: `mode` fast and best now work on Linux (it passed a flag serve.sh didn't take), and precise is the
  default it describes.
- The MCP server's glyd_stop on Ubuntu 22.04: it left the server running and could stop the program that started the MCP server
  (22.04's `kill` read the server's process group as its own). Fixed; tested on 22.04 and 26.04.
- The Monitor tab and the bench block name best mode (they said fast).
- setup.sh installs the engine again when the latest release is newer than the installed one (on Windows, run tools\install.ps1
  after `git pull` for now).
- README: the IPv4 workaround is gone (the engine falls back to IPv4 by itself).

## 0.2.0

- Precise mode is closer to the original model: its experts are packed again with calibration, and the always-on matrices are kept at
  six bits. KL divergence vs the original bf16 (wikitext-2 test, 32 x 512): mean 0.40 -> 0.22, median 0.06 -> 0.02, top-1 82% -> 88%.
  The download stays about 82 GB; writing is about 10% slower; VRAM in use about 15.4 of 16 GB.
- Tool calls work the way agent frameworks send them (streamed, several in one answer, tool_choice), and JSON mode (response_format).
- A second request waits for the first instead of failing; a client that leaves while waiting is dropped cleanly.
- bench/compare.py: the same speed test and questions against any OpenAI-compatible server, to compare on your own machine.
- tools/bench.sh prints a block to paste into a "Bench result" issue. docs/HOW_IT_WORKS.md and docs/CLIENTS.md.
- Windows 10/11 x64, native: setup.ps1 and tools\\*.ps1 (the .sh scripts' twins). Tested on one machine so far (an L4, 64 GB).
- 12 GB cards and 32 GB of RAM: setup picks fast mode below 64 GB of RAM; measured on a 4080 SUPER held back to those sizes.
- The repository is now surya-koritala/glyd-ibis (Glyd Ibis); the old links redirect.

## 0.1.1

- The engine runs on Ubuntu 22.04 and other Linux distributions from 2022 on (it needed a newer system library before).
- setup.sh says how to update an NVIDIA driver older than 580.
- Tested on an NVIDIA A10 (Ampere, 24 GB) on a clean Ubuntu 22.04 machine: 101 tokens/s in chat.

## 0.1.0

- Qwen3.8-Flash-Next on one 16 GB NVIDIA GPU with 64 GB of RAM: fast and precise modes, contexts up to 262,144 tokens.
- The chat app (Chat, Monitor, About), the OpenAI- and Anthropic-compatible APIs, an MCP server and a speed test.
