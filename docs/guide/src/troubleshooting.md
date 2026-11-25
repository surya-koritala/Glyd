# Troubleshooting

Start with `glyd doctor`. It prints one line for each thing a run needs and ends with
"Everything glyd run needs is here." (exit 0) or "N thing(s) to fix." (exit 1). Fix the lines it names, in order.

For a bug report, run `glyd doctor --json` (the bundle on the screen) or `glyd doctor --bundle` (the same in a file, `glyd-bundle-TIME.json`, or
`--bundle FILE`) and attach it. It holds the versions, every check with its fix, each GPU's memory and the terms a plan is made of (what the engine keeps
free, its work, the computer's memory kept for the system), each backend detected or rejected and why, the plan `run` would load for Gemma 4 12B here, the
ports, the models in the caches, the `GLYD_*` and `HF_*` environment, the service's configuration and the end of its recent logs. Tokens, keys, credentials in
addresses and your home folder are taken out of it (`~`); `--no-logs` leaves the logs out. Read it before you post it.

## Exit codes

| Code | Meaning |
| :--- | :--- |
| 0 | Done. For `supports` and `preflight`: the engine runs it. |
| 1 | Something went wrong, or a wrong flag, or the Hub did not answer. The message says what and what to do. |
| 3 | The engine cannot run this model on this computer: the model's kind, the GPU, the driver, or not enough memory. The reason is on stderr. A program that started the engine can then run the model another way. |

## By doctor line

| `doctor` says | Cause | Fix |
| :--- | :--- | :--- |
| `GPU  none usable: no CUDA driver ...` | No NVIDIA driver loaded, or no GPU visible to the process. | Check `nvidia-smi`. Install the driver (580 or newer). In a container, give it the GPU (`--gpus all`). An empty `CUDA_VISIBLE_DEVICES` hides every GPU. |
| `older than the engine runs on (Ampere or newer)` | Compute capability below 8.0: Turing, Pascal and before. | Nothing on this card. RTX 30 series and later, A10, A100, L4, L40S and H100 run. |
| `driver  too old: the driver here runs CUDA 12.x ...` | The engine's kernels need CUDA 13 (driver 580 or newer). | Update the driver. On Ubuntu: `sudo apt install nvidia-driver-580` (or `sudo ubuntu-drivers install`) and reboot; elsewhere the installer at nvidia.com/drivers. A CUDA 12 build is ROADMAP P6. |
| `GPU library  not found` | `GLYD_GPU_LIB` names a GPU library that does not load (the engine loads one only where the variable names it; an AMD build looks for `libglyd_gpu_hip` beside the program). | Unset `GLYD_GPU_LIB`: the engine runs its own kernels. On AMD: `bash gpu/build_lib_hip.sh target/release`. |
| `cuBLAS  not found` | Not an error. Long prompts use the engine's own products, a little slower. | Optional: install the CUDA 13 cuBLAS, or set `GLYD_CUBLAS` to it. |
| `cache`, `models`: a low "free" figure | The packed copy and the downloads need disk: the download is the model's size on the Hub, and the packed copy is extra. | Free space, or move them: `GLYD_CACHE=/big/drive` for packed models, `HF_HOME=/big/drive` for downloads. |
| `Hub token  none` | Fine for public models. | For a gated model, accept its licence on the model's page, then `glyd login` (a token from huggingface.co/settings/tokens), or set `HF_TOKEN`. |
| `backend ...` | Each backend (CUDA, ROCm, Metal, CPU) is said detected or not, with its version, and accepted or rejected with the refusal `run` would give. | Only a rejected backend that is the one you need is a problem: its line has the fix. |
| `GPU budget`, `RAM` | The terms a plan is made of: free memory, what the engine keeps and its work, the computer's memory kept for the system. | A small "for the weights and the cache" means little room: close programs that use the GPU. |
| `sample plan  ... does not load here` | Gemma 4 12B, a small model, has no level that loads on this GPU and memory. | Not an error: a bigger model will not either. `glyd fit MODEL` shows each level. |
| `port  ... is in use by ...` | A program holds the service's port (11434) or a server's first port (8000). | Stop it, or `glyd service --port N`; a server takes the next free port up to 8020. |
| `Hub  ...: <error>` | The Hub is not reachable: no network, a proxy, or IPv6 that fails. | Check the network. `GLYD_IPV4=1` forces IPv4. `HF_ENDPOINT` names a mirror. A model already downloaded runs with `HF_HUB_OFFLINE=1`. |

## Refusals from `run` and `serve`

These are exit 3 unless noted. The message says what to do; the usual ones:

| Message starts | What it means | What to do |
| :--- | :--- | :--- |
| `X is a ...ForCausalLM model, which this engine has no description of yet` | The model's kind is not one of the [Models](models.md) families. | Pick a listed family, or run it with another engine. |
| `GLM-5.3-Flash (GLM-5.3-Flash) runs at the penguin level in this engine, not swift` | The family does not run that level. | Use the level the message offers, or see [Models](models.md). |
| `it needs ... GiB of GPU memory` | The model does not fit at the level and context asked. | A lower level, a smaller `--context`, another GPU, or for a mixture of experts more of the computer's memory (`--host-memory`). `glyd fit MODEL` shows each level's size first. |
| `no GPU was found: the engine's CPU backend is not built yet` | No usable GPU. | See [Install](install.md). |
| `AMD GPUs run in development builds only (GLYD_HIP=1)` | AMD is a development build. | `GLYD_HIP=1 cargo build --release -p glyd-engine`. |
| `Apple GPUs are not supported yet` | macOS. | The engine needs an NVIDIA GPU on Linux. |
| `glyd service` says its port is in use, and names the program | Something else has the service's port. The service never moves to another port. | Stop that program, or `glyd service --port N`; `glyd ps`, `stop` and `run` read `--port N` or `GLYD_SERVICE_PORT=N`. |
| `port 8000 is in use by another program` (exit 1) | You gave `--port` and something else has it. | Another port, `--port 8001`. Without `--port` the server takes the next free port up to 8020. |
| `Gemma 4 ... not in this engine yet` | Gemma 4's 26B-A4B and E models are refused. | Gemma 4 12B or 31B. |
| `the Hugging Face Hub would not show X` (exit 1) | The name is wrong, or the model is gated or private. | Check the name (`ORG/NAME`). For a gated model, accept the licence and run `glyd login`. |

## Still stuck

Run `glyd doctor` and `glyd preflight MODEL` and put what they print in an
[issue](https://github.com/surya-koritala/Glyd/issues) with the model, the command and the GPU. For Condor, choose "Condor:
something went wrong".
