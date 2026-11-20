# Troubleshooting

Start with `glyd-engine doctor`. It prints one line for each thing a run needs and ends with
"Everything glyd run needs is here." (exit 0) or "N thing(s) to fix." (exit 1). Fix the lines it names, in order.

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
| `GPU library  not found` | `libglyd_gpu_cuda13.so` is not beside `glyd-engine` or on the loader's path. | Build it: `bash gpu/build_lib.sh target/release`, or set `GLYD_GPU_LIB=/path/to/libglyd_gpu_cuda13.so`. |
| `cuBLAS  not found` | Not an error. Long prompts use the engine's own products, a little slower. | Optional: install the CUDA 13 cuBLAS, or set `GLYD_CUBLAS` to it. |
| `cache`, `models`: a low "free" figure | The packed copy and the downloads need disk: the download is the model's size on the Hub, and the packed copy is extra. | Free space, or move them: `GLYD_CACHE=/big/drive` for packed models, `HF_HOME=/big/drive` for downloads. |
| `Hub token  none` | Fine for public models. | For a gated model, accept its licence on the model's page, then `glyd-engine login` (a token from huggingface.co/settings/tokens), or set `HF_TOKEN`. |
| `Hub  ...: <error>` | The Hub is not reachable: no network, a proxy, or IPv6 that fails. | Check the network. `GLYD_IPV4=1` forces IPv4. `HF_ENDPOINT` names a mirror. A model already downloaded runs with `HF_HUB_OFFLINE=1`. |

## Refusals from `run` and `serve`

These are exit 3 unless noted. The message says what to do; the usual ones:

| Message starts | What it means | What to do |
| :--- | :--- | :--- |
| `X is a ...ForCausalLM model, and this engine runs Qwen3 models so far` | The model's kind is not one of the [Models](models.md) families. | Pick a listed family, or run it with another engine. |
| `GLM-5.3-Flash (GLM-5.3-Flash) runs at the penguin level in this engine, not swift` | The family does not run that level. | Use the level the message offers, or see [Models](models.md). |
| `it needs ... GiB of GPU memory` | The model does not fit at the level and context asked. | A lower level, a smaller `--context`, another GPU, or for a mixture of experts more of the computer's memory (`--host-memory`). `glyd-engine fit MODEL` shows each level's size first. |
| `no GPU was found: the engine's CPU backend is not built yet` | No usable GPU. | See [Install](install.md). |
| `AMD GPUs run in development builds only (GLYD_HIP=1)` | AMD is a development build. | `GLYD_HIP=1 cargo build --release -p glyd-engine`. |
| `Apple GPUs are not supported yet` | macOS. | The engine needs an NVIDIA GPU on Linux. |
| `port 8000 is in use by another program` (exit 1) | You gave `--port` and something else has it. | Another port, `--port 8001`. Without `--port` the server takes the next free port up to 8020. |
| `Gemma 4 ... not in this engine yet` | Gemma 4's 26B-A4B and E models are refused. | Gemma 4 12B or 31B. |
| `the Hugging Face Hub would not show X` (exit 1) | The name is wrong, or the model is gated or private. | Check the name (`ORG/NAME`). For a gated model, accept the licence and run `glyd-engine login`. |

## Still stuck

Run `glyd-engine doctor` and `glyd-engine preflight MODEL` and put what they print in an
[issue](https://github.com/surya-koritala/Glyd/issues) with the model, the command and the GPU. For Condor, choose "Condor:
something went wrong".
