"""What `glyd run`, `glyd serve` and `glyd doctor` check and choose before vLLM starts: the GPU and its driver, whether a model
fits the memory that is free, and the settings to start vLLM with. Numbers in, numbers out: the standard library only (nvidia-smi
and the Hub are read through functions the tests replace), no torch and no vLLM import.

    gpu, warnings = setup_checks()                        # nvidia-smi, the driver, vLLM
    m = load_model("Qwen/Qwen3.5-9B")                     # the Hub's config and file sizes (or a directory)
    s = settings(m, gpu, mode="run", nvcc=have_nvcc())
    s.util, s.context, s.eager, s.tool_parser, s.reasoning_parser, s.env

A failed check is a Refusal: what is wrong in plain words, and what to do about it.

The settings, in bytes (T: the memory CUDA sees, F: free now; the constants below are measured on an L4 held to a 16 GB card's
memory, benchmarks/gpu/l4-local-chat-2026-09-30):
- util = floor_0.01((F - CTX - HEADROOM) / T), at most 0.92 (vLLM's own default). vLLM's budget is util x T; its CUDA context
  (CTX) is outside it, and HEADROOM stays free for a desktop's own growth.
- weights = (Linears + embedding and LM head) x bits / 16 + the rest + the decode buffer (the largest matrix, bf16) + EXTRA: the Linears
  and the embedding and LM head the plugin packs (10.80 bits a weight tiered, 12.04 12-bit), the norms as they are, and the plugin's buffer
  and workspaces.
- context = the model's own length, or the most the KV cache holds in (util x T - weights - NON_KV), rounded down to a multiple
  of 1024 (at KV_FIT of it: block rounding and the estimate's own error); under MIN_CONTEXT the model does not fit.
- compiled (vLLM's own default: torch.compile and CUDA graphs) wherever it fits, eager (`--enforce-eager`) where only eager leaves room for
  the chat: compiled needs 2.15 GiB beyond the weights where eager needs 0.5, and on an RTX 4090 and an RTX 5090 eager gave Qwen3.5-9B
  under a third of compiled's tokens a second (on an L4 with Qwen3-8B eager was within 2-3% of compiled).
Under the Apache License 2.0 (LICENSE), as the rest of Glyd.
"""
import errno
import glob
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from .fit import RATIO, HubError, _hub, _local, attention_heads, checkpoint, kv_parts, mixers, text_config
from .format import MANIFEST
from .page import own_origins
from .vllm_entry import TESTED, tested

GiB = 2**30
CTX = 0.55 * GiB  # the CUDA context and driver's share, outside vLLM's budget (EngineCore held 14,958 MiB at a 14.06 GiB budget)
HEADROOM = 0.4 * GiB  # left free for a desktop (the owner's card ran with 0.23 GiB of it). Not more where a display is attached (Gpu.display): no desktop's growth has been measured, and this is more than that one took
NON_KV = 0.55 * GiB  # vLLM's non-torch memory and peak activation beyond the weights, eager (Qwen3 0.6B to 8B on an L4: 0.38 to 0.70 GiB, 0.49 for the 8B), at the batch sizes vLLM 0.30 uses below 70 GiB of GPU memory (2,048 tokens, 256 sequences); from 70 GiB it sizes them up (8,192 and 1,024, 16,384 from 160 GiB), and the activation peak with them: not measured there, where a chat's context is the model's own length with GiBs to spare
NON_KV_COMPILED = 2.3 * GiB  # the same compiled (Qwen3-8B: 2.15 GiB, a 1.66 GiB activation peak and CUDA graphs): only for a user's --no-enforce-eager
HYBRID = 0.65 * GiB  # a model with linear-attention or Mamba layers: its activation peak beyond NON_KV's and its layers' state (Qwen3.5-9B compiled on an RTX 4090: 2.34 GiB of peak activation and vLLM's 6.32 GiB of KV cache for 203,776 tokens, against 32 KiB a token; Granite 4.2 8B 0.83 and Gemma 4 12B 1.03 GiB of peak; run-families-2026-10-06)
EXTRA = 0.15 * GiB  # the plugin's other workspaces on top of its packs and its decode buffer (Qwen3-8B: 10.45 GiB measured as vLLM logs it with the embedding and LM head packed, the decode buffer being made for the call now (it was 10.63 with the buffer held), 10.49 from the bits and the buffer: the estimate keeps the buffer, which a step's peak still holds in part)
UTIL_MAX = 0.92
MIN_CONTEXT = 4096
CONTEXT_WANT = 8192  # tokens a chat should have: below it a smaller layout is considered
KV_FIT = 0.97
BATCH = 5120  # tokens a sliding-window layer keeps beyond its window: vLLM 0.30 sizes it by the tokens in flight, two steps of max_num_batched_tokens (2,048, or more for a model with a vision tower: vLLM held about 4,990 beyond Gemma 4 12B's 1,024-token window on an RTX 4090, run-families-2026-10-06)
BITS = {"mma": 10.80, "mma12": 12.04}
MIN_CAPABILITY = (8, 0)  # Ampere: the plugin's get_min_capability
MIN_DRIVER = {(12, 4): "550", (12, 6): "560", (12, 8): "570", (12, 9): "575", (13, 0): "580"}  # the driver that runs a CUDA version (Linux)
HUB = "https://huggingface.co"
INSTALLER = "curl -LsSf https://getglyd.com/install.sh | sh"


class Refusal(Exception):
    """A check that failed: what is wrong, in plain words, and what to do about it (a line each)."""

    def __init__(self, what, fix=""):
        super().__init__(what)
        self.what, self.fix = what, fix


def gb(n):
    """Bytes as the GB a person compares with a spec sheet (10^9), one decimal."""
    return f"{n / 1e9:.1f} GB"


# --- the GPU -----------------------------------------------------------------------------------------------------------------

@dataclass
class Gpu:
    index: int
    name: str
    total: int  # bytes CUDA sees: nvidia-smi's total less what the driver keeps (vLLM's fraction is of this)
    free: int
    cc: tuple  # compute capability, (8, 9)
    driver: str  # "595.91.07"
    cuda: tuple  # the newest CUDA the driver runs, (13, 2); () where not told
    display: bool = False  # a display is attached to it (a desktop)
    mig: bool = False  # Multi-Instance GPU is on: CUDA sees a slice, nvidia-smi the whole GPU


def _run(cmd):
    """A command's output (decoded where it is not UTF-8), or None where it is not there or fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


COMPRESSION = "The compression commands (glyd FILE -o OUT) work here."
DRIVER_PACKAGE = "sudo apt install nvidia-driver-{want}, then reboot"  # (Ubuntu's package for a driver of that number: ubuntu-drivers install takes the recommended one, which may be older)


def no_gpu_refusal(platform=None):
    """Why this computer has no GPU glyd can use, in words that fit it: no driver advice for a Mac, one message for a Linux machine
    that has no NVIDIA GPU and one that has a GPU and no driver (nvidia-smi is missing either way)."""
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        return Refusal("glyd run needs Linux and an NVIDIA GPU, and this computer is a Mac", COMPRESSION)
    if platform in ("win32", "cygwin"):
        return Refusal("glyd run needs Linux and an NVIDIA GPU, and this computer runs Windows", "Run glyd inside WSL2, which uses the NVIDIA driver of Windows. " + COMPRESSION)
    return Refusal("no NVIDIA GPU answered: nvidia-smi is missing or failed",
                   "glyd run needs an NVIDIA GPU (Ampere or newer) on Linux; the compression commands (glyd FILE -o OUT) work without one.\n"
                   "If this computer has one, install its driver (Ubuntu: " + DRIVER_PACKAGE.format(want=MIN_DRIVER[max(MIN_DRIVER)]) + "), then run: glyd doctor")


def probe_gpus(run=_run, platform=None):
    """The NVIDIA GPUs nvidia-smi lists, or a Refusal saying why there are none. `run` (a command's output, or None) is
    replaced by the tests."""
    query = lambda fields: run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    base = "index,name,memory.total,memory.free"
    out = None
    for fields in (base + ",memory.reserved,driver_version,compute_cap,display_active,mig.mode.current", base + ",memory.reserved,driver_version,compute_cap,display_active",
                   base + ",driver_version,compute_cap"):  # (an older driver has no mig.mode, memory.reserved or display_active)
        out = query(fields)
        if out is not None:
            break
    if out is None:
        raise no_gpu_refusal(platform)
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", run(["nvidia-smi"]) or "")
    cuda = (int(m.group(1)), int(m.group(2))) if m else ()
    gpus = []
    for line in out.strip().splitlines():
        row = dict(zip(fields.split(","), (c.strip() for c in line.split(","))))
        mib = lambda k: int(float(row[k])) * 2**20 if re.fullmatch(r"\d+(\.\d+)?", row.get(k, "")) else 0
        try:
            total = mib("memory.total")
            reserved = mib("memory.reserved") if "memory.reserved" in row else int(total * 0.023)  # (the driver keeps about 2%)
            cc = tuple(int(x) for x in row["compute_cap"].split("."))
            gpus.append(Gpu(int(row["index"]), row["name"], total - reserved, mib("memory.free"), cc, row["driver_version"], cuda,
                            row.get("display_active", "").lower() == "enabled", row.get("mig.mode.current", "").lower() == "enabled"))
        except (KeyError, ValueError):
            continue
    if not gpus:
        raise Refusal("nvidia-smi listed no usable GPU", "Check that the driver is loaded (run nvidia-smi), or reboot after installing it")
    return gpus


def pick_gpu(gpus, visible=None):
    """The GPU with the most free memory among those Glyd can use: new enough, and not in MIG mode unless CUDA_VISIBLE_DEVICES names a MIG
    slice (and among those it lists, where it is numbers): an older card, or a MIG-enabled one, with more memory is not the one to refuse
    for. Where none can be used, the freest of those new enough, then the freest, for the refusal."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") if visible is None else visible
    want = [int(x) for x in visible.split(",") if x.strip().isdigit()]
    store = [g for g in gpus if g.index in want] or gpus
    capable = [g for g in store if g.cc >= MIN_CAPABILITY]
    usable = [g for g in capable if not g.mig or "MIG-" in visible]
    return max(usable or capable or store, key=lambda g: g.free)


def other_users(run=_run):
    """[(program, bytes)] holding GPU memory (the compute programs nvidia-smi lists), the largest first."""
    apps = []
    for line in (run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]) or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[2].isdigit():
            apps.append((os.path.basename(parts[1]), int(parts[2]) * 2**20))
    return sorted(apps, key=lambda a: -a[1])


# --- versions and the driver -------------------------------------------------------------------------------------------------

def package_version(name):
    """An installed package's version from its metadata (nothing is imported), else None."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_cuda():
    """The CUDA version PyTorch was built for, (13, 0), read from its version.py without importing it (which takes seconds);
    None where PyTorch is not installed or is built without CUDA."""
    spec = importlib.util.find_spec("torch")
    for d in (spec.submodule_search_locations or []) if spec else []:
        try:
            with open(os.path.join(d, "version.py")) as f:
                m = re.search(r"^cuda\b[^=\n]*=\s*'(\d+)\.(\d+)", f.read(), re.M)
        except OSError:
            continue
        return (int(m.group(1)), int(m.group(2))) if m else None
    return None


def check_driver(gpu, built):
    """A Refusal where the driver cannot run the CUDA PyTorch was built for (an older major); a warning string where only the minor
    is older (CUDA's minor-version compatibility covers most of it); else None."""
    if not built or not gpu.cuda or gpu.cuda >= built:
        return None
    want = MIN_DRIVER.get(max([k for k in MIN_DRIVER if k <= built], default=None))
    fix = f"Update the NVIDIA driver to {want} or newer (Ubuntu: {DRIVER_PACKAGE.format(want=want)}; other systems: https://www.nvidia.com/drivers)" if want else "Update the NVIDIA driver (https://www.nvidia.com/drivers)"
    what = f"your NVIDIA driver {gpu.driver} runs CUDA {gpu.cuda[0]}.{gpu.cuda[1]}, and the PyTorch installed here was built for CUDA {built[0]}.{built[1]}"
    if gpu.cuda[0] < built[0]:
        raise Refusal(what, fix)
    return f"{what}. It may work through CUDA's minor-version compatibility; if it does not: {fix[0].lower() + fix[1:]}"


def gpu_library(built, environ=None):
    """Where Glyd's GPU library is for PyTorch's CUDA major (GLYD_GPU_LIB, else libglyd_gpu_cudaN.so beside glyd_gpu); None where
    it is not."""
    env = os.environ if environ is None else environ
    path = env.get("GLYD_GPU_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"libglyd_gpu_cuda{built[0] if built else 0}.so")
    return path if os.path.exists(path) else None


def have_nvcc(environ=None, which=shutil.which):
    """A CUDA compiler where FlashInfer's JIT would look for one: CUDA_HOME, CUDA_PATH, PATH, /usr/local/cuda."""
    env = os.environ if environ is None else environ
    homes = [env.get("CUDA_HOME"), env.get("CUDA_PATH"), "/usr/local/cuda"]
    return any(h and os.path.exists(os.path.join(h, "bin", "nvcc")) for h in homes) or bool(which("nvcc"))


def have_cc(environ=None, which=shutil.which):
    """A C compiler where Triton looks for one: $CC, else gcc or clang on PATH (vLLM 0.30 builds Triton launchers at warmup, eager too)."""
    env = os.environ if environ is None else environ
    return bool(which(env["CC"]) if env.get("CC") else which("gcc") or which("clang"))


def have_zig():
    """Whether ziglang (a C compiler from PyPI, which the installer adds where a machine has none) is in this Python: nothing is imported."""
    return importlib.util.find_spec("ziglang") is not None


def python_headers():
    """Whether this Python has its C headers (Python.h, which Triton's launcher includes); a uv-managed Python does."""
    return os.path.exists(os.path.join(sysconfig.get_path("include") or "", "Python.h"))


def compiler_hint(os_release=None):
    """The command that installs a C compiler and Python's headers, for this Linux distribution."""
    if os_release is None:
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
        except OSError:
            os_release = ""
    ids = " ".join(re.findall(r'^ID(?:_LIKE)?="?([^"\n]*)', os_release, re.M)).lower().split()
    if {"debian", "ubuntu"} & set(ids):
        return "Install one: sudo apt install build-essential python3-dev"
    if {"fedora", "rhel", "centos"} & set(ids):
        return "Install one: sudo dnf install gcc python3-devel"
    if {"arch"} & set(ids):
        return "Install one: sudo pacman -S base-devel"
    return "Install gcc (or clang) and your Python's development headers with your distribution's package manager"


def vllm_ready():
    """vLLM's version, or a Refusal where it is not installed or is not the minor release the plugin is tested with (from metadata:
    nothing is imported)."""
    v = package_version("vllm")
    if v is None:
        raise Refusal("vLLM is not installed in this Python environment", f'Run the installer, which adds it: {INSTALLER}   (or, in a virtual environment: pip install "glyd[vllm]")')
    if not tested(v):
        raise Refusal(f"this is vLLM {v}, and Glyd is tested with vLLM {TESTED}", f'Run the installer, which installs the vLLM Glyd is tested with: {INSTALLER}')
    return v


def setup_checks(need_vllm=True, gpus=None):
    """The checks before a model is looked at: a GPU (the freest), its compute capability, vLLM, a C compiler and Python's headers, a
    driver new enough for PyTorch's CUDA, and Glyd's library. (gpu, [warnings]), or a Refusal."""
    gpus = gpus or probe_gpus()
    gpu = pick_gpu(gpus)
    if gpu.cc < MIN_CAPABILITY:
        raise Refusal(f"{gpu.name} is too old for Glyd (compute capability {gpu.cc[0]}.{gpu.cc[1]}); it needs an NVIDIA Ampere GPU or newer"
                      + (f" (none of this computer's {len(gpus)} GPUs is)" if len(gpus) > 1 else ""),
                      "RTX 30 series, A10, A100, L4, RTX 40 series, H100 and newer work")
    if gpu.mig and "MIG-" not in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise Refusal(f"MIG is on for {gpu.name}: CUDA sees one slice of it, and glyd does not size a slice",
                      "Use the whole GPU (an administrator turns MIG off with: sudo nvidia-smi -i " + str(gpu.index) + " -mig 0), or name a slice with CUDA_VISIBLE_DEVICES=MIG-... and set the memory by hand:\n"
                      "glyd run MODEL -- --gpu-memory-utilization 0.8")
    if gpu.total <= 0:
        raise Refusal(f"nvidia-smi did not say how much memory {gpu.name} has", "Run nvidia-smi to see what it prints; a driver in an odd state (or MIG) can leave the memory out")
    if need_vllm:
        vllm_ready()
        if not (have_cc() or have_zig()):
            raise Refusal("vLLM needs a C compiler to start (Triton builds its GPU launchers with one), and this machine has none",
                          compiler_hint() + f". Or run the installer again, which adds a compiler from PyPI where there is none: {INSTALLER}")
        if not python_headers():
            raise Refusal("this Python has no C headers (Python.h), which vLLM's Triton launchers need", compiler_hint() + " (or install Glyd with the installer, which uses uv's own Python)")
    built = torch_cuda()
    warning = check_driver(gpu, built)
    if built and gpu_library(built) is None:
        raise Refusal(f"Glyd's GPU library for CUDA {built[0]} is not in this install", f"Run the installer again: {INSTALLER}   (the Linux wheels carry the library)")
    return gpu, [warning] if warning else []


# --- the model ---------------------------------------------------------------------------------------------------------------

@dataclass
class Model:
    repo: str  # "Qwen/Qwen3-8B", or a directory
    name: str  # "Qwen3-8B"
    config: dict
    bf16: int  # bytes of the checkpoint's weights (bf16)
    lin: int  # of them, the decoder Linears and experts: what Glyd packs (0 where the config does not say)
    other: int  # embeddings, LM head, norms
    moe: bool
    kv_token: int  # KV cache bytes a token, bf16 (0 where the config does not say)
    max_len: int  # the model's own context
    buffer: int = 0  # the plugin's decode buffer
    files: list = field(default_factory=list)  # [(path, bytes)] to download
    saved: bool = False  # a glyd save: its files as they are
    local: bool = False  # a directory, or a snapshot already downloaded
    dtype: str = "bf16"  # what the checkpoint holds most of: "bf16", or "f16" (float16: the plugin runs bfloat16)
    tables: int = 0  # of `other`, the embedding and LM head (one table where tied): what the plugin packs too
    kv_window: int = 0  # of kv_token, the sliding-window layers' bytes a token: they keep `window` tokens at most
    window: int = 0

    @property
    def text(self):
        return text_config(self.config)


def _experts(t):
    return next((t[n] for n in ("num_experts", "num_local_experts", "n_routed_experts") if t.get(n)), 0)


def _ffn(t):
    """A dense MLP's intermediate size: LFM2's block_ff_dim as its blocks size it (two thirds, the multiplier, rounded up to multiple_of)."""
    ff = t.get("intermediate_size") or t.get("moe_intermediate_size") or 0
    if t.get("block_auto_adjust_ff_dim"):
        ff = int(int(2 * ff / 3) * (t.get("block_ffn_dim_multiplier") or 1))
        ff = (t.get("block_multiple_of") or 1) * -(-ff // (t.get("block_multiple_of") or 1))
    return ff


def _ffn_kind(t, i, E):
    """Layer i's feed-forward: "dense", "moe" (routed experts, with any shared ones), "both" (Gemma 4's MLP beside its experts) or "" (none)."""
    pattern = t.get("hybrid_override_pattern")
    if isinstance(pattern, str) and len(pattern) == t["num_hidden_layers"]:  # Nemotron-H
        return {"-": "dense", "E": "moe"}.get(pattern[i], "")
    blocks = t.get("layers_block_type")
    if isinstance(blocks, list) and len(blocks) == t["num_hidden_layers"]:  # Nemotron 3.5
        return {"mlp": "dense", "moe": "moe"}.get(blocks[i], "")
    if E <= 1:
        return "dense"
    if t.get("enable_moe_block"):
        return "both"
    kinds = t.get("mlp_layer_types")
    if isinstance(kinds, list) and len(kinds) == t["num_hidden_layers"]:
        return "dense" if kinds[i] == "dense" else "moe"
    if i in (t.get("mlp_only_layers") or ()):
        return "dense"
    if t.get("expert_layer_period"):  # Jamba
        return "moe" if i % t["expert_layer_period"] == (t.get("expert_layer_offset") or 0) else "dense"
    if i < (t.get("first_k_dense_replace") or t.get("num_dense_layers") or 0):
        return "dense"
    step = t.get("decoder_sparse_step") or 1
    return "moe" if (i + 1) % step == 0 else "dense"


def linear_bytes(t):
    """A model's Linears and experts (bf16 bytes, as vLLM holds them: what the plugin packs), its embedding tables (bf16 bytes), and whether it
    has experts, from its text config t, layer by layer (fit.mixers): attention (MLA's, Gemma 4's full layers with heads of their own, an
    output gate, a KV-sharing layer's query alone), linear attention, Mamba and LFM2's convolution; a dense MLP, routed experts and shared
    ones, gated (three matrices) or not (relu2: two). (0, 0, False) where the config does not say."""
    try:
        h, L, nh = t["hidden_size"], t["num_hidden_layers"], t["num_attention_heads"]
        E = _experts(t)
        moe_i = t.get("moe_intermediate_size") or t.get("expert_intermediate_size") or t.get("intermediate_size") or 0
        shared = t.get("shared_expert_intermediate_size") or t.get("moe_shared_expert_intermediate_size") or (t.get("n_shared_experts") or 0) * moe_i or t.get("shared_intermediate_size") or 0
        mats = 2 if {"relu2", "xielu"} & {t.get("mlp_hidden_act"), t.get("hidden_act")} else 3  # (no gate: Nemotron-H's relu2, Apertus's xIELU)
        ff, own = _ffn(t), L - (t.get("num_kv_shared_layers") or 0)
        total = 0
        for i, kind in enumerate(mixers(t)):
            if kind in ("full", "sliding") and t.get("kv_lora_rank"):  # MLA
                qk, rope, v = t.get("qk_nope_head_dim") or 0, t.get("qk_rope_head_dim") or 0, t.get("v_head_dim") or 0
                qr = t.get("q_lora_rank")
                total += (h * qr + qr * nh * (qk + rope) if qr else h * nh * (qk + rope)) + h * (t["kv_lora_rank"] + rope) + t["kv_lora_rank"] * nh * (qk + v) + nh * v * h
            elif kind in ("full", "sliding"):
                hd, nkv = attention_heads(t, kind)
                total += h * nh * hd * (2 if t.get("attn_output_gate") else 1) + (2 * h * nkv * hd if i < own else 0) + nh * hd * h
            elif kind == "linear":  # Gated DeltaNet: q, k, v and the gate in, the output out (Ling 3's: the attention's heads)
                kd = (t.get("linear_num_key_heads") or nh) * (t.get("linear_key_head_dim") or t.get("head_dim") or h // nh)
                vd = (t.get("linear_num_value_heads") or nh) * (t.get("linear_value_head_dim") or t.get("head_dim") or h // nh)
                total += h * (2 * kd + 2 * vd) + vd * h
            elif kind == "mamba" and t.get("mamba_num_heads"):  # Mamba2 (Nemotron-H)
                d = t["mamba_num_heads"] * t["mamba_head_dim"]
                total += h * (2 * d + 2 * (t.get("n_groups") or 1) * t["ssm_state_size"] + t["mamba_num_heads"]) + d * h
            elif kind == "mamba":  # Mamba (Jamba)
                d, r = (t.get("mamba_expand") or 2) * h, t.get("mamba_dt_rank") or -(-h // 16)
                total += 3 * h * d + d * (r + 2 * (t.get("mamba_d_state") or 16)) + r * d
            elif kind == "conv":  # LFM2: in (3 x hidden) and out
                total += 4 * h * h
            f = _ffn_kind(t, i, E)
            if f in ("dense", "both"):
                total += mats * h * ff * (2 if t.get("use_double_wide_mlp") and i >= own else 1)
            if f in ("moe", "both"):
                total += mats * h * (E * moe_i + shared)
            total += 2 * h * (t.get("hidden_size_per_layer_input") or 0)  # (Gemma 4's per-layer input gate and projection)
        return 2 * total, _tables(t), bool(E > 1)
    except (KeyError, TypeError, ZeroDivisionError):
        return 0, 0, False


def _tables(t):
    """The bf16 bytes of a model's embedding and LM head (the one table where its config ties them) and of Gemma 4's per-layer embedding."""
    per_layer = (t.get("vocab_size_per_layer_input") or t["vocab_size"]) * t["num_hidden_layers"] * (t.get("hidden_size_per_layer_input") or 0)
    return 2 * (t["vocab_size"] * t["hidden_size"] * (1 if t.get("tie_word_embeddings", t.get("tie_embedding")) else 2) + per_layer)


def table_bytes(t):
    """Of a model's tables (_tables), those the plugin packs too, from its text config t: all of them where the hidden size is a multiple
    of 64, else none; 0 where the config does not say."""
    try:
        return 0 if t["hidden_size"] % 64 else _tables(t)
    except (KeyError, TypeError):
        return 0


def buffer_bytes(t):
    """The plugin's decode buffer (bf16): the largest matrix it decodes, a layer's merged gate and up (2 x intermediate x hidden)."""
    try:
        E = _experts(t)
        wide = 2 if t.get("use_double_wide_mlp") and t.get("num_kv_shared_layers") else 1
        return 2 * 2 * max((t.get("moe_intermediate_size") or t["intermediate_size"]) * E if E > 1 else 0, _ffn(t) * wide) * t["hidden_size"]
    except (KeyError, TypeError):
        return 0


def model_of(repo, config, bf16=None, files=(), saved=False, local=False, dtype="bf16"):
    """A Model from a config, and the checkpoint's bytes where they are known (else the config's own count)."""
    t = text_config(config)
    if "tie_word_embeddings" not in t and "tie_embedding" not in t and "tie_word_embeddings" in config:  # (said once, at the top: Qwen3.5-27B, EXAONE 4.5)
        t = {**t, "tie_word_embeddings": config["tie_word_embeddings"]}
    lin, other, moe = linear_bytes(t)
    if bf16 is None:
        bf16 = lin + other
    else:
        lin = min(lin, bf16)  # (the count is an upper bound: a model with no gate, or tied weights it does not know)
        other = bf16 - lin
    try:
        full, slide, window = kv_parts(config)
    except (KeyError, TypeError, ZeroDivisionError):
        full = slide = window = 0
    return Model(repo, repo.rstrip("/").split("/")[-1], config, bf16, lin, other, moe, full + slide, int(t.get("max_position_embeddings") or 0), buffer_bytes(t), list(files), saved, local, dtype,
                 min(other, table_bytes(t)), slide, window + BATCH if slide else 0)


def kv_need(m, tokens):
    """Bytes of KV cache a `tokens`-token context takes (a sliding-window layer's: its window at most)."""
    return (m.kv_token - m.kv_window) * tokens + m.kv_window * min(tokens, m.window)


def kv_fit(m, room):
    """The longest context whose KV cache fits `room` bytes (kv_need's inverse); 0 where the config does not say."""
    if not m.kv_token or room <= 0:
        return 0
    if room <= m.kv_token * m.window or m.kv_token == m.kv_window:
        return int(room // m.kv_token) if room <= m.kv_token * m.window else 1 << 30
    return int(m.window + (room - m.kv_token * m.window) // (m.kv_token - m.kv_window))


def layout_for(cc, lin, other, total, moe=False):
    """The layout the plugin picks for this GPU (kernels.best_layout, from the compute capability rather than a device): the
    tiered layout on Ada, for a mixture of experts on an A10 too, and wherever only it fits in the memory less 2 GiB; else the
    12-bit one."""
    room = total - 2 * GiB
    banded, twelve = lin * BITS["mma"] / 16 + other, lin * BITS["mma12"] / 16 + other  # (lin: what is packed, tables too; other: what is not)
    return "mma" if twelve > room >= banded or cc == (8, 9) or (moe and cc == (8, 6)) else "mma12"


def weights_on_gpu(m, layout):
    """Bytes of GPU memory the model's weights take with Glyd (a glyd save: its files)."""
    if m.saved:
        return m.bf16 + m.buffer + EXTRA
    if not m.lin:  # (a config that does not say: the mean measured over the models, all of it packed)
        return m.bf16 * RATIO + 0.3 * GiB
    return (m.lin + m.tables) * BITS[layout] / 16 + m.other - m.tables + m.buffer + EXTRA


# --- the Hub -----------------------------------------------------------------------------------------------------------------

def hub_cache():
    """The Hugging Face hub cache directory (HF_HUB_CACHE, else HF_HOME/hub)."""
    env = os.environ
    home = env.get("HF_HOME") or os.path.join(env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "huggingface")
    return env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE") or os.path.join(home, "hub")


def _snapshots(repo):
    return os.path.join(hub_cache(), "models--" + repo.replace("/", "--"), "snapshots")


def cached_snapshot(repo):
    """A repo's newest downloaded snapshot (with its config.json), else None."""
    found = glob.glob(os.path.join(_snapshots(repo), "*", "config.json"))
    return os.path.dirname(max(found, key=os.path.getmtime)) if found else None


def cached_bytes(repo, files):
    """Of [(path, bytes)], the bytes already in the cache's snapshots (a file there of the size the Hub gave)."""
    have = 0
    for p, n in files:
        for f in glob.glob(os.path.join(_snapshots(repo), "*", p)):
            try:
                if os.path.getsize(f) == n:
                    have += n
                    break
            except OSError:
                pass
    return have


def download_files(files):
    """Of the Hub's [(path, bytes)], what vLLM reads: the checkpoint's safetensors (fit.checkpoint), and the small files at the top
    (config, index, tokenizer, chat template, generation config)."""
    keep = {p for p, _ in checkpoint(files)}
    small = lambda p: "/" not in p and (p.endswith((".json", ".txt", ".model", ".jinja")) or p.startswith("tokenizer"))
    return [(p, n) for p, n in files if p in keep or small(p)]


def hub_refusal(repo, e):
    """A Hub error in plain words."""
    if getattr(e, "status", None) in (401, 403, 404):
        return Refusal(f"the Hugging Face Hub would not show {repo}: the name is wrong, or it is a gated or private model",
                       f"Check the name (OWNER/NAME, as in Qwen/Qwen3.5-9B). For a gated model, accept its licence at {HUB}/{repo}, then run: glyd login   (it takes a token from {HUB}/settings/tokens)")
    return Refusal(f"cannot read {repo} from the Hugging Face Hub ({e})", "Check the network connection and run this again")


def load_model(name, hub=_hub, local=_local):
    """The model, from a directory (its config and safetensors headers) or the Hub's metadata; offline, from a snapshot already in the
    cache. A Refusal, in plain words: no such repo, a gated one, weights that are not bf16."""
    cached = False
    try:
        if os.path.isdir(name):
            config, params, files = local(name)
        else:
            try:
                config, params, files = hub(name)
            except HubError as e:
                raise hub_refusal(name, e) from e
            except OSError as e:  # (no network: a model downloaded before is read from the cache)
                snap = cached_snapshot(name)
                if snap is None:
                    raise hub_refusal(name, e) from e
                (config, params, files), cached = local(snap), True
    except (OSError, ValueError, KeyError) as e:  # (a directory with no config.json, a bad header)
        raise Refusal(f"cannot read {name}: {e}", "Check the path, or give a Hugging Face name (OWNER/NAME)") from e
    weights = sum(n for _, n in checkpoint(files))
    if not weights:
        raise Refusal(f"{name} has no safetensors weights", "Glyd reads .safetensors checkpoints, which most models on the Hub have")
    saved = any(p == MANIFEST for p, _ in files)
    bf16 = 2 * (params.get("BF16", 0) + params.get("F16", 0)) if params else weights
    if not saved and weights - bf16 >= 0.05 * weights:
        if params.get("F32", 0) * 4 > 0.5 * weights:
            raise Refusal(f"{name} is stored in 32-bit floats, and Glyd packs bf16 weights", "Run a bf16 version of the model if there is one (most models publish one)")
        fmt = "FP8" if params.get("F8_E4M3", 0) > max(params.get("U8", 0), params.get("I8", 0)) else "4-bit"
        raise Refusal(f"{name} is already quantized ({fmt}), and Glyd packs bf16 weights", "Run the model's bf16 version: usually the same name without -FP8, -AWQ or -GPTQ")
    local_copy = os.path.isdir(name) or cached
    dtype = "f16" if not saved and params.get("F16", 0) > params.get("BF16", 0) else "bf16"
    return model_of(name, config, weights if saved else bf16, [] if local_copy else download_files(files), saved, local_copy, dtype)


def check_dtype(m, given=None):
    """A Refusal for a float16 checkpoint, before its download: the plugin runs bfloat16 and vLLM stops the load with "float16 is not
    supported for quantization method glyd". `-- --dtype bfloat16` makes vLLM convert the weights, which is the user's to ask for."""
    if m.dtype == "f16" and str((given or {}).get("dtype", "")).lower() not in ("bfloat16", "bf16"):
        raise Refusal(f"{m.name} is stored in float16, and Glyd packs bfloat16 weights",
                      "Use the model's bf16 version if it has one, or add  -- --dtype bfloat16  and vLLM converts the weights (the result is then not bit for bit the checkpoint's)")


# --- the settings ------------------------------------------------------------------------------------------------------------

@dataclass
class Settings:
    util: float = 0.0  # gpu-memory-utilization (0: not set by us)
    context: int = 0  # max-model-len (0: vLLM's own "auto")
    eager: bool = True
    layout: str = "mma12"
    weights: int = 0  # bytes on the GPU with Glyd
    needs: int = 0  # free memory to start with, at MIN_CONTEXT
    kv_tokens: int = 0  # tokens the KV cache is expected to hold
    context_why: str = ""  # why the context is what it is
    tool_parser: str = ""
    reasoning_parser: str = ""
    env: dict = field(default_factory=dict)  # what to add to the server's environment (only where the user has set none)
    notes: list = field(default_factory=list)  # why a setting is what it is, for the summary line
    given: dict = field(default_factory=dict)  # the vLLM flags the user passed, which win
    max_seqs: int = 0  # active requests for a local chat; zero keeps the serving default
    local_only: bool = False  # the server answers this computer's own page and programs only (page.py): vLLM's CORS is limited to the page's origin too


PARSERS = {  # model_type: (tool-call parser, reasoning parser), vLLM 0.30.0's names, as each 2026 family's model card or vLLM's recipe for it gives them
    "qwen3_5": ("qwen3_coder", "qwen3"), "qwen3_5_moe": ("qwen3_coder", "qwen3"),  # Qwen3.5, 3.6, 3.8 and their finetunes
    "qwen3_next": ("qwen3_coder", ""),  # Qwen3-Coder-Next
    "gemma4": ("gemma4", "gemma4"), "gemma4_unified": ("gemma4", "gemma4"),
    "nemotron_h": ("qwen3_coder", "nemotron_v3"), "NemotronH_Nano_Omni_Reasoning_V3": ("qwen3_coder", "nemotron_v3"),  # Nemotron 3 and 3.5
    "glm_moe_dsa": ("glm47", "glm47"), "glm5_next": ("glm47", "glm47"), "glm4_moe_lite": ("glm47", "glm45"),  # GLM-5.x, GLM-5.3-Flash, GLM-4.7-Flash
    "minimax_m2": ("minimax_m2", "minimax_m2"), "minimax_m3_vl": ("minimax_m3", "minimax_m3"),
    "kimi_k25": ("kimi_k2", "kimi_k2"), "kimi_k3": ("kimi_k3", "kimi_k3"),
    "deepseek_v4": ("deepseek_v4", "deepseek_v4"), "deepseek_v41": ("deepseek_v41", "deepseek_v41"),
    "mimo_v2": ("mimo", "mimo"), "bailing_hybrid": ("ling3", "ling3"), "exaone4_5": ("hermes", "qwen3"),
    "lfm2": ("lfm2", ""), "lfm2_moe": ("lfm2", ""),  # (a Thinking model: qwen3's reasoning parser, below)
    "apertus": ("apertus", ""),  # (its thinking mode is a chat-template switch without tool calls)
    "muse_glimmer": ("muse_glimmer", "muse_glimmer"), "inkling_mm_model": ("inkling", "inkling"),
    "jamba": ("hermes", ""), "mellum": ("hermes", ""), "cohere2_vision": ("cohere_command4", "cohere_command4"),
    "hy_v3": ("hy_v3", "hy_v3"), "hy_v4": ("hy_v4", "hy_v4"), "step3p5": ("step3p5", "step3p5"), "step3p7": ("step3p5", "step3p5"),
}


def parsers(m):
    """(tool-call parser, reasoning parser) for a model by its family (its config's model_type, else its text config's): vLLM 0.30.0's names,
    "" for none. A model with no tool parser still chats; one this table does not know gets neither."""
    kinds = (m.config.get("model_type") or "", m.text.get("model_type") or "")
    name = m.name.lower()
    if "deepseek-r1-distill" in name:
        return "", "deepseek_r1"
    kind = next((k for k in kinds if k in PARSERS), "")
    if kind:
        tool, think = PARSERS[kind]
        if kind.startswith("lfm2") and "thinking" in name:  # (LFM2.5's Thinking models, as Liquid's vLLM guide gives them)
            think = "qwen3"
        return tool, think
    if "granite-4.2" in name:  # Granite 4.2 (model_type granite, as Granite 3): its card's parsers
        return "qwen3_coder", "nemotron_v3"
    if "granite-4.1" in name:
        return "granite4", ""
    if "minicpm5" in name:  # (model_type llama; its chat template opens the thinking in the prompt: without a reasoning parser it reached the answer, run-families-2026-10-06)
        return "minicpm5", "qwen3"
    kind = kinds[1] or kinds[0]
    if kind in ("qwen3", "qwen3_moe"):
        if "coder" in name:
            return "qwen3_coder", ""
        return "hermes", "" if "instruct-2507" in name else "qwen3"
    if kind == "qwen2" and "qwen2.5" in name:
        return "hermes", ""
    if kind == "llama" and re.search(r"llama-3\.[123]", name):
        return "llama3_json", ""
    if kind == "mistral":
        return "mistral", ""
    return "", ""


def flags_given(extra):
    """The vLLM flags passed after `--`: {name: value, or True}, by the name with hyphens; the no- form of a flag is its opposite
    (--no-enforce-eager: {"enforce-eager": False})."""
    short = {"q": "quantization", "tp": "tensor-parallel-size", "pp": "pipeline-parallel-size", "dp": "data-parallel-size"}
    given, i = {}, 0
    while i < len(extra):
        tok = extra[i]
        i += 1
        if not tok.startswith("-") or tok in ("-", "--"):
            continue
        name, has, value = tok.lstrip("-").partition("=")
        name, on = name.replace("_", "-"), True
        if tok.startswith("--") and name.startswith("no-"):
            name, on = name[3:], False
        elif not tok.startswith("--"):
            name = short.get(name, name)
        if not has and on and i < len(extra) and not extra[i].startswith("-"):
            value, has = extra[i], True
            i += 1
        given[name] = value if has and on else on
    return given


def number(v):
    """A flag's number as vLLM reads it: 8192, 8k (8000) or 8K (8192); None where it is not one."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmM]?)", str(v).strip())
    return float(m.group(1)) * {"": 1, "k": 1000, "K": 1024, "m": 10**6, "M": 2**20}[m.group(2)] if m else None


def need(weights, m, gpu, context=MIN_CONTEXT, non_kv=NON_KV):
    """The free memory a start needs: the weights, vLLM's other memory, a `context`-token KV cache, the CUDA context, the desktop's
    headroom, and what rounding the fraction down to 0.01 can cost."""
    return int(weights + non_kv + kv_need(m, context) / KV_FIT + CTX + HEADROOM + 0.01 * gpu.total)


def take_secret(extra, names, what):
    """(the value, the flags without it): a vLLM flag that carries a secret (`--api-key KEY`, `--hf-token=TOKEN`) among the flags after `--`
    is taken out, so that the secret is not on the server's command line (`ps`, the log): the caller hands it over in the environment. None
    where the flag is not there, or has no value (`--hf-token` alone means the saved login: it stays). Several values go through no one
    variable (vLLM takes several keys after `--api-key`, and VLLM_API_KEY holds one): a Refusal."""
    out, value, i, extra = [], None, 0, list(extra)
    while i < len(extra):
        tok = extra[i]
        name, eq, v = tok.partition("=")
        if name in names:
            vals, j = ([v] if eq else []), i + 1
            while not eq and j < len(extra) and not extra[j].startswith("-"):
                vals.append(extra[j])
                j += 1
            if len(vals) > 1:
                raise Refusal(f"glyd passes one {what}, and {name} was given {len(vals)}",
                              "Give one (VLLM_API_KEY=KEY glyd serve ... for an API key), or start vLLM yourself for several keys: see \"Advanced: vllm serve by hand\" in gpu/vllm/README.md")
            if len(vals) == 1:
                value, i = vals[0], j
                continue
        out.append(tok)
        i += 1
    return value, out


def take_api_key(extra, environ=None):
    """(the API key, the vLLM flags without it): `--api-key KEY` (or `--api-key=KEY`) among the flags after `--` is taken out, so that the key
    is not on the server's command line (`ps`, the log): it reaches the server in VLLM_API_KEY. Without one, the key is VLLM_API_KEY's, or
    None. Several keys after one `--api-key` are refused (take_secret)."""
    env = os.environ if environ is None else environ
    key, out = take_secret(extra, ("--api-key", "--api_key"), "API key")
    return (key or env.get("VLLM_API_KEY") or None), out


def parse_port(v):
    """A port number from a flag's value, or a Refusal in words (it was int(None) and an OverflowError)."""
    n = number(v)
    if n is None or n != int(n) or not 1 <= n <= 65535:
        raise Refusal(f"{v!r} is not a port number", "A port is a whole number from 1 to 65535: --port 8000")
    return int(n)


def path_twin(environ=None, me=None):
    """(the glyd that comes first on PATH, this program's own path on PATH or its real one) where another program named glyd (the Rust
    compression program) comes first, so that `glyd run` typed alone reaches it; None where this program is first, or none is on PATH."""
    env = os.environ if environ is None else environ
    mine = os.path.realpath(me or os.path.join(os.path.dirname(sys.executable), "glyd"))  # (the tool environment's own bin has it beside its Python)
    first, own = None, None
    for d in (env.get("PATH", "") or "").split(os.pathsep):
        exe = os.path.join(d, "glyd") if d else ""
        if not exe or not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            continue
        if os.path.realpath(exe) == mine:
            own = own or exe
        elif first is None and own is None:
            first = exe
    return (first, own or mine) if first else None


def footprint(m, gpu, layouts=None, context=MIN_CONTEXT, non_kv=NON_KV):
    """(weights on the GPU with Glyd, free memory needed to start with a `context`-token chat) in the layout that takes the least."""
    w = min(weights_on_gpu(m, l) for l in layouts or ["mma", "mma12"])
    return int(w), need(w, m, gpu, context, non_kv)


def settings(m, gpu, mode="run", context=None, nvcc=True, environ=None, given=None, cc=""):
    """The settings for this model on this GPU, or a Refusal where it does not fit: compiled where it leaves the chat CONTEXT_WANT tokens
    (or the model's own length, or the context asked for), else eager, which needs 1.75 GiB less, where that makes the chat longer (the
    user's --enforce-eager or --no-enforce-eager is taken as it is)."""
    given = given or {}
    if "enforce-eager" in given:
        return _settings(m, gpu, mode, context, nvcc, environ, given, cc, bool(given["enforce-eager"]))
    try:
        c = _settings(m, gpu, mode, context, nvcc, environ, given, cc, False)
    except Refusal:
        c = None
    want = min(CONTEXT_WANT, m.max_len) if m.max_len else CONTEXT_WANT
    if c is not None and (context or "max-model-len" in given or not c.context or c.context >= want):
        return c
    e = _settings(m, gpu, mode, context, nvcc, environ, given, cc, True)  # (eager's own refusal where it does not fit either)
    if c is not None and e.context <= c.context:
        return c
    e.notes.append("eager: compiled leaves room for a shorter chat")
    return e


def mamba2(m):
    """Whether the model has Mamba2 layers (heads; Mamba v1, as Jamba's, has none). vLLM 0.30's Mamba2 scan asks for memory outside its
    profiled budget while capturing CUDA graphs (a float32 state of heads x head_dim x state a sequence: 960 MiB for
    NVIDIA-Nemotron-3-Nano-4B), and glyd serve at 92% of an RTX 4090 ran out of it, at 88% too; glyd run's share (71%) started every time
    (run-families-2026-10-06 pod4, pod5)."""
    t = m.text
    return "mamba" in mixers(t) and any(t.get(k) for k in ("mamba_num_heads", "mamba_n_heads"))


def _settings(m, gpu, mode, context, nvcc, environ, given, cc, eager):
    """settings() in one mode. `mode` is "run" (one user's chat) or "serve";
    `context` and `given` (flags_given) are what the user chose, each used as it is (a context under MIN_CONTEXT is also what the memory
    check needs). `eager`: the server is started eager (compiled otherwise), and the KV cache is counted for that. The layout is
    chosen here and handed to the plugin (GLYD_LAYOUT), so the plugin packs what was counted: the plugin's own choice for the GPU, or the
    tiered layout where the 12-bit one leaves less KV cache than a CONTEXT_WANT-token chat and the tiered one more. `cc` is a C compiler
    for Triton (run.py's ziglang wrapper) where the machine has none."""
    env = os.environ if environ is None else environ
    s = Settings(given=given, max_seqs=8 if mode == "run" else 0)
    s.tool_parser, s.reasoning_parser = parsers(m)
    kv, T = m.kv_token, gpu.total
    if "VLLM_USE_FLASHINFER_SAMPLER" not in env:  # (FlashInfer's sampler compiles with nvcc at the first request that samples, and the same tokens a
        s.env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # second come from PyTorch's: 21.1 against 21.2 on an L4 with Qwen3-8B)
        if not nvcc:
            s.notes.append("PyTorch sampler (no CUDA toolkit)")
    if "VLLM_NO_USAGE_STATS" not in env and "DO_NOT_TRACK" not in env:
        s.env["VLLM_NO_USAGE_STATS"] = "1"
    if cc and "CC" not in env:
        s.env["CC"] = cc
        s.notes.append("ziglang as the C compiler (no gcc)")
    if mode == "run" and "GLYD_KV" not in env and "additional-config" not in given:  # (one user's chat keeps vLLM's own KV cache: the lossless one's first start makes tables, minutes on an L4, for about 1% of speed for one user; GLYD_KV=auto turns it on, and glyd serve has it)
        s.env["GLYD_KV"] = "off"
    auto, mine = layout_for(gpu.cc, m.lin + m.tables, m.other - m.tables, T, m.moe), (env.get("GLYD_LAYOUT") or "").strip().lower()
    layouts = [mine] if mine in BITS else [auto] + (["mma"] if auto == "mma12" and not m.saved else [])
    weights = {l: int(weights_on_gpu(m, l)) for l in layouts}
    s.eager = eager
    non_kv = (NON_KV if s.eager else NON_KV_COMPILED) + (HYBRID if {"linear", "mamba"} & set(mixers(m.text)) else 0)
    asked = int(number(given["max-model-len"]) or 0) if "max-model-len" in given else (context or 0)
    least = min(MIN_CONTEXT, asked) if asked > 0 else MIN_CONTEXT  # (a chat the user wants shorter needs less)
    tokens = lambda l, util: kv_fit(m, (util * T - weights[l] - non_kv) * KV_FIT)
    s.layout, s.weights = layouts[0], weights[layouts[0]]
    base, s.needs = footprint(m, gpu, layouts, least, non_kv)
    if any((number(given.get(k)) or 1) > 1 for k in ("tensor-parallel-size", "pipeline-parallel-size", "data-parallel-size")):
        s.notes.append("several GPUs: memory settings are yours")
        return s
    if "gpu-memory-utilization" in given:
        cap = float(number(given["gpu-memory-utilization"]) or 0)
        s.notes.append("memory use as you set it")
    elif s.needs > gpu.free:
        raise Refusal(f"{m.name} needs about {gb(s.needs)} of GPU memory with Glyd ({gb(base)} of weights and room for a {least:,}-token chat); your GPU has {gb(gpu.free)} free")
    else:
        cap = min(UTIL_MAX, int((gpu.free - CTX - HEADROOM) / T * 100) / 100)
    want = min(CONTEXT_WANT, m.max_len) if m.max_len else CONTEXT_WANT
    if len(layouts) > 1 and tokens(s.layout, cap) < want and tokens("mma", cap) > tokens(s.layout, cap):
        s.layout = "mma"
        s.notes.append("the smallest layout (mma): smaller weights, room for a longer chat")
    s.weights, s.util = weights[s.layout], round(cap, 2)
    s.kv_tokens = tokens(s.layout, cap)
    if kv and "gpu-memory-utilization" not in given and s.kv_tokens < least:  # (vLLM's own 92% limit leaves less than the memory check counted on: a 14B on 22.9 GiB)
        raise Refusal(f"{m.name} with Glyd ({gb(base)} of weights) and a {least:,}-token chat need more than the {UTIL_MAX:.0%} of this GPU's memory that vLLM uses ({gb(UTIL_MAX * T)}); your GPU has {gb(gpu.free)} free")
    fits = s.kv_tokens // 1024 * 1024
    if "max-model-len" in given:
        s.context = int(number(given["max-model-len"]) or 0)
        s.context_why = "as you set it"
    elif context:
        if m.max_len and context > m.max_len:
            raise Refusal(f"{m.name}'s own window is {m.max_len:,} tokens, and --context {context:,} is more", f"Use --context {m.max_len:,} or less")
        if kv and context > s.kv_tokens:
            raise Refusal(f"a {context:,}-token context needs more memory than is free: at most {fits:,} tokens fit beside {m.name}", f"Use --context {fits:,} or less")
        s.context, s.context_why = context, "as you asked"
    elif kv:
        s.context = min(m.max_len, fits) if m.max_len else fits
        s.context_why = "the model's own limit" if m.max_len and m.max_len <= fits else "the most that fits"
    else:
        s.notes.append("context: vLLM's choice (the config does not say enough to size it)")
    if (mode == "run" or mamba2(m)) and kv and s.context and "gpu-memory-utilization" not in given:  # (one chat: two windows of KV cache, not the whole GPU; a Mamba2 model served too: mamba2())
        s.util = min(s.util, math.ceil((s.weights + non_kv + 2 * kv_need(m, s.context) / KV_FIT) / T * 100) / 100)
        s.kv_tokens = tokens(s.layout, s.util)
    if not m.saved and "GLYD_LAYOUT" not in env and "additional-config" not in given:
        s.env["GLYD_LAYOUT"] = s.layout
    return s


def vllm_args(m, s, host, port, extra=()):
    """`vllm serve`'s arguments for these settings; a flag the user gave (in `extra`, appended last) is not added here."""
    g = s.given
    args = [m.repo]
    for flag, value in (("quantization", "glyd"), ("host", host), ("port", str(port)), ("middleware", "glyd_gpu.page.ChatPage"),
                        ("max-model-len", str(s.context) if s.context else "auto"), ("gpu-memory-utilization", f"{s.util:.2f}" if s.util else ""),
                        ("max-num-seqs", str(s.max_seqs) if s.max_seqs else "")):
        if value and flag not in g:
            args += [f"--{flag}", value]
    if s.eager and "enforce-eager" not in g:
        args.append("--enforce-eager")
    if s.tool_parser and not {"enable-auto-tool-choice", "tool-call-parser"} & set(g):
        args += ["--enable-auto-tool-choice", "--tool-call-parser", s.tool_parser]
    if s.reasoning_parser and "reasoning-parser" not in g:
        args += ["--reasoning-parser", s.reasoning_parser]
    if s.local_only and "allowed-origins" not in g:  # (vLLM allows any origin by default: another web page's script could use this server from the user's browser)
        args += ["--allowed-origins", json.dumps(own_origins(port))]
    return args + list(extra)


def summary(s, gpu):
    """The one line of what was chosen and why."""
    bits = [f"{s.context:,}-token context" + (f" ({s.context_why})" if s.context_why else "") if s.context else "context chosen by vLLM"]
    bits.append("eager mode" if s.eager else "compiled mode (CUDA graphs)")
    if s.util:
        bits.append(f"{s.util * 100:.0f}% of GPU memory ({gb(s.util * gpu.total)})")
    if s.tool_parser:
        bits.append(f"tool calls ({s.tool_parser})")
    if s.reasoning_parser:
        bits.append(f"thinking shown apart ({s.reasoning_parser})")
    return ", ".join(bits) + "".join("; " + n for n in s.notes) + "."


# --- models to suggest -------------------------------------------------------------------------------------------------------

def _cfg(kind, h, L, I, nh, nkv, hd, V, tie, pos, nbytes=None, **more):
    """A model's config.json as the estimates read it; nbytes: its safetensors files' bytes on the Hub, where the config's own count would be
    off (Qwen3.5: a vision tower and linear-attention layers the count does not know)."""
    return {"model_type": kind, "hidden_size": h, "num_hidden_layers": L, "intermediate_size": I, "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
            "vocab_size": V, "tie_word_embeddings": tie, "max_position_embeddings": pos, **({"safetensors_bytes": nbytes} if nbytes else {}), **more}


_QWEN35 = lambda nv: dict(full_attention_interval=4, attn_output_gate=True, linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=nv, linear_value_head_dim=128)  # (Qwen3.5's linear-attention layers)
LADDERS = {  # models to suggest, smallest first, by their own config.json (the Qwen rungs: Qwen3.5, 2026, with their files' bytes on the Hub; the other
    # 2026 families' rungs: the keys the estimates read, and their tensors' bytes, from the checkpoints' safetensors headers)
    "qwen": [("Qwen/Qwen3.5-0.8B", _cfg("qwen3_5", 1024, 24, 3584, 8, 2, 256, 248320, True, 262144, 1_746_942_600, **_QWEN35(16))),
             ("Qwen/Qwen3.5-2B", _cfg("qwen3_5", 2048, 24, 6144, 8, 2, 256, 248320, True, 262144, 4_548_221_488, **_QWEN35(16))),
             ("Qwen/Qwen3.5-4B", _cfg("qwen3_5", 2560, 32, 9216, 16, 4, 256, 248320, True, 262144, 9_319_828_096, **_QWEN35(32))),
             ("Qwen/Qwen3.5-9B", _cfg("qwen3_5", 4096, 32, 12288, 16, 4, 256, 248320, False, 262144, 19_306_310_880, **_QWEN35(32))),
             ("Qwen/Qwen3.5-27B", _cfg("qwen3_5", 5120, 64, 17408, 24, 4, 256, 248320, False, 262144, 55_563_022_432, **_QWEN35(48)))],
    "gemma": [
        ("google/gemma-4-E2B-it", {"model_type": "gemma4", "hidden_size": 1536, "num_hidden_layers": 35, "intermediate_size": 6144, "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256, "vocab_size": 262144, "tie_word_embeddings": True, "max_position_embeddings": 131072, "sliding_window": 512, "global_head_dim": 512, "num_kv_shared_layers": 20, "use_double_wide_mlp": True, "hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144, "sliding_window_pattern": 5, "safetensors_bytes": 10246357958}),
        ("google/gemma-4-E4B-it", {"model_type": "gemma4", "hidden_size": 2560, "num_hidden_layers": 42, "intermediate_size": 10240, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256, "vocab_size": 262144, "tie_word_embeddings": True, "max_position_embeddings": 131072, "sliding_window": 512, "global_head_dim": 512, "num_kv_shared_layers": 18, "hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144, "sliding_window_pattern": 6, "safetensors_bytes": 15992314836}),
        ("google/gemma-4-12B-it", {"model_type": "gemma4_unified", "hidden_size": 3840, "num_hidden_layers": 48, "intermediate_size": 15360, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256, "vocab_size": 262144, "tie_word_embeddings": True, "max_position_embeddings": 262144, "sliding_window": 1024, "global_head_dim": 512, "attention_k_eq_v": True, "num_global_key_value_heads": 1, "vocab_size_per_layer_input": 262144, "sliding_window_pattern": 6, "safetensors_bytes": 23919460448}),
        ("google/gemma-4-31B-it", {"model_type": "gemma4", "hidden_size": 5376, "num_hidden_layers": 60, "intermediate_size": 21504, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 256, "vocab_size": 262144, "tie_word_embeddings": True, "max_position_embeddings": 262144, "sliding_window": 1024, "global_head_dim": 512, "attention_k_eq_v": True, "num_global_key_value_heads": 4, "vocab_size_per_layer_input": 262144, "sliding_window_pattern": 6, "safetensors_bytes": 62546177752}),
    ],
    "granite": [
        ("ibm-granite/granite-4.2-3b", {"model_type": "granite", "hidden_size": 2560, "num_hidden_layers": 40, "intermediate_size": 8192, "num_attention_heads": 40, "num_key_value_heads": 8, "vocab_size": 100352, "max_position_embeddings": 131072, "safetensors_bytes": 7319475200}),
        ("ibm-granite/granite-4.2-8b", {"model_type": "granite", "hidden_size": 4096, "num_hidden_layers": 40, "intermediate_size": 12800, "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 100352, "max_position_embeddings": 131072, "safetensors_bytes": 17583185920}),
        ("ibm-granite/granite-4.2-30b", {"model_type": "granite", "hidden_size": 4096, "num_hidden_layers": 64, "intermediate_size": 32768, "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 100352, "max_position_embeddings": 131072, "safetensors_bytes": 58553540608}),
    ],
    "lfm": [
        ("LiquidAI/LFM2.5-350M", {"model_type": "lfm2", "hidden_size": 1024, "num_hidden_layers": 16, "intermediate_size": 6656, "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 65536, "tie_embedding": True, "max_position_embeddings": 128000, "block_auto_adjust_ff_dim": True, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256, "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "full_attention", "conv", "full_attention", "conv", "full_attention", "conv"], "safetensors_bytes": 708967936}),
        ("LiquidAI/LFM2.5-1.2B-Instruct", {"model_type": "lfm2", "hidden_size": 2048, "num_hidden_layers": 16, "intermediate_size": 12288, "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 65536, "tie_embedding": True, "max_position_embeddings": 128000, "block_auto_adjust_ff_dim": True, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256, "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "full_attention", "conv", "full_attention", "conv", "full_attention", "conv"], "safetensors_bytes": 2340681216}),
        ("LiquidAI/LFM2.5-2.6B", {"model_type": "lfm2", "hidden_size": 2048, "num_hidden_layers": 30, "intermediate_size": 10752, "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128000, "tie_word_embeddings": True, "max_position_embeddings": 131072, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256, "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv"], "safetensors_bytes": 5394397184}),
        ("LiquidAI/LFM2.5-8B-A1B", {"model_type": "lfm2_moe", "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 7168, "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128000, "tie_word_embeddings": True, "max_position_embeddings": 128000, "num_experts": 32, "moe_intermediate_size": 1792, "num_dense_layers": 2, "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv"], "safetensors_bytes": 16935715072}),
    ],
    "nemotron": [
        ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", {"model_type": "nemotron_h", "hidden_size": 3136, "num_hidden_layers": 42, "intermediate_size": 12544, "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 131072, "max_position_embeddings": 262144, "hybrid_override_pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-", "mamba_num_heads": 96, "mamba_head_dim": 80, "n_groups": 8, "ssm_state_size": 128, "mlp_hidden_act": "relu2", "safetensors_bytes": 7947113664}),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", {"model_type": "nemotron_h", "hidden_size": 2688, "num_hidden_layers": 52, "intermediate_size": 1856, "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 131072, "max_position_embeddings": 262144, "moe_intermediate_size": 1856, "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "mlp_hidden_act": "relu2", "n_routed_experts": 128, "moe_shared_expert_intermediate_size": 3712, "n_shared_experts": 1, "layers_block_type": ["mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe"], "safetensors_bytes": 65826538624}),
    ],
    "minicpm": [
        ("openbmb/MiniCPM5-1B", {"model_type": "llama", "hidden_size": 1536, "num_hidden_layers": 24, "intermediate_size": 4608, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 130560, "max_position_embeddings": 131072, "safetensors_bytes": 2161265664}),
        ("openbmb/MiniCPM5-2B", {"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 42, "intermediate_size": 6144, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 130560, "max_position_embeddings": 131072, "safetensors_bytes": 5033512960}),
    ],
    "llama": [("meta-llama/Llama-3.2-1B-Instruct", _cfg("llama", 2048, 16, 8192, 32, 8, 64, 128256, True, 131072)), ("meta-llama/Llama-3.2-3B-Instruct", _cfg("llama", 3072, 28, 8192, 24, 8, 128, 128256, True, 131072)),
              ("meta-llama/Llama-3.1-8B-Instruct", _cfg("llama", 4096, 32, 14336, 32, 8, 128, 128256, False, 131072))],
    "mistral": [("mistralai/Mistral-7B-Instruct-v0.3", _cfg("mistral", 4096, 32, 14336, 32, 8, 128, 32768, False, 32768))],
}
COMMON = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"]  # what `glyd doctor` says fits


def ladder_model(repo):
    for rungs in LADDERS.values():
        for r, cfg in rungs:
            if r == repo:
                return model_of(r, cfg, bf16=cfg.get("safetensors_bytes"))


def family(m):
    """The ladder of a model's family (LADDERS), by its model_type and name; Qwen3.5's where none is its own."""
    kinds, name = {m.config.get("model_type"), m.text.get("model_type")}, m.name.lower()
    if kinds & {"gemma4", "gemma4_unified", "gemma4_text", "gemma4_unified_text"}:
        return "gemma"
    if "granite" in kinds and "granite-4" in name:
        return "granite"
    if kinds & {"lfm2", "lfm2_moe"}:
        return "lfm"
    if kinds & {"nemotron_h", "NemotronH_Nano_Omni_Reasoning_V3"}:
        return "nemotron"
    if "minicpm5" in name:
        return "minicpm"
    if "llama" in kinds and "llama" in name:
        return "llama"
    return "mistral" if "mistral" in kinds else "qwen"


def suggest(m, gpu, mode="run"):
    """The largest model of the requested one's family (family()) that is smaller than it and fits this GPU: (repo, Settings), or None."""
    best = None
    for repo, cfg in LADDERS[family(m)]:
        cand = model_of(repo, cfg, bf16=cfg.get("safetensors_bytes"))
        if cand.bf16 < m.bf16:
            try:
                best = (repo, settings(cand, gpu, mode))
            except Refusal:
                pass
    return best


def memory_fixes(m, gpu, mode="run", users=()):
    """The lines that say what to do about a model that does not fit: what holds the GPU's memory, and the largest model of its family that fits."""
    fixes = []
    if users:
        fixes.append("Close what holds GPU memory now: " + ", ".join(f"{n} ({gb(b)})" for n, b in users[:3]))
    pick = suggest(m, gpu, mode)
    if pick:
        fixes.append(f"Or try {pick[0]}, which needs about {gb(pick[1].needs)}: glyd {mode} {pick[0]}")
    return fixes


def refusal_with_fix(m, gpu, refusal, mode="run", users=()):
    """A Refusal for a model that does not fit, with what to do: stop what holds the GPU's memory, or run the largest model that fits."""
    fixes = memory_fixes(m, gpu, mode, users)
    if not fixes:
        fixes.append("No model Glyd suggests fits this GPU's free memory; run glyd doctor to see what it has")
    return Refusal(refusal.what, "\n".join(fixes))


# --- disk and port -----------------------------------------------------------------------------------------------------------

def disk_free(path):
    """Free bytes on the drive a (maybe not yet existing) directory is on."""
    while path and not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    return shutil.disk_usage(path or ".").free


def check_disk(m, free=None):
    """A Refusal where the download does not fit the drive the Hub cache is on (the bytes not yet downloaded, and 1 GiB)."""
    if not m.files:
        return
    need = sum(n for _, n in m.files) - cached_bytes(m.repo, m.files)
    free = disk_free(hub_cache()) if free is None else free
    if need > 0 and need + GiB > free:
        raise Refusal(f"{m.name} needs {gb(need)} of disk to download, and {hub_cache()} has {gb(free)} free",
                      f"Free some space, or keep the models on another drive: HF_HOME=/path/on/that/drive glyd run {m.repo}")


def port_free(host, port):
    """Whether host:port can be bound now: for a local address, on 127.0.0.1 and on ::1 (a program listening on [::1]:8000, a dev server
    say, is what a browser opening http://localhost:8000 reaches first)."""
    for h in (["127.0.0.1", "::1"] if host in ("127.0.0.1", "localhost", "::1") else [host]):
        try:
            sock = socket.socket(socket.AF_INET6 if ":" in h else socket.AF_INET, socket.SOCK_STREAM)
        except OSError:  # (no IPv6 here)
            continue
        with sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((h, port))
            except OSError as e:
                if e.errno in (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT):  # (IPv6 is not set up on this machine)
                    continue
                return False
    return True


def pick_port(host, port, explicit, tries=20):
    """The port to serve on: `port` if it is free; else, where the user did not ask for that one, the next free one; else a Refusal."""
    if port_free(host, port):
        return port
    if not explicit:
        for p in range(port + 1, port + tries):
            if port_free(host, p):
                return p
    raise Refusal(f"port {port} is in use by another program", f"Choose another: --port {port + 1}")
