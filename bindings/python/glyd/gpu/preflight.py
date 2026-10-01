"""What `glyd run`, `glyd serve` and `glyd doctor` check and choose before vLLM starts: the GPU and its driver, whether a model
fits the memory that is free, and the settings to start vLLM with. Numbers in, numbers out: the standard library only (nvidia-smi
and the Hub are read through functions the tests replace), no torch and no vLLM import.

    gpu, warnings = setup_checks()                        # nvidia-smi, the driver, vLLM
    m = load_model("Qwen/Qwen3-8B")                       # the Hub's config and file sizes (or a directory)
    s = settings(m, gpu, mode="run", nvcc=have_nvcc())
    s.util, s.context, s.eager, s.tool_parser, s.reasoning_parser, s.env

A failed check is a Refusal: what is wrong in plain words, and what to do about it.

The settings, in bytes (T: the memory CUDA sees, F: free now; the constants below are measured on an L4 held to a 16 GB card's
memory, benchmarks/gpu/l4-local-chat-2026-09-30):
- util = floor_0.01((F - CTX - HEADROOM) / T), at most 0.92 (vLLM's own default). vLLM's budget is util x T; its CUDA context
  (CTX) is outside it, and HEADROOM stays free for a desktop's own growth.
- weights = Linears x bits / 16 + the rest + the decode buffer (the largest matrix, bf16) + EXTRA: the Linears the plugin packs (10.80
  bits a weight tiered, 12.04 12-bit), the embeddings, LM head and norms as they are, and the plugin's buffer and workspaces.
- context = the model's own length, or the most the KV cache holds in (util x T - weights - NON_KV), rounded down to a multiple
  of 1024 (at KV_FIT of it: block rounding and the estimate's own error); under MIN_CONTEXT the model does not fit.
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of Glyd's GPU code.
"""
import glob
import importlib.metadata
import importlib.util
import math
import os
import re
import shutil
import socket
import subprocess
import sysconfig
from dataclasses import dataclass, field
from .fit import RATIO, HubError, _hub, _local, checkpoint, kv_bytes
from .format import MANIFEST
from .vllm_entry import TESTED, tested

GiB = 2**30
CTX = 0.55 * GiB  # the CUDA context and driver's share, outside vLLM's budget (EngineCore held 14,958 MiB at a 14.06 GiB budget)
HEADROOM = 0.4 * GiB  # left free for a desktop (the owner's card ran with 0.23 GiB of it)
NON_KV = 0.65 * GiB  # vLLM's non-torch memory (0.42 GiB) and peak activation (0.2) beyond the weights, with slack
EXTRA = 0.07 * GiB  # the plugin's other workspaces on top of its packs and its decode buffer (Qwen3-8B: 11.31 GiB, 11.05 from the bits, 0.19 buffer)
UTIL_MAX = 0.92
MIN_CONTEXT = 4096
CONTEXT_WANT = 8192  # tokens a chat should have: below it a smaller layout is considered
KV_FIT = 0.97
COMPILE_SPARE = 2.5 * GiB  # memory left after the context's KV cache that a compiled `serve` wants (CUDA graphs, the compile's peak)
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


def _run(cmd):
    """A command's output, or None where it is not there or fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def probe_gpus(run=_run):
    """The NVIDIA GPUs nvidia-smi lists, or a Refusal saying why there are none. `run` (a command's output, or None) is
    replaced by the tests."""
    query = lambda fields: run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    fields = "index,name,memory.total,memory.free,memory.reserved,driver_version,compute_cap,display_active"
    out = query(fields)
    if out is None:  # (an older driver has no memory.reserved or display_active)
        fields = "index,name,memory.total,memory.free,driver_version,compute_cap"
        out = query(fields)
    if out is None:
        raise Refusal("no NVIDIA GPU answered: nvidia-smi is missing or failed",
                      "Glyd runs models on an NVIDIA GPU (Ampere or newer). Install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install, then reboot), then run: glyd doctor")
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
            gpus.append(Gpu(int(row["index"]), row["name"], total - reserved, mib("memory.free"), cc, row["driver_version"], cuda, row.get("display_active", "").lower() == "enabled"))
        except (KeyError, ValueError):
            continue
    if not gpus:
        raise Refusal("nvidia-smi listed no usable GPU", "Check that the driver is loaded (run nvidia-smi), or reboot after installing it")
    return gpus


def pick_gpu(gpus, visible=None):
    """The GPU with the most free memory (among those CUDA_VISIBLE_DEVICES lists, where it is numbers)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") if visible is None else visible
    want = [int(x) for x in visible.split(",") if x.strip().isdigit()]
    return max([g for g in gpus if g.index in want] or gpus, key=lambda g: g.free)


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
    fix = f"Update the NVIDIA driver to {want} or newer (Ubuntu: sudo ubuntu-drivers install, then reboot)" if want else "Update the NVIDIA driver (https://www.nvidia.com/drivers)"
    what = f"your NVIDIA driver {gpu.driver} runs CUDA {gpu.cuda[0]}.{gpu.cuda[1]}, and the PyTorch installed here was built for CUDA {built[0]}.{built[1]}"
    if gpu.cuda[0] < built[0]:
        raise Refusal(what, fix)
    return f"{what}. It may work through CUDA's minor-version compatibility; if it does not: {fix[0].lower() + fix[1:]}"


def gpu_library(built, environ=None):
    """Where Glyd's GPU library is for PyTorch's CUDA major (GLYD_GPU_LIB, else libglyd_gpu_cudaN.so beside glyd.gpu); None where
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
    gpu = pick_gpu(gpus or probe_gpus())
    if gpu.cc < MIN_CAPABILITY:
        raise Refusal(f"{gpu.name} is too old for Glyd (compute capability {gpu.cc[0]}.{gpu.cc[1]}); it needs an NVIDIA Ampere GPU or newer",
                      "RTX 30 series, A10, A100, L4, RTX 40 series, H100 and newer work")
    if need_vllm:
        vllm_ready()
        if not have_cc():
            raise Refusal("vLLM needs a C compiler to start (Triton builds its GPU launchers with one), and this machine has none", compiler_hint())
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
    other: int  # embeddings, LM head, norms: kept as they are
    moe: bool
    kv_token: int  # KV cache bytes a token, bf16 (0 where the config does not say)
    max_len: int  # the model's own context
    buffer: int = 0  # the plugin's decode buffer
    files: list = field(default_factory=list)  # [(path, bytes)] to download
    saved: bool = False  # a glyd save: its files as they are
    local: bool = False  # a directory, or a snapshot already downloaded

    @property
    def text(self):
        return self.config.get("text_config", self.config)


def linear_bytes(t):
    """A model's Linears and experts (bf16 bytes), the rest (embeddings and LM head), and whether it has experts, from its text
    config t: the plugin's count (vllm_plugin._linear_bytes), which takes a gated MLP (gate, up, down; a mixture of experts'
    layer its E experts' matrices, moe_intermediate_size each): at most a third over for a model without a gate. (0, 0, False)
    where the config does not say."""
    try:
        h, L = t["hidden_size"], t["num_hidden_layers"]
        E = next((t[n] for n in ("num_experts", "num_local_experts", "n_routed_experts") if t.get(n)), 0)
        I = (t.get("moe_intermediate_size") or t["intermediate_size"]) * E if E else t["intermediate_size"]
        nh = t["num_attention_heads"]
        nkv = t.get("num_key_value_heads") or nh
        hd = t.get("head_dim") or h // nh
        per = h * (nh + 2 * nkv) * hd + nh * hd * h + 3 * h * I
        other = t["vocab_size"] * h * (1 if t.get("tie_word_embeddings") else 2)
        return 2 * L * per, 2 * other, bool(E)
    except (KeyError, TypeError, ZeroDivisionError):
        return 0, 0, False


def buffer_bytes(t):
    """The plugin's decode buffer (bf16): the largest matrix it decodes, a layer's merged gate and up (2 x intermediate x hidden)."""
    try:
        E = next((t[n] for n in ("num_experts", "num_local_experts", "n_routed_experts") if t.get(n)), 0)
        return 2 * 2 * ((t.get("moe_intermediate_size") or t["intermediate_size"]) * (E or 1)) * t["hidden_size"]
    except (KeyError, TypeError):
        return 0


def model_of(repo, config, bf16=None, files=(), saved=False, local=False):
    """A Model from a config, and the checkpoint's bytes where they are known (else the config's own count)."""
    t = config.get("text_config", config)
    lin, other, moe = linear_bytes(t)
    if bf16 is None:
        bf16 = lin + other
    else:
        lin = min(lin, bf16)  # (the count is an upper bound: a model with no gate, or tied weights it does not know)
        other = bf16 - lin
    try:
        kv = kv_bytes(config, 1)
    except (KeyError, TypeError, ZeroDivisionError):
        kv = 0
    return Model(repo, repo.rstrip("/").split("/")[-1], config, bf16, lin, other, moe, kv, int(t.get("max_position_embeddings") or 0), buffer_bytes(t), list(files), saved, local)


def layout_for(cc, lin, other, total, moe=False):
    """The layout the plugin picks for this GPU (kernels.best_layout, from the compute capability rather than a device): the
    tiered layout on Ada, for a mixture of experts on an A10 too, and wherever only it fits in the memory less 2 GiB; else the
    12-bit one."""
    room = total - 2 * GiB
    tiered, twelve = lin * BITS["mma"] / 16 + other, lin * BITS["mma12"] / 16 + other
    return "mma" if twelve > room >= tiered or cc == (8, 9) or (moe and cc == (8, 6)) else "mma12"


def weights_on_gpu(m, layout):
    """Bytes of GPU memory the model's weights take with Glyd (a glyd save: its files)."""
    if m.saved:
        return m.bf16 + m.buffer + EXTRA
    if not m.lin:  # (a config that does not say: the mean measured over the models, all of it packed)
        return m.bf16 * RATIO + 0.3 * GiB
    return m.lin * BITS[layout] / 16 + m.other + m.buffer + EXTRA


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
                       f"Check the name (OWNER/NAME, as in Qwen/Qwen3-8B). For a gated model (Llama, Gemma), accept its licence at {HUB}/{repo}, then run: glyd login   (it takes a token from {HUB}/settings/tokens)")
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
    return model_of(name, config, weights if saved else bf16, [] if local_copy else download_files(files), saved, local_copy)


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


def parsers(m):
    """(tool-call parser, reasoning parser) for a model by its family: vLLM 0.30.0's names, "" for none. A model with no tool parser
    still chats; one this table does not know gets neither."""
    kind = m.text.get("model_type") or m.config.get("model_type") or ""
    name = m.name.lower()
    if "deepseek-r1-distill" in name:
        return "", "deepseek_r1"
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


def footprint(m, gpu, layouts=None):
    """(weights on the GPU with Glyd, free memory needed to start with a MIN_CONTEXT-token chat) in the layout that takes the least."""
    layouts = layouts or ["mma", "mma12"]
    w = min(weights_on_gpu(m, l) for l in layouts)
    return int(w), int(w + NON_KV + MIN_CONTEXT * m.kv_token / KV_FIT + CTX + HEADROOM + 0.01 * gpu.total)  # (+ what rounding the fraction down to 0.01 can cost)


def settings(m, gpu, mode="run", context=None, nvcc=True, cc=True, environ=None, given=None):
    """The settings for this model on this GPU, or a Refusal where it does not fit. `mode` is "run" (one user's chat) or "serve";
    `context` and `given` (flags_given) are what the user chose, each used as it is. The layout is chosen here and handed to the
    plugin (GLYD_LAYOUT), so the plugin packs what was counted: the plugin's own choice for the GPU, or the tiered layout where the
    12-bit one leaves less KV cache than a CONTEXT_WANT-token chat and the tiered one more."""
    env = os.environ if environ is None else environ
    given = given or {}
    s = Settings(given=given)
    s.tool_parser, s.reasoning_parser = parsers(m)
    kv, T = m.kv_token, gpu.total
    if "VLLM_USE_FLASHINFER_SAMPLER" not in env:  # (FlashInfer's sampler compiles with nvcc at the first request that samples, and the same tokens a
        s.env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # second come from PyTorch's: 21.1 against 21.2 on an L4 with Qwen3-8B)
        if not nvcc:
            s.notes.append("PyTorch sampler, as no CUDA compiler is installed")
    if "PYTORCH_CUDA_ALLOC_CONF" not in env and "PYTORCH_ALLOC_CONF" not in env:
        s.env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if "VLLM_NO_USAGE_STATS" not in env and "DO_NOT_TRACK" not in env:
        s.env["VLLM_NO_USAGE_STATS"] = "1"
    auto, mine = layout_for(gpu.cc, m.lin, m.other, T, m.moe), (env.get("GLYD_LAYOUT") or "").strip().lower()
    layouts = [mine] if mine in BITS else [auto] + (["mma"] if auto == "mma12" and not m.saved else [])
    weights = {l: int(weights_on_gpu(m, l)) for l in layouts}
    tokens = lambda l, util: int(max(0.0, (util * T - weights[l] - NON_KV) * KV_FIT) // kv) if kv else 0
    s.layout, s.weights = layouts[0], weights[layouts[0]]
    s.needs = footprint(m, gpu, layouts)[1]
    if any((number(given.get(k)) or 1) > 1 for k in ("tensor-parallel-size", "pipeline-parallel-size", "data-parallel-size")):
        s.notes.append("several GPUs: memory settings are yours")
        return s
    if "gpu-memory-utilization" in given:
        cap = float(number(given["gpu-memory-utilization"]) or 0)
        s.notes.append("memory use as you set it")
    elif s.needs > gpu.free:
        raise Refusal(f"{m.name} needs about {gb(s.needs)} of GPU memory with Glyd ({gb(footprint(m, gpu, layouts)[0])} of weights and room for a {MIN_CONTEXT:,}-token chat); your GPU has {gb(gpu.free)} free")
    else:
        cap = min(UTIL_MAX, int((gpu.free - CTX - HEADROOM) / T * 100) / 100)
    want = min(CONTEXT_WANT, m.max_len) if m.max_len else CONTEXT_WANT
    if len(layouts) > 1 and tokens(s.layout, cap) < want and tokens("mma", cap) > tokens(s.layout, cap):
        s.layout = "mma"
        s.notes.append("tiered layout: smaller weights, room for a longer chat")
    s.weights, s.util = weights[s.layout], round(cap, 2)
    s.kv_tokens = tokens(s.layout, cap)
    fits = s.kv_tokens // 1024 * 1024
    if "max-model-len" in given:
        s.context = int(number(given["max-model-len"]) or 0)
        s.context_why = "as you set it"
    elif context:
        if kv and context > s.kv_tokens:
            raise Refusal(f"a {context:,}-token context needs more memory than is free: at most {fits:,} tokens fit beside {m.name}", f"Use --context {fits:,} or less")
        s.context, s.context_why = context, "as you asked"
    elif kv:
        s.context = min(m.max_len, fits) if m.max_len else fits
        s.context_why = "the model's own limit" if m.max_len and m.max_len <= fits else "the most that fits"
    else:
        s.notes.append("context: vLLM's choice (the config does not say enough to size it)")
    if mode == "run" and kv and s.context and "gpu-memory-utilization" not in given:  # (one chat: two windows of KV cache, not the whole GPU)
        s.util = min(s.util, math.ceil((s.weights + NON_KV + 2 * s.context * kv / KV_FIT) / T * 100) / 100)
        s.kv_tokens = tokens(s.layout, s.util)
    s.eager = not (mode == "serve" and cc and kv and s.util * T - s.weights - NON_KV - s.context * kv >= COMPILE_SPARE)
    if "enforce-eager" in given:
        s.eager = bool(given["enforce-eager"])
    if not m.saved and "GLYD_LAYOUT" not in env and "additional-config" not in given:
        s.env["GLYD_LAYOUT"] = s.layout
    return s


def vllm_args(m, s, host, port, extra=()):
    """`vllm serve`'s arguments for these settings; a flag the user gave (in `extra`, appended last) is not added here."""
    g = s.given
    args = [m.repo]
    for flag, value in (("quantization", "glyd"), ("host", host), ("port", str(port)), ("middleware", "glyd.gpu.page.ChatPage"),
                        ("max-model-len", str(s.context) if s.context else "auto"), ("gpu-memory-utilization", f"{s.util:.2f}" if s.util else "")):
        if value and flag not in g:
            args += [f"--{flag}", value]
    if s.eager and "enforce-eager" not in g:
        args.append("--enforce-eager")
    if s.tool_parser and not {"enable-auto-tool-choice", "tool-call-parser"} & set(g):
        args += ["--enable-auto-tool-choice", "--tool-call-parser", s.tool_parser]
    if s.reasoning_parser and "reasoning-parser" not in g:
        args += ["--reasoning-parser", s.reasoning_parser]
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

def _cfg(kind, h, L, I, nh, nkv, hd, V, tie, pos):
    return {"model_type": kind, "hidden_size": h, "num_hidden_layers": L, "intermediate_size": I, "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
            "vocab_size": V, "tie_word_embeddings": tie, "max_position_embeddings": pos}


LADDERS = {  # models to suggest, smallest first, by their own config.json
    "qwen": [("Qwen/Qwen3-0.6B", _cfg("qwen3", 1024, 28, 3072, 16, 8, 128, 151936, True, 40960)), ("Qwen/Qwen3-1.7B", _cfg("qwen3", 2048, 28, 6144, 16, 8, 128, 151936, True, 40960)),
             ("Qwen/Qwen3-4B", _cfg("qwen3", 2560, 36, 9728, 32, 8, 128, 151936, True, 40960)), ("Qwen/Qwen3-8B", _cfg("qwen3", 4096, 36, 12288, 32, 8, 128, 151936, False, 40960)),
             ("Qwen/Qwen3-14B", _cfg("qwen3", 5120, 40, 17408, 40, 8, 128, 151936, False, 40960)), ("Qwen/Qwen3-32B", _cfg("qwen3", 5120, 64, 25600, 64, 8, 128, 151936, False, 40960))],
    "llama": [("meta-llama/Llama-3.2-1B-Instruct", _cfg("llama", 2048, 16, 8192, 32, 8, 64, 128256, True, 131072)), ("meta-llama/Llama-3.2-3B-Instruct", _cfg("llama", 3072, 28, 8192, 24, 8, 128, 128256, True, 131072)),
              ("meta-llama/Llama-3.1-8B-Instruct", _cfg("llama", 4096, 32, 14336, 32, 8, 128, 128256, False, 131072))],
    "mistral": [("mistralai/Mistral-7B-Instruct-v0.3", _cfg("mistral", 4096, 32, 14336, 32, 8, 128, 32768, False, 32768))],
}
COMMON = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]  # what `glyd doctor` says fits


def ladder_model(repo):
    for rungs in LADDERS.values():
        for r, cfg in rungs:
            if r == repo:
                return model_of(r, cfg)


def suggest(m, gpu, mode="run"):
    """The largest model of the requested one's family (else Qwen3) that is smaller than it and fits this GPU: (repo, Settings), or None."""
    kind = m.text.get("model_type") or ""
    best = None
    for repo, cfg in LADDERS["llama" if kind == "llama" else "mistral" if kind == "mistral" else "qwen"]:
        cand = model_of(repo, cfg)
        if cand.bf16 < m.bf16:
            try:
                best = (repo, settings(cand, gpu, mode))
            except Refusal:
                pass
    return best


def refusal_with_fix(m, gpu, refusal, mode="run", users=()):
    """A Refusal for a model that does not fit, with what to do: stop what holds the GPU's memory, or run the largest model that fits."""
    fixes = []
    if users:
        fixes.append("Close what holds GPU memory now: " + ", ".join(f"{n} ({gb(b)})" for n, b in users[:3]))
    pick = suggest(m, gpu, mode)
    if pick:
        fixes.append(f"Or try {pick[0]}, which needs about {gb(pick[1].needs)}: glyd {mode} {pick[0]}")
    elif not fixes:
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
    """Whether host:port can be bound now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
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
