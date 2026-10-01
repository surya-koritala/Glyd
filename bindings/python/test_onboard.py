"""glyd run / serve / doctor, what needs no GPU and no vLLM: nvidia-smi read (an L4, an RTX 4080 SUPER with a desktop, two GPUs, an older
driver, none); the driver rule; the settings for a 16 GB card, an L4, an 8 GB card and a 16 GB Blackwell; a refusal and the model it
suggests; the parsers; vLLM's flags and the user's own; the Hub's answers (gated, missing, FP8, offline); disk and port; vLLM's
failures in plain words (real log lines); the download's progress; the `glyd` command's dispatch and forwarding; the terminal chat
and the chat page's server side against a fake OpenAI server; scripts/install.sh with a fake uv. vLLM's parser names are checked
where vLLM is installed, else that test is skipped.

    python test_onboard.py              (or pytest test_onboard.py)
    python test_onboard.py --serve 8011   a fake server with the chat page, to try the page in a browser"""
import asyncio
import glob
import http.server
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import signal  # noqa: E402
try:
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:  # (a background job of a non-interactive shell: Python leaves it ignored, and the tests that send themselves a Ctrl-C need the default)
        signal.signal(signal.SIGINT, signal.default_int_handler)
except ValueError:  # (imported outside the main thread)
    pass
import importlib  # noqa: E402
from glyd.gpu import chat, page, preflight as pf, run  # noqa: E402
from glyd import cli  # noqa: E402

fit = importlib.import_module("glyd.gpu.fit")  # (glyd.gpu's own `fit` is the function)

GiB = 2**30
QWEN = dict(pf.LADDERS["qwen"])


def raises(f, text=""):
    try:
        f()
    except pf.Refusal as r:
        assert text in r.what + r.fix, (text, r.what, r.fix)
        return r
    raise AssertionError(f"not refused: {text}")


# --- nvidia-smi ---------------------------------------------------------------------------------------------------------------

HEADER = "| NVIDIA-SMI 595.91.07              Driver Version: 595.91.07      CUDA Version: 13.2     |\n"
L4 = "0, NVIDIA L4, 23034, 22566, 469, 595.91.07, 8.9, Disabled\n"
OWNER = "0, NVIDIA GeForce RTX 4080 SUPER, 16376, 14827, 433, 595.58.03, 8.9, Enabled\n"


def smi(rows, header=HEADER, apps=""):
    """A fake nvidia-smi: the GPU query answers `rows`, a bare call the header, the apps query `apps`."""
    def run(cmd):
        if len(cmd) == 1:
            return header
        if "--query-compute-apps" in cmd[1]:
            return apps
        return rows
    return run


def test_probe_gpus():
    g = pf.probe_gpus(smi(L4))[0]
    assert (g.name, g.cc, g.cuda, g.driver, g.display) == ("NVIDIA L4", (8, 9), (13, 2), "595.91.07", False)
    assert g.total == (23034 - 469) * 2**20 and g.free == 22566 * 2**20  # (what CUDA sees: vLLM's fraction is of this)
    o = pf.probe_gpus(smi(OWNER))[0]
    assert abs(o.total - 16_717_119_488) < 2**20 and o.display  # (the owner's card: 15.57 GiB, as its vLLM log says)
    assert abs(o.free / GiB - 14.48) < 0.01
    two = pf.probe_gpus(smi(L4 + "1, NVIDIA RTX A6000, 49140, 40000, 400, 595.91.07, 8.6, Disabled\n"))
    assert [x.index for x in two] == [0, 1] and pf.pick_gpu(two, "").index == 1 and pf.pick_gpu(two, "0").index == 0 and pf.pick_gpu(two, "GPU-abc").index == 1
    older = lambda cmd: None if len(cmd) > 1 and "memory.reserved" in cmd[1] else smi("0, NVIDIA L4, 23034, 22566, 535.183, 8.9\n")(cmd)
    g = pf.probe_gpus(older)[0]  # (an older driver: no reserved field; the driver keeps about 2%)
    assert g.cc == (8, 9) and 0.97 * 23034 * 2**20 < g.total < 23034 * 2**20
    r = raises(lambda: pf.probe_gpus(lambda cmd: None, platform="linux"), "no NVIDIA GPU answered")
    assert "nvidia-driver-580" in r.fix and "ubuntu-drivers install" not in r.fix and "work without one" in r.fix  # (the package that has the number, not whichever is recommended)
    mac = raises(lambda: pf.probe_gpus(lambda cmd: None, platform="darwin"), "needs Linux and an NVIDIA GPU")
    assert "driver" not in (mac.what + mac.fix).lower() and "glyd FILE -o OUT" in mac.fix and "Mac" in mac.what  # (S12: no driver advice for a Mac)
    win = raises(lambda: pf.probe_gpus(lambda cmd: None, platform="win32"), "needs Linux and an NVIDIA GPU")
    assert "WSL2" in win.fix
    raises(lambda: pf.probe_gpus(smi("")), "no usable GPU")
    assert pf.other_users(smi(L4, apps="123, /usr/bin/python3, 6495\n456, ollama, 2000\n")) == [("python3", 6495 * 2**20), ("ollama", 2000 * 2**20)]


def test_driver_rule():
    g = lambda cuda: pf.Gpu(0, "x", 16 * GiB, 15 * GiB, (8, 9), "550.163", cuda)
    assert pf.check_driver(g((13, 2)), (13, 0)) is None and pf.check_driver(g((13, 0)), (13, 0)) is None and pf.check_driver(g(()), (13, 0)) is None
    assert pf.check_driver(g((12, 4)), None) is None
    r = raises(lambda: pf.check_driver(g((12, 4)), (13, 0)), "CUDA 12.4")  # (an older major: refused, with the driver to install)
    assert "580" in r.fix and "13.0" in r.what
    w = pf.check_driver(g((12, 4)), (12, 8))  # (the same major: a warning)
    assert "570" in w and "minor-version" in w
    assert pf.torch_cuda() is None or isinstance(pf.torch_cuda(), tuple)


def test_torch_cuda_from_version_py():
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, "version.py"), "w") as f:
            f.write("__version__ = '2.13.0+cu130'\ndebug = False\ncuda: Optional[str] = '13.0'\nhip: Optional[str] = None\n")
        found = types.SimpleNamespace(submodule_search_locations=[d])
        real, pf.importlib.util.find_spec = pf.importlib.util.find_spec, lambda name: found
        try:
            assert pf.torch_cuda() == (13, 0)
            with open(os.path.join(d, "version.py"), "w") as f:
                f.write("cuda: Optional[str] = None\n")
            assert pf.torch_cuda() is None  # (built without CUDA)
        finally:
            pf.importlib.util.find_spec = real
    finally:
        shutil.rmtree(d)


def test_compiler_checks():
    assert pf.have_cc({}, lambda c: "/usr/bin/gcc" if c == "gcc" else None) and pf.have_cc({}, lambda c: "/usr/bin/clang" if c == "clang" else None)
    assert not pf.have_cc({}, lambda c: None)
    assert pf.have_cc({"CC": "/opt/zig/cc"}, lambda c: c) and not pf.have_cc({"CC": "/nope"}, lambda c: None)  # (Triton builds with $CC)
    assert "build-essential" in pf.compiler_hint('NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\n') and "python3-dev" in pf.compiler_hint("ID=debian\n")
    assert "dnf install gcc" in pf.compiler_hint("ID=fedora\n") and "base-devel" in pf.compiler_hint('ID=endeavouros\nID_LIKE="arch"\n')
    assert "package manager" in pf.compiler_hint("ID=plan9\n")
    assert pf.have_nvcc({"CUDA_HOME": "/nope"}, lambda c: None) in (True, False)  # (/usr/local/cuda may exist)
    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "bin"))
        open(os.path.join(d, "bin", "nvcc"), "w").close()
        assert pf.have_nvcc({"CUDA_HOME": d}, lambda c: None) and pf.have_nvcc({"CUDA_PATH": d}, lambda c: None)
        assert pf.have_nvcc({}, lambda c: "/x/nvcc") and (not pf.have_nvcc({}, lambda c: None) or os.path.exists("/usr/local/cuda/bin/nvcc"))
    finally:
        shutil.rmtree(d)


def test_ziglang_stands_in_for_a_missing_compiler():
    """No gcc or clang: setup_checks accepts ziglang (which install.sh adds), and the `cc` script gives Triton's -l:libcuda.so.1 to zig as a path."""
    saved = pf.package_version, pf.have_cc, pf.have_zig, pf.python_headers, pf.torch_cuda
    gpu = pf.Gpu(0, "NVIDIA L4", 22 * GiB, 20 * GiB, (8, 9), "595", (13, 2))
    try:
        pf.package_version, pf.python_headers, pf.torch_cuda = (lambda n: "0.30.0"), (lambda: True), (lambda: None)
        pf.have_cc, pf.have_zig = (lambda env=None: False), (lambda: False)
        r = raises(lambda: pf.setup_checks(True, gpus=[gpu]), "needs a C compiler")
        assert "build-essential" in r.fix or "package manager" in r.fix or "dnf" in r.fix or "pacman" in r.fix
        assert "installer again" in r.fix and "PyPI" in r.fix
        pf.have_zig = lambda: True
        assert pf.setup_checks(True, gpus=[gpu])[0] is gpu  # (ziglang is there: nothing refused)
    finally:
        pf.package_version, pf.have_cc, pf.have_zig, pf.python_headers, pf.torch_cuda = saved
    s = pf.settings(M8, L4_GPU, "run", environ={}, cc="/state/glyd/zigcc")
    assert s.env["CC"] == "/state/glyd/zigcc" and "ziglang as the C compiler (no gcc)" in pf.summary(s, L4_GPU)
    assert "CC" not in pf.settings(M8, L4_GPU, "run", environ={}).env and "CC" not in pf.settings(M8, L4_GPU, "run", environ={"CC": "gcc-13"}, cc="/x").env  # (the user's own $CC wins)
    d, fake = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:  # (a ziglang that prints what it was given, and the script run against it)
        os.makedirs(os.path.join(fake, "ziglang"))
        open(os.path.join(fake, "ziglang", "__init__.py"), "w").close()
        open(os.path.join(fake, "ziglang", "__main__.py"), "w").write("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        libs = os.path.join(d, "lib")
        os.makedirs(libs)
        open(os.path.join(libs, "libcuda.so.1"), "w").close()
        os.environ["XDG_STATE_HOME"] = d
        try:
            path = run.zig_cc()
        finally:
            del os.environ["XDG_STATE_HOME"]
        assert path == os.path.join(d, "glyd", "zigcc") and os.access(path, os.X_OK)
        out = subprocess.run([path, "k.c", "-O3", "-shared", "-L/nowhere", f"-L{libs}", "-l:libcuda.so.1", "-l:libmissing.so", "-lm", "-o", "k.so"],
                             capture_output=True, text=True, env={"PYTHONPATH": fake, "PATH": os.environ.get("PATH", "")})
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout) == ["cc", "-w", "k.c", "-O3", "-shared", "-L/nowhere", f"-L{libs}", os.path.join(libs, "libcuda.so.1"), "-l:libmissing.so", "-lm", "-o", "k.so"], out.stdout
    finally:
        shutil.rmtree(d)
        shutil.rmtree(fake)


# --- the settings -------------------------------------------------------------------------------------------------------------

def model(name, bf16):
    return pf.model_of(f"Qwen/{name}", QWEN[f"Qwen/{name}"], bf16=bf16)


M8, M4, M14, M32 = model("Qwen3-8B", 16_381_470_720), model("Qwen3-4B", 8_044_000_000), model("Qwen3-14B", 29_540_000_000), model("Qwen3-32B", 65_500_000_000)
OWNER_GPU = pf.probe_gpus(smi(OWNER))[0]
L4_GPU = pf.probe_gpus(smi(L4))[0]


def test_weights_match_what_vllm_measured():
    # Qwen3-8B, tiered, on the L4 with PyTorch's default allocator and the plugin of 0.26.0 (the load's allocator warnings gone): "Model loading took 11.39 GiB"
    assert abs(pf.weights_on_gpu(M8, "mma") / GiB - 11.39) < 0.01
    assert M8.kv_token == 147456 and M8.max_len == 40960 and not M8.moe  # (36 layers x 8 heads x 128 x 2 x 2 bytes)
    assert pf.layout_for((8, 9), M8.lin, M8.other, L4_GPU.total) == "mma" and pf.layout_for((8, 0), M8.lin, M8.other, 40 * 10**9) == "mma12"
    assert pf.layout_for((8, 6), M8.lin, M8.other, 23 * GiB, moe=True) == "mma" and pf.layout_for((8, 6), M8.lin, M8.other, 23 * GiB) == "mma12"
    assert pf.layout_for((8, 0), M8.lin, M8.other, 14 * GiB) == "mma"  # (only the tiered layout fits)


def test_settings_owner_card():
    """An RTX 4080 SUPER (15.57 GiB CUDA sees, 14.48 free with a desktop), Qwen3-8B: vLLM's budget is util x 15.57 GiB, not x the
    16,376 MiB of nvidia-smi."""
    env = {}
    s = pf.settings(M8, OWNER_GPU, "run", environ=env, nvcc=False)
    assert s.util == 0.86 and s.context % 1024 == 0 and s.context >= 8192, (s.util, s.context)
    budget = s.util * OWNER_GPU.total
    assert budget + pf.CTX + pf.HEADROOM <= OWNER_GPU.free  # (the desktop's room is kept)
    assert s.kv_tokens * M8.kv_token + s.weights + pf.NON_KV <= budget  # (and the context's KV cache fits the budget)
    assert s.eager and (s.tool_parser, s.reasoning_parser) == ("hermes", "qwen3") and s.layout == "mma"
    assert s.env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and s.env["GLYD_LAYOUT"] == "mma" and not {"PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"} & set(s.env)  # (the default allocator: expandable segments made the load's warnings worse)
    assert s.env["VLLM_NO_USAGE_STATS"] == "1"
    t = pf.settings(M8, OWNER_GPU, "run", environ={"VLLM_USE_FLASHINFER_SAMPLER": "1", "PYTORCH_ALLOC_CONF": "x", "DO_NOT_TRACK": "1"}, nvcc=False)
    assert not {"VLLM_USE_FLASHINFER_SAMPLER", "VLLM_NO_USAGE_STATS"} & set(t.env)  # (the user's own are left alone)
    withnvcc = pf.settings(M8, OWNER_GPU, "run", environ=env, nvcc=True)
    assert withnvcc.env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and "PyTorch sampler" not in pf.summary(withnvcc, OWNER_GPU)  # (off with nvcc too: no compile at the first request)
    line = pf.summary(s, OWNER_GPU)
    assert "PyTorch sampler" in line and "eager mode" in line and "(the most that fits)" in line and "86% of GPU memory" in line and line.endswith(".") and "\n" not in line
    a = pf.settings(M8, OWNER_GPU, "run", context=4096, environ=env)  # (--context)
    assert a.context == 4096
    raises(lambda: pf.settings(M8, OWNER_GPU, "run", context=30000, environ=env), "--context")
    g = pf.settings(M8, OWNER_GPU, "run", environ=env, given=pf.flags_given(["--max-model-len", "6k", "--gpu-memory-utilization=0.8"]))
    assert g.context == 6000 and g.util == 0.8 and "(as you set it)" in pf.summary(g, OWNER_GPU) and "memory use as you set it" in pf.summary(g, OWNER_GPU)


def test_settings_l4_and_modes():
    s = pf.settings(M8, L4_GPU, "run", environ={})
    assert s.context == 40960 and s.util <= 0.92 and s.eager  # (the model's own length: nothing to squeeze on 24 GB)
    sv = pf.settings(M8, L4_GPU, "serve", environ={})
    assert sv.eager and sv.util == 0.92 and "--enforce-eager" in pf.vllm_args(M8, sv, "127.0.0.1", 8000)  # (eager for serve too: 2-3% fewer tokens a second, up in a third of the time)
    comp = pf.settings(M8, L4_GPU, "serve", environ={}, given=pf.flags_given(["--no-enforce-eager"]))  # (compiled where the user asks: it needs 2.15 GiB more than eager)
    assert not comp.eager and comp.kv_tokens < sv.kv_tokens - 8000 and "compiled mode" in pf.summary(comp, L4_GPU)
    assert "--enforce-eager" not in pf.vllm_args(M8, comp, "127.0.0.1", 8000, ["--no-enforce-eager"])
    big = pf.Gpu(0, "A100", 40 * 10**9, 39 * 10**9, (8, 0), "580", (13, 0))
    r = pf.settings(M8, big, "run", environ={})
    assert r.layout == "mma12" and r.util < 0.8  # (one chat: two windows of KV cache, not the whole GPU)
    assert pf.settings(M8, big, "serve", environ={}).util >= 0.9
    t = pf.settings(M8, L4_GPU, "run", environ={}, given=pf.flags_given(["--tensor-parallel-size", "2"]))
    assert t.util == 0 and t.context == 0 and "several GPUs" in pf.summary(t, L4_GPU) and "context chosen by vLLM" in pf.summary(t, L4_GPU)  # (memory settings are the user's)


def test_a_shorter_context_needs_less_memory():
    """A context the user asks for (--context, or --max-model-len after --) under MIN_CONTEXT is the chat the memory check counts."""
    gpu = pf.Gpu(0, "x", 8 * 10**9, 0, (8, 6), "580", (13, 0))
    gpu.free = pf.footprint(M4, gpu)[1] - 2 * 10**8  # (a card 0.2 GB short of a 4,096-token chat)
    raises(lambda: pf.settings(M4, gpu, "run", environ={}), "room for a 4,096-token chat")
    s = pf.settings(M4, gpu, "run", context=1024, environ={})
    assert s.context == 1024 and s.needs <= gpu.free
    g = pf.settings(M4, gpu, "run", environ={}, given=pf.flags_given(["--max-model-len", "2K"]))  # (and so is a flag after --)
    assert g.context == 2048 and g.needs <= gpu.free
    raises(lambda: pf.settings(M4, gpu, "run", context=2 * 10**5, environ={}), "needs about")  # (a longer one is the 4,096-token check: refused)
    raises(lambda: pf.settings(M4, gpu, "run", environ={}, given=pf.flags_given(["--max-model-len", "auto"])), "needs about")


def test_settings_small_card_and_suggestion():
    small = pf.Gpu(0, "NVIDIA GeForce RTX 3070", 7_700_000_000, 7_300_000_000, (8, 6), "580", (13, 0))
    r = raises(lambda: pf.settings(M8, small, "run", environ={}), "Qwen3-8B needs about")
    assert "your GPU has 7.3 GB free" in r.what
    fixed = pf.refusal_with_fix(M8, small, r, "run", [("ollama", 6 * 10**9)])
    assert "ollama (6.0 GB)" in fixed.fix and "glyd run Qwen/Qwen3-" in fixed.fix and fixed.what == r.what
    pick = pf.suggest(M8, small)
    assert pick and pick[0] in ("Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B") and pick[1].needs <= small.free
    eight = pf.Gpu(0, "x", 8 * 10**9, 7_800_000_000, (8, 6), "580", (13, 0))
    assert pf.suggest(M8, eight)[0] in ("Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B")
    assert pf.suggest(M4, pf.Gpu(0, "tiny", 2 * 10**9, 10**9, (8, 6), "580", (13, 0))) is None
    assert pf.suggest(M32, L4_GPU)[0] == "Qwen/Qwen3-8B" and "Llama" in pf.suggest(pf.model_of("m", pf.LADDERS["llama"][2][1]), eight)[0] + "Llama"
    f = pf.refusal_with_fix(M4, pf.Gpu(0, "tiny", 2 * 10**9, 10**9, (8, 6), "580", (13, 0)), r, "run")
    assert "No model Glyd suggests" in f.fix


def test_settings_blackwell_16gb_uses_tiered():
    """A 16 GB GeForce Blackwell (compute capability 12.0): the plugin's own choice there is the 12-bit layout, which leaves less KV cache
    than one 8,192-token chat: the tiered layout is chosen and handed to the plugin."""
    r5080 = pf.Gpu(0, "NVIDIA GeForce RTX 5080", (16303 - 450) * 2**20, (16303 - 450 - 1100) * 2**20, (12, 0), "595", (13, 2), True)
    assert pf.layout_for(r5080.cc, M8.lin, M8.other, r5080.total) == "mma12"
    s = pf.settings(M8, r5080, "run", environ={})
    assert s.layout == "mma" and s.context >= 8192 and s.env["GLYD_LAYOUT"] == "mma" and "tiered layout" in pf.summary(s, r5080)
    raises(lambda: pf.settings(M8, r5080, "run", environ={"GLYD_LAYOUT": "mma12"}), "needs about")  # (the user's own layout is counted: it does not fit)
    roomy = pf.Gpu(0, "NVIDIA GeForce RTX 5080", r5080.total, r5080.free + GiB, (12, 0), "595", (13, 2), True)
    s12 = pf.settings(M8, roomy, "run", environ={"GLYD_LAYOUT": "mma12"})
    assert s12.layout == "mma12" and "GLYD_LAYOUT" not in s12.env
    big = pf.Gpu(0, "NVIDIA GeForce RTX 5090", 31 * GiB, 29 * GiB, (12, 0), "595", (13, 2), True)
    assert pf.settings(M8, big, "run", environ={}).layout == "mma12"  # (room: the plugin's own choice)
    saved = pf.model_of("qwen3-8b-glyd", QWEN["Qwen/Qwen3-8B"], bf16=11_400_000_000, saved=True, local=True)
    assert "GLYD_LAYOUT" not in pf.settings(saved, r5080, "run", environ={}).env  # (a save loads in its own layout)


def test_unknown_architecture_leaves_the_context_to_vllm():
    """A config that does not say enough to size a KV cache: the weights counted as the measured mean, the context left to vLLM (--max-model-len auto)."""
    m = pf.model_of("some/Odd-3B", {"model_type": "odd"}, bf16=6_000_000_000)
    assert m.lin == 0 and m.kv_token == 0 and m.max_len == 0
    s = pf.settings(m, L4_GPU, "run", environ={})
    assert s.context == 0 and abs(s.weights - (6_000_000_000 * 0.673 + 0.3 * GiB)) < 1e6 and "context: vLLM's choice" in pf.summary(s, L4_GPU)
    args = pf.vllm_args(m, s, "127.0.0.1", 8000)
    assert args[args.index("--max-model-len") + 1] == "auto" and "--gpu-memory-utilization" in args
    raises(lambda: pf.settings(pf.model_of("big/Odd-70B", {"model_type": "odd"}, bf16=140 * 10**9), L4_GPU, "run", environ={}), "needs about")


def test_parsers():
    p = lambda repo, kind, **kw: pf.parsers(pf.model_of(repo, {"model_type": kind, **kw}))
    assert p("Qwen/Qwen3-8B", "qwen3") == ("hermes", "qwen3") and p("Qwen/Qwen3-30B-A3B", "qwen3_moe") == ("hermes", "qwen3")
    assert p("Qwen/Qwen3-4B-Instruct-2507", "qwen3") == ("hermes", "") and p("Qwen/Qwen3-4B-Thinking-2507", "qwen3") == ("hermes", "qwen3")
    assert p("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_moe") == ("qwen3_coder", "")
    assert p("Qwen/Qwen2.5-7B-Instruct", "qwen2") == ("hermes", "") and p("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "qwen2") == ("", "deepseek_r1")
    assert p("meta-llama/Llama-3.1-8B-Instruct", "llama") == ("llama3_json", "") and p("meta-llama/Llama-3.2-3B-Instruct", "llama") == ("llama3_json", "")
    assert p("meta-llama/Llama-2-7b-chat-hf", "llama") == ("", "") and p("mistralai/Mistral-7B-Instruct-v0.3", "mistral") == ("mistral", "")
    assert p("some/Unknown-1B", "gpt2") == ("", "")
    assert p("x/y", "llama", text_config={}) == ("", "")
    # (the names are vLLM 0.30's; where it is installed, every name in the table is one of its registered parsers)
    try:
        from vllm.tool_parsers import ToolParserManager
        from vllm.reasoning import ReasoningParserManager
    except ImportError:
        assert not os.environ.get("GLYD_REQUIRE_VLLM"), "GLYD_REQUIRE_VLLM is set and vLLM is not installed: the parser names were not checked"
        return print("test_parsers: parser names not checked against vLLM (not installed)")
    tools, reasoning = set(ToolParserManager.list_registered()), set(ReasoningParserManager.list_registered())
    assert {"hermes", "llama3_json", "mistral", "qwen3_coder"} <= tools and {"qwen3", "deepseek_r1"} <= reasoning


def test_flags_and_vllm_args():
    g = pf.flags_given(["--max_model_len", "4096", "--enforce-eager", "--no-enable-prefix-caching", "--port=9000", "-tp", "2", "--served-model-name", "a"])
    assert g == {"max-model-len": "4096", "enforce-eager": True, "enable-prefix-caching": False, "port": "9000", "tensor-parallel-size": "2", "served-model-name": "a"}
    assert pf.number("8192") == 8192 and pf.number("8k") == 8000 and pf.number("8K") == 8192 and pf.number("x") is None and pf.number("0.5") == 0.5
    s = pf.settings(M8, OWNER_GPU, "run", environ={})
    args = pf.vllm_args(M8, s, "127.0.0.1", 8000)
    assert args[0] == "Qwen/Qwen3-8B" and args[args.index("--quantization") + 1] == "glyd" and args[args.index("--middleware") + 1] == "glyd.gpu.page.ChatPage"
    assert args[args.index("--max-model-len") + 1] == str(s.context) and args[args.index("--gpu-memory-utilization") + 1] == "0.86"
    assert "--enforce-eager" in args and args[args.index("--tool-call-parser") + 1] == "hermes" and "--enable-auto-tool-choice" in args
    assert args[args.index("--reasoning-parser") + 1] == "qwen3" and args[args.index("--host") + 1] == "127.0.0.1"
    given = pf.flags_given(["--max-model-len", "4096", "--no-enforce-eager", "--tool-call-parser", "x", "--host", "0.0.0.0"])  # (the user's flags win)
    s2 = pf.settings(M8, L4_GPU, "run", environ={}, given=given)  # (compiled needs 2.15 GiB more than eager: not on the 16 GB card)
    a2 = pf.vllm_args(M8, s2, "0.0.0.0", 8000, ["--max-model-len", "4096", "--no-enforce-eager", "--tool-call-parser", "x", "--host", "0.0.0.0"])
    raises(lambda: pf.settings(M8, OWNER_GPU, "run", environ={}, given=given), "needs about")
    assert a2.count("--max-model-len") == 1 and "--enforce-eager" not in a2 and a2.count("--tool-call-parser") == 1 and a2.count("--host") == 1 and "--enable-auto-tool-choice" not in a2
    plain = pf.model_of("some/Unknown-1B", {"model_type": "gpt2", **QWEN["Qwen/Qwen3-0.6B"], "model_type2": 1})
    plain.config["model_type"] = "gpt2"
    sp = pf.settings(plain, L4_GPU, "run", environ={})
    ap = pf.vllm_args(plain, sp, "127.0.0.1", 8000)
    assert "--tool-call-parser" not in ap and "--enable-auto-tool-choice" not in ap and "--reasoning-parser" not in ap  # (a family the table does not know)


# --- the Hub, the disk, the port ----------------------------------------------------------------------------------------------

def hub_answer(config, params, files):
    return lambda repo: (config, params, files)


def test_load_model_and_hub_errors():
    cfg = QWEN["Qwen/Qwen3-8B"]
    files = [("config.json", 700), ("model-00001-of-00002.safetensors", 8_000_000_000), ("model-00002-of-00002.safetensors", 8_381_470_720), ("model.safetensors.index.json", 30000),
             ("tokenizer.json", 11_000_000), ("original/consolidated.pth", 16_000_000_000), ("README.md", 9000), ("generation_config.json", 200)]
    params = {"BF16": 16_381_470_720 // 2}
    m = pf.load_model("Qwen/Qwen3-8B", hub=hub_answer(cfg, params, files))
    assert m.bf16 == 16_381_470_720 and m.name == "Qwen3-8B" and not m.saved and not m.local
    names = [p for p, _ in m.files]
    assert "config.json" in names and "tokenizer.json" in names and "model.safetensors.index.json" in names and "original/consolidated.pth" not in names and "README.md" not in names
    assert sum(n for _, n in m.files) < 16_600_000_000
    gated = lambda repo: (_ for _ in ()).throw(pf.HubError("x", 401))
    r = raises(lambda: pf.load_model("meta-llama/Llama-3.1-8B-Instruct", hub=gated), "glyd login")
    assert "gated" in r.what and "huggingface.co/meta-llama/Llama-3.1-8B-Instruct" in r.fix
    raises(lambda: pf.load_model("Qwen/Nope", hub=lambda repo: (_ for _ in ()).throw(pf.HubError("x", 404))), "OWNER/NAME")
    fp8 = {"F8_E4M3": 7_000_000_000, "BF16": 1_000_000_000}
    r = raises(lambda: pf.load_model("Qwen/Qwen3-8B-FP8", hub=hub_answer(cfg, fp8, [("model.safetensors", 9_000_000_000)])), "already quantized (FP8)")
    assert "bf16 version" in r.fix
    r = raises(lambda: pf.load_model("old/Model-F32", hub=hub_answer(cfg, {"F32": 2_000_000_000}, [("model.safetensors", 8_000_000_000)])), "32-bit floats")
    assert "bf16 version" in r.fix
    raises(lambda: pf.load_model("x/y", hub=hub_answer(cfg, {}, [("config.json", 1)])), "no safetensors")
    down = lambda repo: (_ for _ in ()).throw(OSError("Network is unreachable"))
    d = tempfile.mkdtemp()
    try:
        os.environ["HF_HUB_CACHE"] = d  # (an empty cache: this machine's own must not answer)
        raises(lambda: pf.load_model("Qwen/Qwen3-8B", hub=down), "cannot read Qwen/Qwen3-8B from the Hugging Face Hub")
        # (offline, with the model downloaded: read from the cache)
        snap = os.path.join(d, "models--Qwen--Qwen3-8B", "snapshots", "abc")
        os.makedirs(snap)
        open(os.path.join(snap, "config.json"), "w").close()
        assert pf.cached_snapshot("Qwen/Qwen3-8B") == snap and pf.cached_snapshot("Qwen/Other") is None
        m = pf.load_model("Qwen/Qwen3-8B", hub=down, local=hub_answer(cfg, params, files))
        assert m.local and not m.files and m.repo == "Qwen/Qwen3-8B"
        assert pf.load_model(snap, local=hub_answer(cfg, params, files)).local  # (a directory)
        # the bytes already there (of the size the Hub gave) are not downloaded again, and the disk check counts the rest
        for p, n in files[1:3]:
            with open(os.path.join(snap, p), "wb") as f:
                f.truncate(n)
        todo = [("model-00001-of-00002.safetensors", 8_000_000_000), ("model-00002-of-00002.safetensors", 8_381_470_720)]
        assert pf.cached_bytes("Qwen/Qwen3-8B", todo) == 16_381_470_720 and pf.cached_bytes("Qwen/Qwen3-8B", [("model-00001-of-00002.safetensors", 1)]) == 0
        ghost = pf.Model("Qwen/Qwen3-8B", "Qwen3-8B", cfg, 1, 1, 0, False, 1, 1, 0, todo)
        pf.check_disk(ghost, free=10)  # (all of it is there already: nothing to fit)
        ghost.files = [("model-00003.safetensors", 5 * 10**9)]
        raises(lambda: pf.check_disk(ghost, free=5 * 10**9), "of disk to download")
        pf.check_disk(ghost, free=7 * 10**9)
    finally:
        os.environ.pop("HF_HUB_CACHE", None)
        shutil.rmtree(d)


def test_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy = s.getsockname()[1]
    try:
        assert not pf.port_free("127.0.0.1", busy)
        assert pf.pick_port("127.0.0.1", busy, False) != busy  # (run: the next free one)
        r = raises(lambda: pf.pick_port("127.0.0.1", busy, True), f"port {busy} is in use")  # (asked for, or serve: refused)
        assert f"--port {busy + 1}" in r.fix
    finally:
        s.close()
    assert pf.port_free("127.0.0.1", busy)


def test_vllm_ready_and_setup_checks():
    real = pf.package_version
    try:
        pf.package_version = lambda n: None
        raises(lambda: pf.vllm_ready(), "vLLM is not installed")
        pf.package_version = lambda n: "0.31.0"
        raises(lambda: pf.vllm_ready(), "tested with vLLM 0.30")
        pf.package_version = lambda n: "0.30.0"
        assert pf.vllm_ready() == "0.30.0"
    finally:
        pf.package_version = real
    old = pf.Gpu(0, "Tesla T4", 15 * GiB, 14 * GiB, (7, 5), "550", (12, 4))
    real_probe = pf.probe_gpus
    try:
        r = raises(lambda: pf.setup_checks(False, gpus=[old]), "too old for Glyd")
        assert "7.5" in r.what and "Ampere" in r.what
    finally:
        pf.probe_gpus = real_probe


# --- vLLM's failures, in its own words (lines from real logs on the dev L4) -----------------------------------------------------

def test_diagnose():
    d = run.diagnose
    f = d("ValueError: To serve at least one request with the model's max seq len (4096), (0.58 GiB KV cache is needed, which is larger than the available KV cache memory (0.47 GiB). Based on the available memory, the estimated maximum model length is 3280. Try increasing")
    assert (f.kind, f.value) == ("context", 3280)
    f = d("(EngineCore pid=17635) ValueError: Free memory on device cuda:0 (15.32/22.04 GiB) on startup is less than desired GPU memory utilization (0.88, 19.39 GiB). Decrease GPU")
    assert f.kind == "free" and f.value == (15.32, 22.04) and "16.4 GB was free" in f.what
    assert d("ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`").kind == "kv"
    assert d("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 22.04 GiB").kind == "oom"
    assert d("ERROR: [Errno 98] error while attempting to bind on address ('127.0.0.1', 8000): address already in use").kind == "port"
    f = d("RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.")
    assert f.kind == "compiler" and ("build-essential" in f.fix or "package manager" in f.fix or "dnf" in f.fix or "pacman" in f.fix)
    f = d("usage: main.py [-h] [-v]\n               {chat,complete,serve,launch,bench,collect-env,run-batch} ...\nmain.py: error: unrecognized arguments: --max-model-length 4096")
    assert f.kind == "args" and "unrecognized arguments: --max-model-length 4096" in f.what and "lone --" in f.fix
    assert d("fatal error: Python.h: No such file or directory").kind == "headers"
    assert d("RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist").kind == "nvcc"
    assert d("RuntimeError: The NVIDIA driver on your system is too old (found version 12040).").kind == "driver"
    f = d("(APIServer pid=6143) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}")
    assert f.kind == "other" and "Engine core initialization failed" in f.what and "pid=" not in f.what
    assert d("").kind == "other" and "without saying why" in d("").what


# --- the download's progress ---------------------------------------------------------------------------------------------------

class Sink:
    def __init__(self, tty=False):
        self.buf, self.tty = [], tty

    def write(self, t):
        self.buf.append(t)

    def flush(self):
        pass

    def isatty(self):
        return self.tty

    def text(self):
        return "".join(self.buf)


def test_stage_of_a_log():
    d = tempfile.mkdtemp()
    try:
        log = os.path.join(d, "x.log")
        assert run.stage(os.path.join(d, "none.log")) == ""
        open(log, "w").write("$ vllm serve\n(APIServer pid=1) INFO [api_utils.py:286] non-default args: {'model_tag': 'Qwen/Qwen3-8B'}\n")
        assert run.stage(log) == ""
        open(log, "a").write("(EngineCore pid=2) INFO [model.py:692] Resolved architecture: Qwen3ForCausalLM\n")
        assert run.stage(log) == "reading the model's details"
        open(log, "a").write("(EngineCore pid=2) INFO [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config\n")
        assert run.stage(log) == "starting the engine"
        open(log, "a").write("(APIServer pid=1) WARNING [vllm.py:1547] Enforce eager set, disabling torch.compile and CUDAGraphs.\n")
        assert run.stage(log) == "starting the engine"  # (not "compiling": eager mode compiles nothing)
        open(log, "a").write("(EngineCore pid=2) Loading safetensors checkpoint shards:   0% Completed | 0/5 [00:00<?, ?it/s]\r(EngineCore pid=2) Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:13<00:20,  6.7s/it]\n")
        assert run.stage(log) == "loading the weights, 2 of 5 parts"
        open(log, "a").write("(EngineCore pid=2) INFO [model_runner.py:428] Model loading took 11.31 GiB memory and 34.1 seconds\n")
        assert run.stage(log) == "warming up"
        open(log, "a").write("(EngineCore pid=2) INFO Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 10%\n")
        assert run.stage(log) == "capturing CUDA graphs"
    finally:
        shutil.rmtree(d)


def test_progress():
    d = tempfile.mkdtemp()
    try:
        os.environ["HF_HUB_CACHE"] = d
        blobs = os.path.join(d, "models--a--b", "blobs")
        os.makedirs(blobs)
        with open(os.path.join(blobs, "done"), "wb") as f:
            f.write(b"x" * 1000)
        with open(os.path.join(blobs, "part.incomplete"), "wb") as f:  # (a download's file: sparse, its size the whole file's)
            f.truncate(10**9)
            f.seek(0)
            f.write(b"y" * 5000)
        got = run.progress_bytes("a/b")
        assert 1000 + 5000 <= got < 1000 + 10**6, got  # (the blocks written, not the size set)
        assert run.progress_bytes("a/none") == 0
    finally:
        os.environ.pop("HF_HUB_CACHE", None)
        shutil.rmtree(d)
    out = Sink()
    bar = run.Bar(run.Ui(out), "Downloading X", 1000)
    for done in (50, 120, 400, 999, 1000):
        bar.update(done)
    lines = out.text().strip().splitlines()
    assert lines == ["Downloading X: 10% (0.0 GB of 0.0 GB)", "Downloading X: 40% (0.0 GB of 0.0 GB)", "Downloading X: 90% (0.0 GB of 0.0 GB)", "Downloading X: 100% (0.0 GB of 0.0 GB)"], lines
    out = Sink(tty=True)
    bar = run.Bar(run.Ui(out), "Downloading X", 10**9)
    bar.update(5 * 10**8)
    assert "[############" in out.text() and " 50%" in out.text() and "\r" in out.text()
    assert run.fmt_time(75) == "1m15s" and run.fmt_time(9) == "9s"


def test_ui_and_args():
    out = Sink(tty=True)
    ui = run.Ui(out)
    ui.status("Loading... 3s")
    ui.line("Ready.")
    assert out.text().endswith("\r\x1b[KReady.\n")
    a = run.parse("run", ["Qwen/Qwen3-8B", "--prompt", "hi", "--context", "4096", "--no-think"])
    assert (a.model, a.prompt, a.context, a.no_think, a.port) == ("Qwen/Qwen3-8B", "hi", 4096, True, None)
    assert run.split_passthrough(["M", "--", "--max-model-len", "4096"]) == (["M"], ["--max-model-len", "4096"]) and run.split_passthrough(["M"]) == (["M"], [])
    b = run.parse("serve", ["M", "--host", "0.0.0.0", "--port", "8123"])
    assert (b.host, b.port) == ("0.0.0.0", 8123)
    real_err, sys.stderr = sys.stderr, Sink()
    try:
        run.parse("run", ["M", "--prom", "x"])
    except SystemExit:
        pass
    else:
        raise AssertionError("an abbreviated option was taken")
    finally:
        sys.stderr = real_err


def test_doctor():
    rows, gpu = run.doctor_lines(environ={}, run=smi(L4))
    labels = {r[1]: r for r in rows}
    assert labels["GPU"][0] == "ok" and "NVIDIA L4" in labels["GPU"][2] and "Driver" in labels and "C compiler" in labels and "Disk" in labels
    assert gpu.name == "NVIDIA L4"
    busy = "0, NVIDIA L4, 23034, 2000, 469, 595.91.07, 8.9, Disabled\n"  # (another program holds the GPU: said, and what fits an idle one)
    rows, gpu = run.doctor_lines(environ={}, run=smi(busy, apps="9, VLLM::EngineCore, 20000\n"))
    assert any(r[0] == "warn" and "GPU in use" == r[1] and "VLLM::EngineCore" in r[2] for r in rows)
    rows, gpu = run.doctor_lines(environ={}, run=smi(busy))  # (nothing named: the memory held is still said)
    assert any(r[0] == "warn" and "GPU in use" == r[1] and "held by other programs" in r[2] for r in rows)
    out, real, real_rows = Sink(), sys.stdout, run.doctor_lines
    sys.stdout = out
    try:
        with Patched(run__doctor_lines=lambda: (lambda r, g: ([x for x in r if x[0] != "fail"], g))(*real_rows(environ={}, run=smi(busy)))):  # (this machine's own vLLM and compiler left out)
            assert run.cmd_doctor([]) == 0
    finally:
        sys.stdout = real
    assert "With the GPU to itself: glyd run Qwen/Qwen3-8B" in out.text() and "Only 2.1 GB of the GPU's 23.7 GB is free now" in out.text() and "Ready:" not in out.text(), out.text()
    rows, gpu = run.doctor_lines(environ={}, run=lambda cmd: None, platform="linux")  # (S12: no GPU is information, not a failure)
    assert rows[-1][0] == "info" and "no NVIDIA GPU" in rows[-1][2] and "nvidia-driver-580" in rows[-1][2] and gpu is None
    rows, gpu = run.doctor_lines(environ={}, run=lambda cmd: None, platform="darwin")
    assert rows[-1][0] == "info" and "needs Linux and an NVIDIA GPU" in rows[-1][2] and "driver" not in rows[-1][2].lower() and gpu is None
    for platform, said in (("darwin", "this computer is a Mac"), ("linux", "no NVIDIA GPU answered")):  # (and the command says so, and exits 0)
        out, real = Sink(), sys.stdout
        sys.stdout = out
        try:
            with Patched(run__doctor_lines=lambda platform=platform: real_rows(environ={}, run=lambda cmd: None, platform=platform)):
                assert run.cmd_doctor([]) == 0
        finally:
            sys.stdout = real
        assert said in out.text() and "glyd run cannot run models on this computer" in out.text() and "Not ready" not in out.text(), out.text()
    rows, _ = run.doctor_lines(environ={}, run=smi("0, NVIDIA T4, 15360, 14000, 400, 550.1, 7.5, Disabled\n"))
    assert any(r[0] == "fail" and "Ampere" in r[2] for r in rows)
    mig = "0, NVIDIA A100 80GB, 81920, 80000, 400, 595.91.07, 8.0, Disabled, Enabled\n"
    rows, _ = run.doctor_lines(environ={}, run=smi(mig))  # (S11b: MIG on: CUDA sees a slice)
    assert any(r[0] == "fail" and r[1] == "GPU" and "MIG is on" in r[2] for r in rows)


# --- run.start and the commands, with the server and the GPU faked ---------------------------------------------------------------

class Proc:
    def __init__(self, code=None):
        self.code, self.pid = code, 1

    @property
    def returncode(self):
        return self.code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code


class Patched:
    """Names in run and pf replaced for a with-block, then put back."""

    def __init__(self, **names):
        self.names, self.saved = names, []

    def __enter__(self):
        for key, value in self.names.items():
            mod, attr = key.split("__")
            obj = {"run": run, "pf": pf}[mod]
            self.saved.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, value)

    def __exit__(self, *exc):
        for obj, attr, value in reversed(self.saved):
            setattr(obj, attr, value)


def test_start_retries_and_attaches():
    srv = Fake(window=4096)
    d = tempfile.mkdtemp()
    os.environ["XDG_STATE_HOME"] = d
    try:
        m = pf.model_of("fake/Model-1B", QWEN["Qwen/Qwen3-8B"], bf16=16_381_470_720, files=[("model.safetensors", 1)])
        launched, logs = [], []

        def launch(model, s, host, port, extra, log, environ=None):
            launched.append((s.context, s.util, port))
            with open(log, "w") as f:
                f.write("ValueError: To serve at least one request with the model's max seq len (9216), (1.3 GiB KV cache is needed, which is larger than the available KV cache memory (1.0 GiB). Based on the available memory, the estimated maximum model length is 7000. Try increasing\n" if len(launched) == 1 else "")
            return run.Server(srv.base, log, Proc(1) if len(launched) == 1 else Proc(None))

        ui = run.Ui(Sink())
        a = run.parse("run", ["fake/Model-1B"])
        with Patched(pf__setup_checks=lambda gpus=None: (OWNER_GPU, ["a warning"]), pf__probe_gpus=lambda: [OWNER_GPU], pf__load_model=lambda name: m, pf__check_disk=lambda m: None,
                     pf__pick_port=lambda host, port, explicit: 8123, run__download=lambda m, ui: None, run__launch=launch, run__time=types.SimpleNamespace(time=time.time, sleep=lambda s: None, strftime=time.strftime)):
            # attach: this user's own glyd serve of this model is already there (its record is the proof: another user can bind the port first and
            # answer to the same model name, and a chat sent there is theirs)
            a.port = int(srv.base.rsplit(":", 1)[1])
            run.write_marker(a.port, "fake/Model-1B", srv.base)
            got = run.start(a, [], "run", ui)
            assert got.proc is None and launched == [] and "already being served" in ui.f.text()
            run.drop_marker(a.port)
            got = run.start(a, [], "run", ui)  # (no record: a server that answers to the name is not adopted: this one starts its own)
            assert got.proc is not None and len(launched) == 2 and "already being served" not in ui.f.text().split("Ready in")[-1], launched
            launched.clear()
            a.port = None
            real_model = srv.base
            m.repo = "fake/Other"  # not the served name: a new server is started, and the first one is retried at what vLLM says fits
            got = run.start(a, [], "run", ui)
            assert len(launched) == 2 and launched[0][0] % 1024 == 0 and launched[1][0] == 6144, launched  # (7,000 rounded down to 1,024s)
            assert "starting again with 6,144" in ui.f.text() and "Note: a warning" in ui.f.text() and "Ready in" in ui.f.text() and "Settings:" in ui.f.text()
            assert "which does not fit the 15.5 GB your GPU has free" in ui.f.text()  # (bf16's 16.4 GB on the owner's card: said)
            # two GPUs: the freest one by nvidia-smi's index, which CUDA must count the same way
            second = pf.Gpu(1, "second", L4_GPU.total, L4_GPU.free, L4_GPU.cc, "595", (13, 2))
            launched.clear()
            seen = []
            real_launch = run.launch
            run.launch = lambda model, s, host, port, extra, log, environ=None: (seen.append(dict(s.env)), real_launch(model, s, host, port, extra, log, environ))[1]
            try:
                with Patched(pf__setup_checks=lambda gpus=None: (second, []), pf__probe_gpus=lambda: [OWNER_GPU, second]):
                    m.repo = "fake/Another"
                    run.start(a, [], "run", ui, environ={})
            finally:
                run.launch = real_launch
            assert seen[0]["CUDA_VISIBLE_DEVICES"] == "1" and seen[0]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID", seen[0]
            # S11c: the retry after "Free memory ... less than desired" is sized for the same GPU: it keeps the pin
            pinned = []

            def launch_free(model, s, host, port, extra, log, environ=None):
                pinned.append(dict(s.env))
                with open(log, "w") as f:
                    f.write("(EngineCore pid=1) ValueError: Free memory on device cuda:0 (15.32/22.04 GiB) on startup is less than desired GPU memory utilization (0.88, 19.39 GiB). Decrease GPU\n" if len(pinned) == 1 else "")
                return run.Server(srv.base, log, Proc(1) if len(pinned) == 1 else Proc(None))

            with Patched(pf__setup_checks=lambda gpus=None: (second, []), pf__probe_gpus=lambda: [OWNER_GPU, second], run__launch=launch_free):
                m.repo = "fake/Third"
                run.start(a, [], "run", ui, environ={})
            assert len(pinned) == 2 and all(e.get("CUDA_VISIBLE_DEVICES") == "1" and e.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" for e in pinned), pinned
            # a memory refusal comes with what to do
            tiny = pf.Gpu(0, "tiny", 6 * 10**9, 5 * 10**9, (8, 6), "580", (13, 0))
            with Patched(pf__setup_checks=lambda gpus=None: (tiny, []), pf__probe_gpus=lambda: [tiny]):
                r = raises(lambda: run.start(a, [], "run", ui), "needs about")
                assert "glyd run Qwen/Qwen3-" in r.fix
    finally:
        os.environ.pop("XDG_STATE_HOME", None)
        srv.shutdown()
        shutil.rmtree(d)


def test_commands_with_a_running_server():
    srv = Fake(window=400)
    d = tempfile.mkdtemp()
    out, err, real = Sink(), Sink(), (sys.stdout, sys.stderr)
    try:
        server = run.Server(srv.base, os.path.join(d, "x.log"))
        with Patched(run__start=lambda a, extra, mode, ui, environ=None, **k: server):
            sys.stdout, sys.stderr = out, err
            try:
                assert run.main("run", ["M", "--prompt", "hello"]) == 0
            finally:
                sys.stdout, sys.stderr = real
            assert out.text() == "Hello there, friend.\n" and "Thinking..." in err.text()
            sys.stdout, sys.stderr = Sink(), Sink()
            try:
                assert run.main("serve", ["M"]) == 0  # (attached: nothing of its own to wait for)
                try:
                    run.main("run", ["M", "--prompt", "x", "--bogus"])
                    raise AssertionError("an option glyd run does not have was taken")
                except SystemExit as e:  # (argparse: usage, status 2)
                    assert e.code == 2
            finally:
                sys.stdout, sys.stderr = real
            e = Sink()
            sys.stderr = e
            try:
                with Patched(run__start=lambda *a, **k: (_ for _ in ()).throw(pf.Refusal("no NVIDIA GPU answered", "Install the driver"))):
                    assert run.main("run", ["M", "--prompt", "x"]) == 1
            finally:
                sys.stderr = real[1]
            assert "glyd: no NVIDIA GPU answered." in e.text() and "  Install the driver" in e.text()
    finally:
        srv.shutdown()
        shutil.rmtree(d)


# --- glyd run against a fake vLLM: real processes, signals and logs --------------------------------------------------------------

FAKE_MAIN = '''
import os, signal, subprocess, sys, time
sys.path.insert(0, os.environ["FAKE_HERE"])
from test_onboard import Fake


def main():
    args = sys.argv[2:]  # serve MODEL --flags
    port = int(args[args.index("--port") + 1])
    mode = open(os.environ["FAKE_MODE"]).read().strip() if os.path.exists(os.environ["FAKE_MODE"]) else "ok"
    window = int(args[args.index("--max-model-len") + 1]) if "--max-model-len" in args and args[args.index("--max-model-len") + 1].isdigit() else 4096
    if os.environ.get("FAKE_DUMP"):  # (what this server was started with)
        import json
        json.dump({"argv": sys.argv, "env": {k: os.environ.get(k) for k in ("VLLM_API_KEY", "HF_TOKEN", "HF_HUB_OFFLINE", "GLYD_LOCAL_ONLY", "CUDA_VISIBLE_DEVICES")}}, open(os.environ["FAKE_DUMP"], "w"))
    print("(EngineCore pid=1) Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:13<00:20,  6.7s/it]", flush=True)
    if mode == "context-then-oom":  # (the first start's words, then a second start that fails of something else: the second one's cause is what is reported)
        open(os.environ["FAKE_MODE"], "w").write("oom")
        print("(EngineCore pid=1) ValueError: To serve at least one request with the model's max seq len (%d), (1.3 GiB KV cache is needed, which is larger than the available KV cache memory (1.0 GiB). Based on the available memory, the estimated maximum model length is 7000. Try increasing" % window, flush=True)
        sys.exit(1)
    if mode == "oom":
        print("(EngineCore pid=1) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 22.04 GiB", flush=True)
        sys.exit(1)
    if mode == "context-once":  # (vLLM's own words; the next start works)
        open(os.environ["FAKE_MODE"], "w").write("ok")
        print("(EngineCore pid=1) ValueError: To serve at least one request with the model's max seq len (%d), (1.3 GiB KV cache is needed, which is larger than the available KV cache memory (1.0 GiB). Based on the available memory, the estimated maximum model length is 7000. Try increasing" % window, flush=True)
        sys.exit(1)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])  # (the engine process)
    open(os.environ["FAKE_CHILD"], "w").write(str(child.pid))
    # (vLLM's API server stops its engine on SIGTERM; mode "orphan" is one that does not, and leaves it for the group's sweep)
    signal.signal(signal.SIGTERM, (lambda *a: os._exit(0)) if mode == "orphan" else (lambda *a: (child.kill(), os._exit(0))))
    if mode == "slow":
        time.sleep(60)
    print("(EngineCore pid=1) INFO [model_runner.py:428] Model loading took 11.31 GiB memory and 34.1 seconds", flush=True)
    key = args[args.index("--api-key") + 1] if "--api-key" in args else os.environ.get("VLLM_API_KEY")  # (the flag wins over the variable, as in vLLM)
    Fake(window=window, port=port, start=True, key=key or None)
    print("INFO:     Application startup complete.", flush=True)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
'''


def alive(pid):
    """Whether a process is running (a zombie is not: nothing has reaped it yet)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        pass
    try:  # (no /proc: macOS)
        return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()[:1] not in ("Z", "")
    except OSError:
        return True


class FakeVllmHome:
    """A fake `vllm.entrypoints.cli.main` on PYTHONPATH (real subprocesses, real signals and logs), a model folder and a state directory, for
    glyd run and glyd serve end to end: `go(argv, out, err, cancel_when)` runs the command with the GPU faked and Ctrl-C sent once the terminal shows a text."""
    NAMES = ("PYTHONPATH", "FAKE_HERE", "FAKE_MODE", "FAKE_CHILD", "FAKE_DUMP", "XDG_STATE_HOME", "HF_HUB_OFFLINE", "VLLM_API_KEY", "HF_TOKEN")

    def __enter__(self):
        self.d, self.state = tempfile.mkdtemp(), tempfile.mkdtemp()
        self.saved = {k: os.environ.get(k) for k in self.NAMES}
        os.makedirs(os.path.join(self.d, "vllm/entrypoints/cli"))
        for pkg in ("vllm", "vllm/entrypoints", "vllm/entrypoints/cli"):
            open(os.path.join(self.d, pkg, "__init__.py"), "w").write('__version__ = "0.30.0"\n' if pkg == "vllm" else "")
        open(os.path.join(self.d, "vllm/entrypoints/cli/main.py"), "w").write(FAKE_MAIN)
        self.model = os.path.join(self.d, "qwen3-8b")  # (a folder: a config and a safetensors file of 1,000 weights)
        os.makedirs(self.model)
        json.dump(QWEN["Qwen/Qwen3-8B"], open(os.path.join(self.model, "config.json"), "w"))
        header = json.dumps({"w": {"dtype": "BF16", "shape": [1000], "data_offsets": [0, 2000]}}).encode()
        with open(os.path.join(self.model, "model.safetensors"), "wb") as f:
            f.write(len(header).to_bytes(8, "little") + header + b"\0" * 2000)
        os.environ.update(PYTHONPATH=self.d + os.pathsep + HERE, FAKE_HERE=HERE, FAKE_MODE=os.path.join(self.d, "mode"), FAKE_CHILD=os.path.join(self.d, "child.pid"),
                          FAKE_DUMP=os.path.join(self.d, "dump.json"), XDG_STATE_HOME=self.state)
        os.environ.pop("VLLM_API_KEY", None)
        os.environ.pop("HF_HUB_OFFLINE", None)
        self.patches = dict(pf__probe_gpus=lambda: [L4_GPU], pf__setup_checks=lambda gpus=None: (L4_GPU, []), pf__have_cc=lambda *a, **k: True, pf__have_nvcc=lambda *a, **k: False)
        return self

    def __exit__(self, *exc):
        run.STOP_WAIT, run.STOP_API_WAIT = (5, 5, 10), 30
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.d)
        shutil.rmtree(self.state)

    def mode(self, text):
        open(os.environ["FAKE_MODE"], "w").write(text)

    def free_port(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def go(self, argv, out, err, cancel_when=None):
        import signal
        real = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err

        def cancel():  # (Ctrl-C, once the terminal has shown this)
            for _ in range(600):
                if cancel_when in err.text():
                    time.sleep(0.5)
                    return os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.1)

        if cancel_when:
            threading.Thread(target=cancel, daemon=True).start()
        try:
            return run.main("run" if argv[0] != "serve" else "serve", argv[1:] if argv[0] == "serve" else argv)
        finally:
            sys.stdout, sys.stderr = real


def test_serve_with_an_api_key_is_ready_and_the_key_stays_off_the_command_line():
    """B1 and S2, end to end: vLLM answers 401 on /v1 and 200 on /health when it has a key. glyd's readiness check asked /v1/models without the key, so
    `glyd serve ... -- --api-key K` never said ready and, 30 minutes on, stopped the server that worked. The key goes in VLLM_API_KEY: not on the
    server's command line (ps), not in its log."""
    with FakeVllmHome() as h:
        saved_wait = run.wait_ready.__defaults__
        run.wait_ready.__defaults__ = (8,)  # (the failure was a wait of 30 minutes)
        try:
            with Patched(**h.patches):
                out, err, port = Sink(), Sink(), h.free_port()
                code = h.go(["serve", h.model, "--port", str(port), "--", "--api-key", "SECRET123"], out, err, cancel_when="Press Ctrl-C to stop.")
        finally:
            run.wait_ready.__defaults__ = saved_wait
        assert code == 130 and "Press Ctrl-C to stop." in err.text() and "did not come up" not in err.text(), err.text()
        started = json.load(open(os.environ["FAKE_DUMP"]))
        assert "--api-key" not in started["argv"] and "SECRET123" not in " ".join(started["argv"]) and started["env"]["VLLM_API_KEY"] == "SECRET123", started
        logs = "".join(open(f).read() for f in glob.glob(os.path.join(h.state, "glyd", "logs", "*.log")))
        assert "SECRET123" not in logs and "SECRET123" not in err.text() and "taken off the server's command line" in err.text()
        assert "(asks for the API key)" in err.text() and f"OpenAI API   http://localhost:{port}/v1" in err.text()


def test_end_to_end_with_a_fake_vllm():
    import signal

    with FakeVllmHome() as h:
        d, state, model, go, free_port, patches = h.d, h.state, h.model, h.go, h.free_port, h.patches
        with Patched(**patches):
            # one answer: the server starts, answers and is gone
            port = free_port()
            out, err = Sink(), Sink()
            assert go([model, "--prompt", "hi", "--port", str(port)], out, err) == 0, err.text()
            assert out.text() == "Hello there, friend.\n", (out.text(), err.text())
            text = err.text()
            assert "Settings: " in text and "40,960-token context (the model's own limit)" in text and "Ready in" in text and "Stopping the server" in text and "PyTorch sampler" in text
            log = glob.glob(os.path.join(state, "glyd", "logs", "run-*.log"))[0]
            first = open(log).read().splitlines()[0]
            assert first.startswith("$ ") and "--quantization glyd" in first and "--middleware glyd.gpu.page.ChatPage" in first and "VLLM_USE_FLASHINFER_SAMPLER=0" in first and "--enforce-eager" in first
            assert "--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3" in first
            child = int(open(os.environ["FAKE_CHILD"]).read())
            time.sleep(0.3)
            assert not alive(child), "the engine process outlived the server's stop"
            try:
                chat.Api(f"http://127.0.0.1:{port}").get("/v1/models", timeout=1)
                raise AssertionError("the server was left running")
            except OSError:
                pass
            started = json.load(open(os.environ["FAKE_DUMP"]))  # (what the server was started with)
            assert started["env"]["HF_HUB_OFFLINE"] == "1", started  # (a folder, a cached snapshot or a download: vLLM does not wait on a network)
            assert started["env"]["GLYD_LOCAL_ONLY"] == "1" and "--allowed-origins" in started["argv"], started  # (S1: this computer's own page and programs only)
            assert json.loads(started["argv"][started["argv"].index("--allowed-origins") + 1]) == page.own_origins(port)
            assert os.stat(log).st_mode & 0o077 == 0 and os.stat(os.path.dirname(log)).st_mode & 0o077 == 0  # (the log and its folder are the user's alone)
            # vLLM says the context does not fit: one more start, at the number it gives
            open(os.environ["FAKE_MODE"], "w").write("context-once")
            out, err = Sink(), Sink()
            assert go([model, "--prompt", "hi", "--port", str(free_port()), "--context", "9000"], out, err) == 0, err.text()
            assert "starting again with 6,144" in err.text() or "room for 7,000 tokens" in err.text(), err.text()
            assert out.text() == "Hello there, friend.\n"
            # Ctrl-C while it loads: the server is stopped, the status is 130
            open(os.environ["FAKE_MODE"], "w").write("slow")
            out, err = Sink(), Sink()
            assert go([model, "--prompt", "hi", "--port", str(free_port())], out, err, cancel_when="with Glyd: ") == 130 and "Stopped." in err.text()
            assert "Stopping the server" in err.text()  # (S7a: said, not a silence of up to 30 seconds)
            time.sleep(0.5)
            assert not alive(int(open(os.environ["FAKE_CHILD"]).read()))
            # S8b: a second start that fails of something else is reported as that, not as the first start's words that are still in the log
            open(os.environ["FAKE_MODE"], "w").write("context-then-oom")
            out, err = Sink(), Sink()
            assert go([model, "--prompt", "hi", "--port", str(free_port())], out, err) == 1
            assert "ran out of memory" in err.text() and "conversation cache did not fit" not in err.text(), err.text()
            assert "Close programs that use the GPU" in err.text() or "Or try Qwen/Qwen3-" in err.text()  # (and what to do)
            # serve: up with its address, stopped by Ctrl-C
            open(os.environ["FAKE_MODE"], "w").write("ok")
            out, err = Sink(), Sink()
            port = free_port()
            assert go(["serve", model, "--port", str(port)], out, err, cancel_when="Press Ctrl-C to stop.") == 130, err.text()
            assert f"OpenAI API   http://localhost:{port}/v1" in err.text() and f"Chat page    http://localhost:{port}" in err.text() and "Press Ctrl-C to stop." in err.text()
            time.sleep(0.5)
            assert not alive(int(open(os.environ["FAKE_CHILD"]).read()))
            # a server that leaves its engine behind when it is stopped: the group's sweep (TERM, then KILL) takes it
            open(os.environ["FAKE_MODE"], "w").write("orphan")
            run.STOP_WAIT = (0.3, 0.3, 2)
            out, err = Sink(), Sink()
            assert go(["serve", model, "--port", str(free_port())], out, err, cancel_when="Press Ctrl-C to stop.") == 130, err.text()
            child = int(open(os.environ["FAKE_CHILD"]).read())
            time.sleep(0.3)
            assert not alive(child), "an engine left behind by its API server was not swept"


# --- the `glyd` command -----------------------------------------------------------------------------------------------------------

def fake_native(d):
    """A native executable named glyd in d that prints its arguments (built with cc; else a copy of echo, which macOS kills): None where
    neither runs."""
    exe = os.path.join(d, "glyd")
    src = os.path.join(d, "echo.c")
    with open(src, "w") as f:
        f.write('#include <stdio.h>\nint main(int c, char **v) { for (int i = 1; i < c; i++) printf("%s%s", v[i], i + 1 < c ? " " : "\\n"); return 0; }\n')
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not (cc and subprocess.run([cc, src, "-o", exe], capture_output=True).returncode == 0):
        shutil.copy("/bin/echo", exe)
    os.remove(src)
    os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR)
    ran = subprocess.run([exe, "x"], capture_output=True, text=True)
    return exe if ran.returncode == 0 and ran.stdout.strip() == "x" else None


def test_cli_forwarding():
    d, e, w, cwd = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        native = fake_native(d)
        if native is None:
            return print("test_cli_forwarding: skipped (no native program to forward to)")
        with open(os.path.join(e, "glyd"), "w") as f:  # (another copy of this Python tool, a console script: skipped, so that two of them never forward to each other)
            f.write("#!/usr/bin/python3\nimport sys\nfrom glyd.cli import main\nsys.exit(main())\n")
        os.chmod(os.path.join(e, "glyd"), 0o755)
        assert cli.find_native(e + os.pathsep + d, me="/nowhere") == native and cli.find_native(e, me="/nowhere") is None
        assert cli.find_native(d, me=native) is None  # (itself, by its real path)
        with open(os.path.join(w, "glyd"), "w") as f:  # (S13b: a wrapper around the real program, as Nix's wrapProgram or an asdf shim makes, is the program)
            f.write(f'#!/bin/sh\nexec {native} "$@"\n')
        os.chmod(os.path.join(w, "glyd"), 0o755)
        assert cli.find_native(w, me="/nowhere") == os.path.join(w, "glyd")
        shutil.copy(native, os.path.join(cwd, "glyd"))  # (S13b: an empty entry of PATH is not the current directory)
        here = os.getcwd()
        os.chdir(cwd)
        try:
            assert cli.find_native(os.pathsep + "/nonexistent", me="/nowhere") is None and cli.find_native(":/nonexistent:", me="/nowhere") is None
        finally:
            os.chdir(here)
        env = {"PATH": e + os.pathsep + d, "PYTHONPATH": HERE}
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "input.tar", "-o", "input.tar.glyd"], env=env, capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "input.tar -o input.tar.glyd", (r.stdout, r.stderr)  # (forwarded as it came)
        plain = subprocess.run([native, "--version"], capture_output=True, text=True).stdout.strip() == "--version"  # (GNU echo, the fallback, answers --version and --help itself)
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "--help"], env=env, capture_output=True, text=True)
        assert "glyd run MODEL" in r.stdout and r.returncode == 0 and (not plain or "--help" in r.stdout)  # (its help, then the program's)
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "--version"], env=env, capture_output=True, text=True)
        assert r.stdout.startswith("glyd 0.") and (not plain or "--version" in r.stdout)
        none = {"PATH": e, "PYTHONPATH": HERE}
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "input.tar"], env=none, capture_output=True, text=True)
        assert r.returncode == 127 and "brew install surya-koritala/glyd/glyd" in r.stderr
        r = subprocess.run([sys.executable, "-m", "glyd.cli"], env=none, capture_output=True, text=True)
        assert "glyd run" in r.stdout and "brew install" in r.stdout and r.returncode == 0
        for cmd in ("doctor", "run", "serve", "login"):
            assert cmd in cli.GPU_COMMANDS
        # S13b: no loop. The program it forwards to is started with GLYD_FORWARDED=1, and a glyd that finds it set forwards nothing
        with open(os.path.join(w, "glyd"), "w") as f:
            f.write('#!/bin/sh\necho "forwarded=$GLYD_FORWARDED $@"\n')
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "input.tar"], env={"PATH": w, "PYTHONPATH": HERE}, capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "forwarded=1 input.tar", (r.stdout, r.stderr)
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "input.tar"], env={"PATH": w, "PYTHONPATH": HERE, "GLYD_FORWARDED": "1"}, capture_output=True, text=True)
        assert r.returncode == 127 and r.stdout == "" and "another glyd forwarded" in r.stderr, (r.stdout, r.stderr)
        # a file named glyd that is no program for this machine: a message, not a traceback
        with open(os.path.join(w, "glyd"), "w") as f:
            f.write("not a program\n")
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "input.tar"], env={"PATH": w, "PYTHONPATH": HERE}, capture_output=True, text=True)
        assert r.returncode == 126 and "cannot run" in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stderr)
        # S13c: the hint is the README's own command
        readme = open(os.path.join(HERE, "..", "..", "README.md")).read()
        assert "cargo install --git https://github.com/surya-koritala/Glyd glyd glyd-store glyd-gpu" in readme and "cargo install --git https://github.com/surya-koritala/Glyd glyd glyd-store glyd-gpu" in cli.MISSING
        assert "brew install surya-koritala/glyd/glyd" in readme and "cargo install glyd\n" not in cli.MISSING
    finally:
        for x in (d, e, w, cwd):
            shutil.rmtree(x)


def test_a_glyd_ahead_on_path_is_found_and_said():
    """S13a: with the Rust glyd earlier on PATH, `glyd run` typed alone reaches it, which takes `run` for a file name. glyd doctor says so, with the way out."""
    t = tempfile.mkdtemp()
    try:
        import venv
        envdir, home, rust = (os.path.join(t, n) for n in ("env", "home", "rust"))
        venv.EnvBuilder(symlinks=True, with_pip=False).create(envdir)  # (the tool environment: its Python is run by its own path, and its glyd is beside it)
        envbin, py = os.path.join(envdir, "bin"), os.path.join(envdir, "bin", "python")
        for x in (home, rust):
            os.makedirs(x)
        open(os.path.join(envbin, "glyd"), "w").write("#!/bin/sh\n")
        os.chmod(os.path.join(envbin, "glyd"), 0o755)
        os.symlink(os.path.join(envbin, "glyd"), os.path.join(home, "glyd"))  # (what uv puts on PATH)
        open(os.path.join(rust, "glyd"), "w").write("#!/bin/sh\n")
        os.chmod(os.path.join(rust, "glyd"), 0o755)
        code = "from glyd.gpu import preflight as pf; import json, sys; print(json.dumps(pf.path_shadow()))"
        run1 = lambda path: json.loads(subprocess.run([py, "-c", code], env={"PATH": path, "PYTHONPATH": HERE}, capture_output=True, text=True, check=True).stdout)
        assert run1(rust + os.pathsep + home) == [os.path.join(rust, "glyd"), os.path.join(home, "glyd")]  # (the Rust one first: said, with ours by the name on PATH)
        assert run1(home + os.pathsep + rust) is None and run1(home) is None and run1(rust) == [os.path.join(rust, "glyd"), os.path.realpath(os.path.join(envbin, "glyd"))]  # (ours is not on PATH at all: by its real path)
        ours = os.path.join(envbin, "glyd")
        rows, _ = run.doctor_lines(environ={"PATH": rust + os.pathsep + home}, run=smi(L4), mine=ours)
        row = [r for r in rows if r[1] == "glyd on PATH"]
        assert len(row) == 1 and row[0][0] == "warn" and os.path.join(rust, "glyd") in row[0][2], rows
        assert os.path.join(home, "glyd") in row[0][2] and f'export PATH="{home}:$PATH"' in row[0][2], row[0][2]  # (the exact way out: ours by its name on PATH, and its folder first)
        rows, _ = run.doctor_lines(environ={"PATH": home + os.pathsep + rust}, run=smi(L4), mine=ours)
        assert not [r for r in rows if r[1] == "glyd on PATH"]  # (ours first: nothing to say)
    finally:
        shutil.rmtree(t)


# --- a fake OpenAI server: the terminal chat and the page's server side -------------------------------------------------------------

class Fake(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, window=40, reasoning=True, field="reasoning_content", port=0, start=True, key=None):
        super().__init__(("127.0.0.1", port), FakeHandler)
        self.window, self.reasoning, self.field, self.requests, self.key = window, reasoning, field, [], key  # (key: vLLM's --api-key / VLLM_API_KEY: /v1 asks for it, /health does not)
        if start:
            threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class FakeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def unauthorized(self):
        """vLLM's AuthenticationMiddleware: /v1 wants the key, every other path is open. True where the request was refused."""
        if self.server.key and self.path.startswith("/v1") and self.headers.get("Authorization") != "Bearer " + self.server.key:
            self.send(401, {"error": "Unauthorized"})
            return True
        return False

    def do_GET(self):
        if self.unauthorized():
            return
        if self.path == "/health":
            self.send(200, b"")
        elif self.path == "/v1/models":
            self.send(200, {"object": "list", "data": [{"id": "fake/Model-1B", "max_model_len": self.server.window}]})
        elif self.path == "/":
            self.send(200, page.PAGE, "text/html; charset=utf-8")
        else:
            self.send(404, {"detail": "Not Found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.unauthorized():
            return
        self.server.requests.append(body)
        window = self.server.window
        prompt = sum(len(m["content"].split()) + 3 for m in body["messages"])
        if prompt > window:
            return self.send(400, {"error": {"message": f"This model's maximum context length is {window} tokens. However, you requested 0 output tokens and your prompt contains at least {prompt} input tokens, for a total of at least {prompt} tokens. Please reduce the length of the input prompt or the number of requested output tokens.", "type": "BadRequestError", "param": "input_tokens", "code": 400}})
        last = body["messages"][-1]["content"]
        think = body.get("chat_template_kwargs", {}).get("enable_thinking", True) and self.server.reasoning
        pieces = ([("r", w) for w in ("Let me ", "think. ", "Two words.")] if think else []) + [("c", w) for w in ("Hello ", "there, ", "friend.")]
        finish = "stop"
        if "MARKDOWN" in last:  # (markdown, and HTML that must stay text)
            text = "# Title\nSome **bold**, *italic* and `code` with a [link](https://example.com).\n\n- one\n- two\n\n1. first\n2. second\n\n```python\nprint('<b>not bold</b>')\n```\n<img src=x onerror=alert(1)> and <script>alert(2)</script>"
            pieces = [("c", text[i:i + 7]) for i in range(0, len(text), 7)]
        if "ESCAPES" in last:  # (a model, or text it was given, that writes terminal control sequences: OSC 52 sets the clipboard, CSI moves the cursor)
            pieces = [("c", "ok \x1b]52;c;aGk=\x07 then "), ("c", "\x1b[31mred\x1b[0m\x9b2J\x00 end\ttabbed\nsecond line")]
        if "NEWLINES" in last:  # (what vLLM streams around </think>: the template's newlines, and an answer's first words as their own deltas)
            pieces = [("r", "\n"), ("r", "Hm."), ("r", "\n"), ("c", "\n\n"), ("c", "Loss"), ("c", "less.")]
        if "SLOW" in last:
            pieces = ([("r", "hmm ")] * 8 if think else []) + [("c", "slow ")] * 50
        if "LONGANSWER" in last:
            pieces = pieces + [("c", "word ")] * (window - prompt)
            finish = "length"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        event = lambda d: b"data: " + json.dumps(d).encode() + b"\n\n"
        if "EMPTYANSWER" in last:  # (an empty generation: a finish and nothing else)
            pieces = []
        if "ERROREVENT" in last:  # (vLLM 0.30 sends the failure as an event, then [DONE])
            self.wfile.write(event({"choices": [{"index": 0, "delta": {"content": "Half "}, "finish_reason": None}]}))
            self.wfile.write(event({"error": {"message": "EngineCore died: out of memory.", "type": "InternalServerError", "code": 500}}))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if "EOFSTREAM" in last:  # (the connection ends with no finish and no [DONE])
            self.wfile.write(event({"choices": [{"index": 0, "delta": {"content": "Half an answer "}, "finish_reason": None}]}))
            return
        if "BADJSON" in last:  # (cut inside a message)
            self.wfile.write(b'data: {"choices": [{"index": 0, "delta": {"content": "Hel')
            return
        used = prompt
        for kind, text in pieces:
            if used >= window and finish == "length":
                break
            used += 1
            if "SLOW" in last:
                time.sleep(0.15)
            delta = {"content": text} if kind == "c" else {self.server.field: text}
            self.wfile.write(b"data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}).encode() + b"\n\n")
        self.wfile.write(b"data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}).encode() + b"\n\n")
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": used - prompt, "total_tokens": used}}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


def new_chat(srv, key=None, **kw):
    api = chat.Api(srv.base, token=key) if key else chat.Api(srv.base)
    try:
        name, window = api.model()
    except chat.ApiError:  # (a server that wants a key the chat has none for)
        name, window = "fake/Model-1B", 0
    out, err = Sink(), Sink()
    return chat.Chat(api, name, window, out=out, err=err, **kw), out, err


def test_chat_turns():
    srv = Fake(window=60)
    try:
        c, out, err = new_chat(srv)
        assert (c.model, c.window) == ("fake/Model-1B", 60)
        assert c.turn("hi there") == "Hello there, friend."
        assert out.text().endswith("Hello there, friend.\n") and "Thinking..." in err.text() + out.text() and "Let me think. Two words." in out.text() + err.text()
        assert c.messages == [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "Hello there, friend."}]
        assert srv.requests[0]["stream"] and srv.requests[0]["chat_template_kwargs"] == {"enable_thinking": True} and srv.requests[0]["stream_options"] == {"include_usage": True}
        c.think = False
        c.turn("again")
        assert srv.requests[1]["chat_template_kwargs"] == {"enable_thinking": False} and [m["role"] for m in srv.requests[1]["messages"]] == ["user", "assistant", "user"]
    finally:
        srv.shutdown()


def test_chat_template_newlines_are_not_shown():
    srv = Fake()
    try:
        c, out, err = new_chat(srv)
        assert c.turn("NEWLINES") == "Lossless." and c.messages[-1]["content"] == "Lossless."
        assert out.text() == "Thinking...\nHm.\n\n...done thinking.\n\nLossless.\n", repr(out.text())  # (no blank line after Thinking... or before the answer)
    finally:
        srv.shutdown()


def test_terminal_text_has_no_control_sequences():
    """S5: what a model writes goes to the terminal as text: no ESC (OSC 52 sets the clipboard, CSI moves the cursor), no C1 control, no NUL, no
    bell; newlines and tabs stay."""
    srv = Fake()
    try:
        c, out, err = new_chat(srv)
        reply = c.turn("ESCAPES please")
        shown = out.text() + err.text()
        assert not any(ch in shown for ch in "\x1b\x07\x00\x9b"), repr(shown)
        assert "ok ]52;c;aGk= then" in shown and "red" in shown and "end\ttabbed\nsecond line" in shown, repr(shown)  # (the text of it stays, inert)
        assert chat.clean("a\x1b]52;c;x\x07b\tc\nd\r\x7f\x85e") == "a]52;c;xb\tc\ndе".replace("е", "e") or chat.clean("a\x1b]52;c;x\x07b\tc\nd\r\x7f\x85e") == "a]52;c;xb\tc\nde"
        assert reply and "\x1b" not in reply
        ui = run.Ui(Sink())
        ui.line("a refusal quoting a log: \x1b]0;title\x07 and \x1b[2J")
        assert "\x1b" not in ui.f.text() and "\x07" not in ui.f.text()
    finally:
        srv.shutdown()


def test_a_stream_that_does_not_end_is_said_so():
    """S6: an error event, a connection closed before the answer finished, a message cut in two, an empty answer: each says what happened,
    keeps nothing in the conversation, and a one-shot answer exits 1 (it said 'done' and 0 for the first, and nothing for the others)."""
    srv = Fake()
    try:
        for trigger, said in (("ERROREVENT", "EngineCore died: out of memory"), ("EOFSTREAM", "stopped answering in the middle of the answer"),
                              ("BADJSON", "stopped answering in the middle of the answer"), ("EMPTYANSWER", "The model sent no answer")):
            c, out, err = new_chat(srv, log="/tmp/x.log")
            assert c.once(trigger) == 1, trigger  # (a one-shot answer that did not come: status 1)
            assert said in err.text() and ".." not in err.text(), (trigger, err.text())  # (a server's message that ends in a full stop is not given a second)
            assert c.messages == [], (trigger, c.messages)  # (no question left unanswered, no empty assistant turn)
            if trigger in ("EOFSTREAM", "ERROREVENT"):
                assert "Half" in out.text() + err.text()  # (what came stays on the screen)
        c, out, err = new_chat(srv)  # (the next question works, on the same server)
        assert c.turn("hi") == "Hello there, friend."
    finally:
        srv.shutdown()


def test_the_api_key_is_sent_and_readiness_does_not_need_it():
    """B1: a server started with an API key answers 401 on /v1 and 200 on /health. glyd's readiness check used /v1/models without the key, so
    `glyd serve ... --api-key` never reported ready and was stopped at 30 minutes, working."""
    srv = Fake(key="s3cret")
    try:
        assert chat.Api(srv.base).status("/health", timeout=2) == 200  # (open, with or without a key; its body is empty, not JSON)
        try:
            chat.Api(srv.base).model()
            raise AssertionError("a request with no key was answered")
        except chat.ApiError as e:
            assert e.status == 401
        assert chat.Api(srv.base, token="s3cret").model() == ("fake/Model-1B", 40)
        server = run.Server(srv.base, "/nonexistent.log", Proc(None))
        saved = run.wait_ready.__defaults__
        run.wait_ready.__defaults__ = (4,)  # (a short wait: the failure was a 30-minute one)
        try:
            assert run.wait_ready(server, run.Ui(Sink()), "fake", "x") is None  # (ready: /health, which asks for no key)
        finally:
            run.wait_ready.__defaults__ = saved
        c, out, err = new_chat(srv, key="s3cret")
        assert c.turn("hi") == "Hello there, friend." and srv.requests
        c, out, err = new_chat(srv)  # (no key: a plain message, not a trace and not silence)
        assert c.turn("hi") == "" and "API key" in err.text() and "VLLM_API_KEY" in err.text(), err.text()
    finally:
        srv.shutdown()


def test_chat_reasoning_field_names_and_inline_think():
    for field in ("reasoning_content", "reasoning"):  # (vLLM 0.30 names it "reasoning"; older ones "reasoning_content")
        srv = Fake(field=field)
        try:
            c, out, err = new_chat(srv)
            assert c.turn("x") == "Hello there, friend." and "Let me think." in out.text() + err.text()
        finally:
            srv.shutdown()
    t = chat.Think()
    got = []
    for piece in ("<thi", "nk>\nIn", "ner</thi", "nk>\n\nAnswer ", "text"):
        got += t.feed(piece)
    got += t.feed("", final=True)
    assert "".join(x for k, x in got if k == "reasoning") == "Inner" and "".join(x for k, x in got if k == "content") == "Answer text"
    t = chat.Think()
    assert t.feed("No tags here", final=True) == [("content", "No tags here")] and chat.Think().feed("<think>only", final=True) == [("reasoning", "only")]


def test_chat_context_full():
    srv = Fake(window=30)
    try:
        c, out, err = new_chat(srv)
        long_question = " ".join(["word"] * 40)
        assert c.turn(long_question) == "" and c.messages == []  # (refused by the server: the question is taken back)
        assert "This conversation is longer than the model's window (30 tokens). Start a new chat with /clear." in err.text()
        assert "Traceback" not in err.text() + out.text()
        c, out, err = new_chat(srv)
        assert c.once(long_question) == 1 and "This prompt is longer than the model's window (30 tokens)" in err.text() and "--context" in err.text()  # (one answer: no chat to start)
        c, out, err = new_chat(srv)
        c.turn("LONGANSWER")  # (the answer itself fills the window)
        assert "reached the model's window (30 tokens)" in err.text() and "/clear" in err.text()
        c.messages, c.used = [], 0
        c.turn("a b c d e")
        c.turn("f g h")  # (a conversation that grows toward it: a note at 80%)
        assert "tokens of this conversation used" in err.text() or c.used < 0.8 * 30
    finally:
        srv.shutdown()


def test_chat_loop_and_once():
    srv = Fake(window=200)
    try:
        c, out, err = new_chat(srv)
        lines = iter(["hello", "", "/think", "/clear", "/?", '"""two', 'lines"""', "/bogus", "/bye"])
        assert c.loop(read=lambda prompt: next(lines)) == 0
        assert "Thinking off." in out.text() and "Started a new chat." in out.text() and "/clear" in out.text() and "Unknown command /bogus" in out.text()
        assert srv.requests[-1]["messages"][-1]["content"] == "two\nlines" and len(srv.requests[-1]["messages"]) == 1  # (after /clear: a chat of its own)
        c2, out2, err2 = new_chat(srv)
        assert c2.loop(read=lambda prompt: (_ for _ in ()).throw(EOFError())) == 0  # (Ctrl-D)
        c3, out3, err3 = new_chat(srv)
        assert c3.once("one shot") == 0 and out3.text() == "Hello there, friend.\n" and "Thinking..." in err3.text()  # (the answer alone on stdout)
    finally:
        srv.shutdown()
    dead = chat.Chat(chat.Api("http://127.0.0.1:9"), "m", 10, out=Sink(), err=Sink(), log="/tmp/x.log")
    assert dead.turn("x") == "" and "The server stopped answering" in dead.err.text() and "/tmp/x.log" in dead.err.text()


# --- the page ------------------------------------------------------------------------------------------------------------------

def call_asgi(app, method, path, scope_type="http", headers=()):
    sent = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request"}

    scope = {"type": scope_type, "path": path, "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}
    if scope_type == "http":
        scope["method"] = method
    asyncio.run(app(scope, receive, send))
    return sent


def test_page_middleware():
    seen = []

    async def app(scope, receive, send):
        seen.append((scope.get("method"), scope["path"]))
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    mw = page.ChatPage(app)
    sent = call_asgi(mw, "GET", "/")
    assert sent[0]["status"] == 200 and sent[1]["body"] == page.PAGE and not seen
    hdr = dict(sent[0]["headers"])
    assert hdr[b"content-type"].startswith(b"text/html") and int(hdr[b"content-length"]) == len(page.PAGE) and b"connect-src 'self'" in hdr[b"content-security-policy"]
    assert b"default-src 'none'" in hdr[b"content-security-policy"] and hdr[b"cache-control"] == b"no-store"
    assert call_asgi(mw, "HEAD", "/")[1]["body"] == b""
    for method, path in (("POST", "/"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("GET", "/health"), ("GET", "/docs")):
        assert call_asgi(mw, method, path)[0]["status"] == 404  # (handed to vLLM's app untouched)
    assert seen == [("POST", "/"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("GET", "/health"), ("GET", "/docs")]
    ws = call_asgi(mw, "GET", "/chat", scope_type="websocket")  # (a websocket: handed on, with no method in its scope, as vLLM's realtime API is)
    assert ws[0]["status"] == 404 and seen[-1] == (None, "/chat") or seen[-1][1] == "/chat"


def test_page_is_self_contained():
    html = page.PAGE.decode()
    import re
    assert "<!doctype html>" in html.lower() and "</html>" in html
    urls = re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", html) + re.findall(r"url\(([^)]+)\)", html) + re.findall(r"""@import\s+["']([^"']+)""", html)
    assert all(u.startswith(("data:", "#")) for u in urls), urls  # (no script, stylesheet, font or image from anywhere)
    assert "http://" not in re.sub(r"http://www\.w3\.org/2000/svg", "", html).replace("https?:\\/\\/", "")  # (the regex for links, and the svg namespace, are the only ones)
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "srcdoc", "DOMParser", "createContextualFragment", "setAttribute('on", 'setAttribute("on', "javascript:"):
        assert sink not in html, sink  # (model text goes in as text)
    assert "a.title = a.hostname" in html  # (a link shows where it goes)
    for text in ("/v1/models", "/v1/chat/completions", "reasoning_content", "max_model_len", "New chat", "maximum context length"):
        assert text in html, text


# --- install.sh ------------------------------------------------------------------------------------------------------------------

INSTALL = os.path.join(HERE, "..", "..", "scripts", "install.sh")
CONSTRAINTS_PY = os.path.join(HERE, "..", "..", "scripts", "install_constraints.py")
REPO_URL = "https://github.com/surya-koritala/Glyd"

FAKE_UV = r'''#!/bin/sh
echo "uv $*" >> "$HOME/calls.log"
BIN="${UV_TOOL_BIN_DIR:-$HOME/.local/bin}"
case "$1 $2" in
  "--version "*) echo "uv ${FAKE_UV_VERSION:-0.12.21} (fake)" ;;
  "tool dir") echo "$BIN" ;;
  "tool list") if [ -e "$HOME/.fake-tool-glyd" ]; then printf 'glyd v0.26.0rc3\n- glyd\n'; else echo "No tools installed" >&2; fi ;;
  "tool install")
    [ -z "${FAKE_UV_FAIL:-}" ] || { echo "error: fake uv failed" >&2; exit 2; }
    prev=
    for a in "$@"; do [ "$prev" = --constraints ] && cp "$a" "$HOME/constraints.seen"; prev=$a; done
    mkdir -p "$BIN"
    printf '#!/bin/sh\necho "glyd $*" >> "%s"\nexit 0\n' "$HOME/calls.log" > "$BIN/glyd"
    chmod +x "$BIN/glyd"
    touch "$HOME/.fake-tool-glyd" ;;
  "tool update-shell") echo "Updated configuration file: $HOME/.zshenv" ;;
esac
'''
FAKE_UV_INSTALLER = '#!/bin/sh\necho "uv-installer UV_NO_MODIFY_PATH=${UV_NO_MODIFY_PATH:-}" >> "$HOME/calls.log"\nmkdir -p "$HOME/.local/bin"\ncp "$FAKE_DIR/uv" "$HOME/.local/bin/uv"\nchmod +x "$HOME/.local/bin/uv"\n'
FAKE_CURL = r'''#!/bin/sh
url=; out=
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift ;; http*) url=$1 ;; esac; shift; done
echo "curl $url" >> "$HOME/calls.log"
[ -z "${FAKE_CURL_FAIL:-}" ] || { echo "curl: (22) The requested URL returned error: 503" >&2; exit 22; }
case "$url" in
  */install.sh) cp "$FAKE_DIR/uv-installer.sh" "$out" ;;
  *.tar.gz.sha256) cp "$FAKE_DIR/release.sha256" "$out" ;;
  *.tar.gz) cp "$FAKE_DIR/release.tar.gz" "$out" ;;
  *) echo "fake curl: unexpected $url" >&2; exit 22 ;;
esac
'''
PLATFORMS = {("Linux", "x86_64"): "linux-x86_64", ("Linux", "aarch64"): "linux-aarch64", ("Darwin", "arm64"): "macos-arm64"}


class Inst:
    """scripts/install.sh run against fakes: uv (its tool list, tool dir, tool install and update-shell), curl (uv's installer, a release's
    tarball and its checksum, or a failure), nvidia-smi, uname, and a compiler (gcc, clang, or none). PATH holds only those and the few
    programs the script uses, so this machine's own compilers, nvidia-smi and uv do not answer. After .run(): .calls (what was run, one
    line each), .home (the folder everything happened in)."""

    TOOLS = ("head", "tr", "mkdir", "cp", "chmod", "cat", "sed", "sh", "dirname", "cut", "awk", "df", "grep", "rm", "mv", "ln", "tar", "mktemp",
             "readlink", "sha256sum", "shasum", "basename", "touch")

    def __init__(self, nvidia=None, uv=True, os_name="Linux", arch="x86_64", compiler="gcc", other_glyd=False, curl_fail=False, bad_sha=False,
                 uv_version="0.12.21", df_kb=None, on_path=False, path_extra=(), uv_fail=False, no_home=False, tool_glyd=False, foreign=None, broken_program=False):
        self.home, self.fake, self.tools, self.tmpdir = (tempfile.mkdtemp() for _ in range(4))
        self.bindir = os.path.join(self.fake, "bin")
        os.makedirs(self.bindir)
        self.os_name, self.arch, self.no_home, self.uv_fail = os_name, arch, no_home, uv_fail
        self.uv_version, self.curl_fail, self.on_path, self.path_extra = uv_version, curl_fail, on_path, path_extra
        for t in self.TOOLS:
            if shutil.which(t):
                os.symlink(shutil.which(t), os.path.join(self.tools, t))

        def script(name, body, d=self.bindir):
            with open(os.path.join(d, name), "w") as f:
                f.write(body if body.startswith("#!") else "#!/bin/sh\n" + body)
            os.chmod(os.path.join(d, name), 0o755)

        script("uv", FAKE_UV, self.bindir if uv else self.fake)
        script("uv-installer.sh", FAKE_UV_INSTALLER, self.fake)
        script("curl", FAKE_CURL)
        script("uname", f'case "$1" in -s) echo "{os_name}";; -m) echo "{arch}";; esac\n')
        script("nvidia-smi", f'case "$1" in -L) echo "GPU 0: NVIDIA L4";; --query-gpu=driver_version) echo "{nvidia}";; esac\n' if nvidia else "exit 9\n")
        if compiler:
            script(compiler, "exit 0\n")
        if df_kb is not None:
            script("df", f'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"\necho "/dev/fake 99999999 1 {df_kb} 1% /"\n')
        self.other = None
        if other_glyd:
            self.other = tempfile.mkdtemp()
            script("glyd", "echo compression\n", self.other)
        if foreign is not None:
            os.makedirs(os.path.join(self.home, ".local", "bin"), exist_ok=True)
            script("glyd", foreign, os.path.join(self.home, ".local", "bin"))
        if tool_glyd:
            open(os.path.join(self.home, ".fake-tool-glyd"), "w").close()
        self.broken_program = broken_program
        self.make_release(bad_sha)

    def make_release(self, bad_sha):
        """glyd-vV-PLAT.tar.gz as release.yml makes it (glyd-vV-PLAT/ with the three programs, shell scripts here, and the licenses), its .sha256, and
        the version the fake program says."""
        import hashlib
        import io
        import tarfile
        plat = PLATFORMS.get((self.os_name, self.arch), "nowhere")
        name = f"glyd-v{self.version()}-{plat}"
        path = os.path.join(self.fake, "release.tar.gz")
        with tarfile.open(path, "w:gz") as t:
            def add(member, data, mode):
                ti = tarfile.TarInfo(f"{name}/{member}")
                ti.size, ti.mode = len(data), mode
                t.addfile(ti, io.BytesIO(data))
            for prog in ("glyd", "glyd-store", "glyd-gpu"):
                body = 'exit 4\n' if self.broken_program else f'echo "{prog} $*" >> "$HOME/calls.log"\ncase "$1" in --version) echo "glyd 0.26.0-rc.3"; echo "SIMD: AVX2";; esac\n'
                add(prog, ("#!/bin/sh\n" + body).encode(), 0o755)
            for lic in ("LICENSE", "COPYING", "LICENSE-glyd-store", "LICENSE-glyd-gpu", "README.md"):
                add(lic, b"licence\n", 0o644)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        with open(os.path.join(self.fake, "release.sha256"), "w") as f:
            f.write(("0" * 64 if bad_sha else digest) + f"  {name}.tar.gz\n")
        self.release = f"{REPO_URL}/releases/download/v{self.version()}/{name}.tar.gz"

    @staticmethod
    def version():
        import glyd
        return glyd.__version__

    def script_text(self):
        """install.sh with the pin of uv's installer set to the stand-in's, which is the file this run serves."""
        import hashlib
        text = open(INSTALL).read()
        digest = hashlib.sha256(open(os.path.join(self.fake, "uv-installer.sh"), "rb").read()).hexdigest()
        pinned = [l for l in text.splitlines() if l.startswith("UV_INSTALLER_SHA256=")]
        return text.replace(pinned[0], f"UV_INSTALLER_SHA256={digest}") if len(pinned) == 1 else text

    def run(self, env=None, text=None, stdin=False, shell="/bin/sh", cut=None):
        """Run the script (`text`, else install.sh as it is with uv's installer pinned to the stand-in: a test of the pin passes install.sh's own text), as
        `sh install.sh`, or through stdin as `curl | sh` runs it. Returns a namespace with .rc .out .err .calls."""
        text = self.script_text() if text is None else text
        text = text if cut is None else text[:cut]
        extra = os.pathsep.join([self.other] * bool(self.other) + ([os.path.join(self.home, ".local", "bin")] if self.on_path else []) + list(self.path_extra))
        path = os.pathsep.join(filter(None, [extra, self.bindir, self.tools]))
        e = {"PATH": path, "FAKE_DIR": self.fake, "TMPDIR": self.tmpdir, "FAKE_UV_VERSION": self.uv_version, "SHELL": "/bin/zsh", **({} if self.no_home else {"HOME": self.home})}
        if self.curl_fail:
            e["FAKE_CURL_FAIL"] = "1"
        if self.uv_fail:
            e["FAKE_UV_FAIL"] = "1"
        e.update(env or {})
        script = os.path.join(self.fake, "install.sh")
        with open(script, "w") as f:
            f.write(text)
        if stdin:
            r = subprocess.run([shell], input=text, env=e, capture_output=True, text=True, cwd=self.home)
        else:
            r = subprocess.run([shell, script], env=e, capture_output=True, text=True, cwd=self.home, stdin=subprocess.DEVNULL)
        log = os.path.join(self.home, "calls.log")
        calls = open(log).read().splitlines() if os.path.exists(log) else []
        self.last = types.SimpleNamespace(rc=r.returncode, out=r.stdout, err=r.stderr, calls=calls)
        return self.last

    def at(self, *parts):
        return os.path.join(self.home, *parts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for d in (self.home, self.fake, self.tools, self.tmpdir, self.other):
            if d:
                shutil.rmtree(d, ignore_errors=True)


def load_constraints_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("install_constraints", CONSTRAINTS_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def uv_install_lines(r):
    return [c for c in r.calls if c.startswith("uv tool install")]


SHELLS = [s for s in ("/bin/sh", "/bin/dash") if os.path.exists(s)]


def test_install_sh_on_a_gpu_machine():
    for sh in SHELLS:
        assert subprocess.run([sh, "-n", INSTALL]).returncode == 0, sh
    import glyd
    V = glyd.__version__  # (the release the script pins is the tree's: scripts/bump_version.py keeps them one)
    text = open(INSTALL).read()
    assert f'GLYD_VERSION="${{GLYD_VERSION:-{V}}}"' in text
    for sh in SHELLS:
        for via_stdin in (False, True):  # (as a file, and as `curl | sh` runs it: from its stdin)
            with Inst(nvidia="595.91.07") as i:
                r = i.run(shell=sh, stdin=via_stdin)
                assert r.rc == 0, (sh, via_stdin, r.out, r.err)
                (install,) = uv_install_lines(r)
                assert re.fullmatch(rf"uv tool install --managed-python --python 3\.12 --constraints \S+/constraints\.txt glyd\[vllm\]=={re.escape(V)}", install), install
                assert "--force" not in install and "--prerelease" not in install  # (uv itself refuses to replace what it did not make; a new pin is a reinstall)
                assert "uv tool dir --bin" in r.calls and r.calls[-1] == "glyd doctor" and not r.err.strip(), (r.calls, r.err)
                assert "Next: glyd run Qwen/Qwen3-8B" in r.out
                assert not glob.glob(os.path.join(i.tmpdir, "*"))  # (the temporary folder is gone)
                assert not [f for f in os.listdir(i.home) if f in (".bashrc", ".zshenv", ".zshrc", ".profile", ".bash_profile")]  # (no startup file of its own)
    with Inst(nvidia="595.91.07") as i:  # the constraints uv was given are the list in the script, which is the acceptance run's
        i.run()
        seen = open(i.at("constraints.seen")).read().splitlines()
        assert seen == load_constraints_module().listed(text) and len(seen) > 150 and "vllm==0.30.0" in seen, len(seen)
    with Inst(nvidia="595.91.07") as i:  # a pre-release is named, and only that one is taken
        assert uv_install_lines(i.run(env={"GLYD_VERSION": "0.26.0rc3"}))[0].endswith("glyd[vllm]==0.26.0rc3")
    with Inst(nvidia="595.91.07") as i:  # another build: its own wheel, under the same list of versions (the acceptance run installs a wheel)
        r = i.run(env={"GLYD_SPEC": "/tmp/glyd-0.26.0rc2-py3-none-manylinux_2_28_x86_64.whl[vllm]"})
        (install,) = uv_install_lines(r)
        assert re.fullmatch(r"uv tool install --managed-python --python 3\.12 --constraints \S+/constraints\.txt /tmp/glyd-0\.26\.0rc2-py3-none-manylinux_2_28_x86_64\.whl\[vllm\]", install), install
    with Inst(nvidia="595.91.07", uv_fail=True) as i:  # where uv fails with the list in use, the way round it is named
        r = i.run()
        assert "GLYD_CONSTRAINTS=none in front of sh resolves the packages fresh" in r.err, r.err
    with Inst(nvidia="595.91.07", uv_fail=True) as i:
        assert "GLYD_CONSTRAINTS" not in i.run(env={"GLYD_CONSTRAINTS": "none"}).err
    with Inst(nvidia="595.91.07") as i:  # (or the list left out, where a version in it has been withdrawn)
        assert "--constraints" not in uv_install_lines(i.run(env={"GLYD_CONSTRAINTS": "none"}))[0]
    with Inst(nvidia="595.91.07", arch="aarch64") as i:  # (none was tested on aarch64)
        r = i.run()
        assert r.rc == 0 and "--constraints" not in uv_install_lines(r)[0] and uv_install_lines(r)[0].endswith(f"glyd[vllm]=={V}"), r.calls
    zig = "--managed-python --python 3.12 "
    with Inst(nvidia="595.91.07", compiler=None) as i:  # no gcc or clang: a compiler from PyPI, no sudo; vLLM's Triton builds its launchers with one
        r = i.run()
        assert "--with ziglang==0.16.0 glyd[vllm]" in uv_install_lines(r)[0] and "No C compiler found" in r.out and r.rc == 0, (r.calls, r.out)
        assert "ZIGLANG=0.16.0" in text
    for compiler, extra in (("gcc", None), ("clang", None), (None, {"CC": "/opt/cc/bin/cc"})):  # (a compiler there, or $CC set: none is added)
        with Inst(nvidia="595.91.07", compiler=compiler) as i:
            r = i.run(env=extra)
            assert "ziglang" not in " ".join(r.calls) and "No C compiler" not in r.out, (compiler, r.calls)
    with Inst(nvidia="595.91.07", compiler=None) as i:  # a wheel's path that says vllm is not the extra (the stack is decided by the extra itself)
        assert "ziglang" not in " ".join(i.run(env={"GLYD_SPEC": "/tmp/vllm-build/glyd-0.26.0-py3-none-any.whl"}).calls)
    with Inst(nvidia="550.163.01") as i:  # an older driver: a warning, before the big download
        r = i.run()
        assert "older than 580" in r.err and "nvidia-driver-580" in r.err and r.calls.index(uv_install_lines(r)[0]) > 0 and r.rc == 0
    with Inst(os_name="FreeBSD") as i:
        r = i.run()
        assert r.rc == 1 and "Linux and macOS" in r.err and not r.calls


def test_install_sh_cut_short_runs_nothing():
    """A download that stops partway: the script is one function and its call is the last line, so no prefix of it runs a command (a half
    install, reported as a success, was what the first version did)."""
    text = open(INSTALL).read()
    last = text.rfind("\nmain ") + 1 or len(text)  # (the call, on the last line: `main "$@"`; ending the text at "\nmain" is the whole script run)
    cuts = sorted(c for c in set(range(0, len(text), 487)) | set(m.end() for k, m in enumerate(re.finditer("\n", text)) if k % 9 == 0) | set(range(max(0, last - 30), len(text))) if c < last + 3)
    with Inst(nvidia="595.91.07") as i:
        full = i.script_text()
        assert full.count("\nmain ") <= 1
        for cut in cuts:
            r = i.run(text=full, stdin=True, cut=cut)
            assert r.calls == [], (cut, repr(full[max(0, cut - 40):cut]), r.calls, r.out[-200:])
        assert i.run(text=full, stdin=True).calls  # (and the whole of it does run)


def test_install_sh_does_not_hide_a_failed_download_of_uv():
    with Inst(nvidia="595.91.07", uv=False, curl_fail=True) as i:  # (the first version ran `curl | sh`, which has sh's status)
        r = i.run()
        assert r.rc == 1 and "could not download uv's installer" in r.err and "astral.sh" in r.err, (r.rc, r.err)
        assert "not where its installer says" not in r.err and "open a new terminal" not in r.err.lower()
        assert not any(c.startswith("uv ") or c.startswith("uv-installer") for c in r.calls), r.calls
        assert "Run the same command again" not in r.err  # (nothing was downloaded that a second run keeps)
    with Inst(nvidia="595.91.07", uv=False) as i:  # uv's own installer, at a version, from a file whose sha256 is the one pinned, told to edit no startup file
        r = i.run()
        assert r.rc == 0, (r.out, r.err)
        assert r.calls[0] == "curl https://astral.sh/uv/0.12.21/install.sh" and "uv-installer UV_NO_MODIFY_PATH=1" in r.calls, r.calls
        assert os.path.exists(i.at(".local", "bin", "uv")) and any(c.startswith("uv tool install") for c in r.calls)
    with Inst(nvidia="595.91.07", uv=False) as i:  # a file that is not the one pinned is not run
        r = i.run(text=open(INSTALL).read())
        assert r.rc == 1 and "not the file this script was written for" in r.err, (r.rc, r.err)
        assert not any(c.startswith("uv-installer") for c in r.calls), r.calls
    assert re.search(r"^UV_VERSION=\d+\.\d+\.\d+", open(INSTALL).read(), re.M) and re.search(r"^UV_INSTALLER_SHA256=[0-9a-f]{64}\b", open(INSTALL).read(), re.M)


def test_install_sh_will_not_replace_a_program_it_did_not_make():
    with Inst(nvidia="595.91.07", foreign="echo mine\n") as i:  # a glyd of the user's own in uv's folder, which uv does not list
        mine = open(i.at(".local", "bin", "glyd")).read()
        r = i.run()
        assert r.rc == 1 and "is not Glyd's Python tool" in r.err and "does not replace a program it did not make" in r.err, (r.rc, r.err)
        assert "UV_TOOL_BIN_DIR" in r.err and "Run the same command again" not in r.err
        assert open(i.at(".local", "bin", "glyd")).read() == mine  # (not touched)
        assert not uv_install_lines(r) and not any("--force" in c for c in r.calls), r.calls
    with Inst(nvidia="595.91.07", foreign="echo old tool\n", tool_glyd=True) as i:  # Glyd's own, from an earlier run: an update
        r = i.run()
        assert r.rc == 0 and uv_install_lines(r), (r.out, r.err)
    with Inst(nvidia="595.91.07", foreign="echo mine\n") as i:  # kept, with the tool in another folder (then it is not on PATH, and that is said)
        r = i.run(env={"UV_TOOL_BIN_DIR": i.at("glyd-bin")})
        assert r.rc == 0 and uv_install_lines(r), (r.out, r.err)
        assert "update-shell" in " ".join(r.calls) and open(i.at(".local", "bin", "glyd")).read() == "#!/bin/sh\necho mine\n"


def test_install_sh_says_what_it_does_to_the_path():
    with Inst(nvidia="595.91.07") as i:  # ~/.local/bin is not on PATH: the edit is said, before it is made
        r = i.run()
        assert "uv tool update-shell" in r.calls and "edits your shell's startup file" in r.out, (r.calls, r.out)
        assert "Open a new terminal" in r.out and f'export PATH="{i.at(".local", "bin")}:$PATH"' in r.out
        assert "Updated configuration file" in r.out  # (uv's own line: which file)
    with Inst(nvidia="595.91.07", on_path=True) as i:  # it is: nothing is edited
        r = i.run()
        assert "uv tool update-shell" not in r.calls and "startup file" not in r.out and "Open a new terminal" not in r.out, (r.calls, r.out)
    with Inst(nvidia="595.91.07") as i:  # a folder whose name starts with it is not it
        i.path_extra = (i.at(".local", "bin2"),)
        assert "uv tool update-shell" in i.run().calls


def test_install_sh_a_glyd_ahead_on_path_is_said_with_the_fix():
    with Inst(nvidia="595.91.07", other_glyd=True, on_path=True) as i:  # (~/.local/bin on PATH, behind the compression program's folder)
        r = i.run()
        bin_ = i.at(".local", "bin")
        assert r.rc == 0 and "another glyd comes first on your PATH" in r.err and f"{i.other}/glyd" in r.err, r.err
        assert f'export PATH="{bin_}:$PATH"' in r.err and f"{bin_}/glyd run MODEL" in r.err and "~/.zshrc" in r.err
    with Inst(nvidia="595.91.07", on_path=True) as i:  # (first on PATH: nothing to say)
        assert "another glyd" not in i.run().err


def test_install_sh_checks_the_machine_before_the_big_download():
    with Inst(nvidia="595.91.07", df_kb=3 * 1024 * 1024) as i:  # 3 GB free
        r = i.run()
        assert r.rc == 1 and "3 GB free" in r.err and "about 8 GB" in r.err and "UV_CACHE_DIR" in r.err and not uv_install_lines(r), (r.rc, r.err)
    with Inst(nvidia="595.91.07", df_kb=3 * 1024 * 1024) as i:  # (uv's folders set by the user are the user's to size)
        assert i.run(env={"UV_CACHE_DIR": "/big/cache"}).rc == 0
    with Inst(nvidia="595.91.07", df_kb=40 * 1024 * 1024) as i:
        assert i.run().rc == 0
    with Inst(nvidia="595.91.07", no_home=True) as i:  # (HOME unset died in `set -u`: "HOME: parameter not set")
        r = i.run()
        assert r.rc == 1 and "HOME is not set" in r.err and not r.calls, (r.rc, r.err)
    with Inst(nvidia="595.91.07", uv_version="0.6.0") as i:  # a uv without --managed-python on `tool install`
        r = i.run()
        assert r.rc == 1 and "older than 0.7" in r.err and "self update" in r.err and not uv_install_lines(r), (r.rc, r.err)
    with Inst(nvidia="595.91.07", uv_version="1.2.3") as i:
        assert i.run().rc == 0
    with Inst(nvidia="595.91.07", uv_version="garbage") as i:  # (a version it cannot read: go on)
        assert i.run().rc == 0
    with Inst(nvidia="595.91.07", uv_fail=True) as i:  # the install itself fails: the one hint that is true for it
        r = i.run()
        assert r.rc != 0 and "Run the same command again: uv keeps what it downloaded" in r.err and "glyd doctor" not in r.calls, (r.rc, r.err, r.calls)


def test_install_sh_where_glyd_run_has_nothing_to_run_on():
    import glyd
    V = glyd.__version__
    for os_name, arch, plat, says in (("Linux", "x86_64", "linux-x86_64", "No NVIDIA GPU answered"), ("Linux", "aarch64", "linux-aarch64", "No NVIDIA GPU answered"),
                                      ("Darwin", "arm64", "macos-arm64", "This computer is a Mac")):
        with Inst(nvidia=None, os_name=os_name, arch=arch) as i:
            r = i.run()
            assert r.rc == 0, (os_name, r.out, r.err)
            assert says in r.out and "glyd run" in r.out and "needs Linux with an NVIDIA GPU" in r.out, r.out
            assert ("nvidia-driver-580" in r.out) == (os_name == "Linux"), r.out  # (no driver advice for a Mac)
            assert not any(c.startswith("uv ") for c in r.calls), r.calls  # (no tool environment that holds no compression program)
            base = f"{REPO_URL}/releases/download/v{V}/glyd-v{V}-{plat}.tar.gz"
            assert [c for c in r.calls if c.startswith("curl")] == [f"curl {base}", f"curl {base}.sha256"], r.calls
            for prog in ("glyd", "glyd-store", "glyd-gpu"):
                link = i.at(".local", "bin", prog)
                assert os.path.islink(link) and os.readlink(link) == i.at(".local", "share", "glyd", "cli", prog), prog
            assert os.path.exists(i.at(".local", "share", "glyd", "cli", "LICENSE")) and os.path.exists(i.at(".local", "share", "glyd", "cli", "COPYING"))
            assert "glyd 0.26.0-rc.3 is installed" in r.out and "Open a new terminal" not in r.out
            assert f'export PATH="{i.at(".local", "bin")}:$PATH"' in r.out  # (not on PATH, and no profile edit of its own: the line to add)
            assert not glob.glob(os.path.join(i.tmpdir, "*"))
            again = i.run()  # (an update: its own links are replaced)
            assert again.rc == 0, again.err
    with Inst(nvidia=None, bad_sha=True) as i:  # a download that is not the file the release lists: nothing is installed
        r = i.run()
        assert r.rc == 1 and "not the file the release lists" in r.err, (r.rc, r.err)
        assert not os.path.exists(i.at(".local", "share", "glyd")) and not os.path.exists(i.at(".local", "bin", "glyd"))
    with Inst(nvidia=None, broken_program=True) as i:  # a program that does not start here (a CPU without AVX2): said, and nothing of it left
        r = i.run()
        assert r.rc == 1 and "does not start on this machine" in r.err and "cargo install --git" in r.err, (r.rc, r.err)
        assert not os.path.exists(i.at(".local", "share", "glyd", "cli")) and not os.path.exists(i.at(".local", "share", "glyd", "cli.new")) and not os.path.exists(i.at(".local", "bin", "glyd"))
    with Inst(nvidia=None, curl_fail=True) as i:
        r = i.run()
        assert r.rc == 1 and "could not download" in r.err and "check the network" in r.err
    with Inst(nvidia=None, foreign="echo mine\n") as i:  # not replaced
        r = i.run()
        assert r.rc == 1 and "this installer did not make it" in r.err and not [c for c in r.calls if c.startswith("curl")], (r.rc, r.err, r.calls)
        assert open(i.at(".local", "bin", "glyd")).read() == "#!/bin/sh\necho mine\n"
    with Inst(nvidia=None, os_name="Darwin", arch="x86_64") as i:  # an Intel Mac has no tarball: Homebrew builds one
        r = i.run()
        assert r.rc == 1 and "brew install surya-koritala/glyd/glyd" in r.err and "cargo install --git" in r.err
    with Inst(nvidia=None, other_glyd=True) as i:  # another glyd first on PATH, here too
        i.on_path = True
        assert "another glyd comes first on your PATH" in i.run().err
    with Inst(nvidia="595.91.07", arch="ppc64le") as i:  # a GPU, and no vLLM build for the architecture
        r = i.run()
        assert r.rc == 1 and "no vLLM build for ppc64le" in r.out and "no prebuilt compression program" in r.err
    with Inst(nvidia=None) as i:  # a Linux GPU machine's own spec is kept even where nvidia-smi says nothing
        assert uv_install_lines(i.run(env={"GLYD_SPEC": "/tmp/x.whl"}))[0].endswith("/tmp/x.whl")


def test_install_constraints_are_the_acceptance_runs_versions():
    ic = load_constraints_module()
    text = open(INSTALL).read()
    lines = ic.listed(text)
    names = [l.split("==")[0] for l in lines]
    assert 150 < len(lines) < 400 and names == sorted(names, key=str.lower) and len(set(names)) == len(names), len(lines)
    assert all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][A-Za-z0-9.+!_-]*", l) for l in lines)
    versions = dict(l.split("==") for l in lines)
    assert "glyd" not in versions and versions["vllm"].startswith("0.30.") and versions["torch"].startswith("2.13"), versions["vllm"]
    pre = [l for l in lines if re.search(r"==[0-9][0-9.]*(a|b|rc|dev)[0-9]+$", l) and not l.startswith("opentelemetry-")]
    assert not pre, pre
    pyproject = open(os.path.join(HERE, "pyproject.toml")).read()
    lo, hi = re.search(r'"vllm>=(\d+\.\d+),<(\d+\.\d+)"', pyproject).groups()  # (the list holds the vLLM the package asks for)
    v = tuple(int(x) for x in versions["vllm"].split(".")[:2])
    assert tuple(int(x) for x in lo.split(".")) <= v < tuple(int(x) for x in hi.split(".")), (lo, hi, versions["vllm"])
    freeze = "glyd v0.26.0rc3 [with: ziglang==0.16.0]\n- glyd\nzzz==1.0\nglyd @ file:///w/glyd-0.26.0rc3-py3-none-any.whl\nglyd==0.26.0rc3\nAaa_Bbb==2.0\nopentelemetry-api==1.0.0b3\n"
    assert ic.parse(freeze) == ["Aaa_Bbb==2.0", "opentelemetry-api==1.0.0b3", "zzz==1.0"], ic.parse(freeze)
    try:
        ic.parse("foo==1.0rc1\n")
        raise AssertionError("a pre-release was taken")
    except ValueError:
        pass
    assert ic.listed(ic.rewrite(text, ["a==1", "b==2"])) == ["a==1", "b==2"]
    assert ic.rewrite(ic.rewrite(text, ["a==1"]), ic.listed(text)) == text  # (the list round-trips)


# --- the review's findings (onboard-review-1.md): each of these failed on the code it was written against -------------------------------------

def test_the_page_refuses_another_origin_and_a_rebinding_host():
    """S1: vLLM allows any origin, and checks no Host. A page in the user's browser can use a server on 127.0.0.1 (a script from any site, or a
    name that DNS points at 127.0.0.1: no CORS involved). With GLYD_LOCAL_ONLY (glyd run and glyd serve set it) a Host that is not local is
    421, an Origin that is not the server's own 403, a preflight too, a websocket's handshake too; the API and the page are answered to
    the page's own origin and to programs that send no Origin."""
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"api"})

    saved = os.environ.get("GLYD_LOCAL_ONLY")
    os.environ["GLYD_LOCAL_ONLY"] = "1"
    try:
        mw = page.ChatPage(app)
        ok = lambda method, path, **h: call_asgi(mw, method, path, headers=list(h.items()))[0]["status"]
        assert ok("GET", "/v1/models", Host="localhost:8000") == 200 and ok("GET", "/v1/models", Host="127.0.0.1:8000") == 200 and ok("GET", "/health", Host="[::1]:8000") == 200
        assert ok("POST", "/v1/chat/completions", Host="localhost:8000", Origin="http://localhost:8000") == 200  # (the page's own requests)
        assert ok("POST", "/v1/chat/completions", Host="127.0.0.1:8000", Origin="http://127.0.0.1:8000") == 200 and ok("GET", "/", Host="localhost:8000") == 200
        seen.clear()
        for host in ("evil.example", "evil.example:8000", "localhost.evil.example:8000", "127.0.0.1.evil.example", "192.168.1.5:8000", "0.0.0.0:8000"):  # (a rebinding page's own name)
            assert ok("GET", "/v1/models", Host=host) == 421, host
            assert ok("GET", "/", Host=host) == 421, host  # (not the page either)
        for origin in ("http://evil.example", "https://localhost:8000", "http://localhost:5173", "http://127.0.0.1:8000", "null", "http://localhost:8000.evil.example"):
            assert ok("POST", "/v1/chat/completions", Host="localhost:8000", Origin=origin) == 403, origin  # (only http:// and the Host itself is the server's own)
        assert ok("OPTIONS", "/v1/chat/completions", Host="localhost:8000", Origin="http://evil.example", **{"Access-Control-Request-Method": "POST"}) == 403  # (the preflight)
        assert seen == []  # (nothing refused reached vLLM)
        refused = call_asgi(mw, "GET", "/", headers=[("host", "evil.example")])
        assert refused[0]["status"] == 421 and b"evil.example" in refused[1]["body"] and json.loads(refused[1]["body"])["error"]["code"] == 421
        ws = call_asgi(mw, "GET", "/v1/realtime", scope_type="websocket", headers=[("host", "localhost:8000"), ("origin", "http://evil.example")])  # (CORS does not cover a websocket)
        assert ws == [{"type": "websocket.close", "code": 1008}] and seen == []
        assert call_asgi(mw, "GET", "/v1/realtime", scope_type="websocket", headers=[("host", "localhost:8000")])[0]["status"] == 200
        assert page.own_origins(8000) == ["http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"]
        os.environ.pop("GLYD_LOCAL_ONLY")  # (by hand, or where the user chose an address with --host: vLLM's own rules)
        mw = page.ChatPage(app)
        assert ok("GET", "/v1/models", Host="evil.example") == 200 and ok("POST", "/v1/chat/completions", Host="x:1", Origin="http://evil.example") == 200
    finally:
        if saved is None:
            os.environ.pop("GLYD_LOCAL_ONLY", None)
        else:
            os.environ["GLYD_LOCAL_ONLY"] = saved


def test_the_server_is_started_local_only_unless_a_host_was_chosen():
    """S1: glyd's own choice of address (127.0.0.1) comes with the guard and with vLLM's CORS limited to the page's origins; an address the user chose
    (glyd serve --host, -- --host) is theirs, and so is an --allowed-origins of their own."""
    m = model("Qwen3-8B", 16_381_470_720)
    s = pf.settings(m, L4_GPU, "run", environ={})
    s.local_only = True
    args = pf.vllm_args(m, s, "127.0.0.1", 8123)
    assert json.loads(args[args.index("--allowed-origins") + 1]) == ["http://localhost:8123", "http://127.0.0.1:8123", "http://[::1]:8123"]
    s.given = pf.flags_given(["--allowed-origins", '["https://mine.example"]'])
    assert "--allowed-origins" not in pf.vllm_args(m, s, "127.0.0.1", 8123)
    s.local_only, s.given = False, {}
    assert "--allowed-origins" not in pf.vllm_args(m, s, "0.0.0.0", 8123)
    seen = []
    srv = Fake(window=4096)
    d = tempfile.mkdtemp()
    os.environ["XDG_STATE_HOME"] = d
    try:
        mm = pf.model_of("fake/Model-1B", QWEN["Qwen/Qwen3-8B"], bf16=16_381_470_720, files=[("model.safetensors", 1)])

        def launch(model, st, host, port, extra, log, environ=None):
            seen.append((st.local_only, st.env.get("GLYD_LOCAL_ONLY"), host))
            open(log, "w").close()
            return run.Server(srv.base, log, Proc(None))

        with Patched(pf__setup_checks=lambda gpus=None: (L4_GPU, []), pf__probe_gpus=lambda: [L4_GPU], pf__load_model=lambda name: mm, pf__check_disk=lambda m: None,
                     pf__pick_port=lambda host, port, explicit: 8123, run__download=lambda m, ui: None, run__launch=launch, run__wait_ready=lambda *a, **k: None):
            for argv, extra, mode in ((["fake/Model-1B"], [], "run"), (["fake/Model-1B"], [], "serve"), (["fake/Model-1B", "--host", "0.0.0.0"], [], "serve"),
                                      (["fake/Model-1B"], ["--host", "127.0.0.1"], "serve"), (["fake/Model-1B"], ["--host", "127.0.0.1"], "run")):
                run.start(run.parse(mode, argv), extra, mode, run.Ui(Sink()))
        assert seen == [(True, "1", "127.0.0.1"), (True, "1", "127.0.0.1"), (False, None, "0.0.0.0"), (False, None, "127.0.0.1"), (False, None, "127.0.0.1")], seen
    finally:
        os.environ.pop("XDG_STATE_HOME", None)
        srv.shutdown()
        shutil.rmtree(d)


def test_the_open_server_warning_names_what_stays_open():
    """S2: a server on an address other than this computer's says it is plain HTTP, that the key goes in the environment (not on a command line,
    where ps shows it), and which paths stay open with a key."""
    no_key, with_key = run.open_note("0.0.0.0", None), run.open_note("0.0.0.0", "k")
    for text in (no_key, with_key):
        assert "plain HTTP" in text and "/health" in text and "/metrics" in text and "/tokenize" in text and "chat page" in text and "SSH tunnel" in text, text
    assert "VLLM_API_KEY=KEY glyd serve" in no_key and "--api-key" not in no_key and "There is no API key" in no_key
    assert "guards /v1 only" in with_key and "There is no API key" not in with_key
    srv = Fake(window=4096)
    d = tempfile.mkdtemp()
    os.environ["XDG_STATE_HOME"] = d
    try:
        mm = pf.model_of("fake/Model-1B", QWEN["Qwen/Qwen3-8B"], bf16=16_381_470_720, files=[("model.safetensors", 1)])
        with Patched(pf__setup_checks=lambda gpus=None: (L4_GPU, []), pf__probe_gpus=lambda: [L4_GPU], pf__load_model=lambda name: mm, pf__check_disk=lambda m: None, pf__pick_port=lambda host, port, explicit: 8123,
                     run__download=lambda m, ui: None, run__wait_ready=lambda *a, **k: None, run__launch=lambda *a, **k: (open(a[5], "w").close(), run.Server(srv.base, a[5], Proc(None)))[1]):
            ui = run.Ui(Sink())
            run.start(run.parse("serve", ["fake/Model-1B", "--host", "0.0.0.0"]), [], "serve", ui, environ={})
            assert "Note: listening on 0.0.0.0" in ui.f.text() and "plain HTTP" in ui.f.text() and "There is no API key" in ui.f.text()
            ui = run.Ui(Sink())
            run.start(run.parse("serve", ["fake/Model-1B", "--host", "0.0.0.0"]), [], "serve", ui, environ={"VLLM_API_KEY": "k"})
            assert "guards /v1 only" in ui.f.text() and "taken off" not in ui.f.text()  # (a key from the environment was never on a command line)
            ui = run.Ui(Sink())
            run.start(run.parse("serve", ["fake/Model-1B", "--host", "0.0.0.0"]), ["--api-key", "k"], "serve", ui, environ={})
            assert "taken off the server's command line" in ui.f.text() and "VLLM_API_KEY=KEY yourself" in ui.f.text()
    finally:
        os.environ.pop("XDG_STATE_HOME", None)
        srv.shutdown()
        shutil.rmtree(d)


def test_a_second_ctrl_c_does_not_skip_the_sweep():
    """S7b: Ctrl-C while the server is stopped raised out of Server.stop before its sweep, and the engine kept the GPU's memory."""
    if not sys.platform.startswith(("linux", "darwin")):
        return print("test_a_second_ctrl_c_does_not_skip_the_sweep: skipped (needs process groups)")
    import signal
    d = tempfile.mkdtemp()
    pidfile = os.path.join(d, "engine.pid")
    code = ("import signal, subprocess, sys, time\nchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\nopen(sys.argv[1], 'w').write(str(child.pid))\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(600)")  # (an API server that does not stop on SIGTERM, with an engine)
    proc = subprocess.Popen([sys.executable, "-c", code, pidfile], start_new_session=True)
    saved = run.STOP_API_WAIT, run.STOP_WAIT
    try:
        for _ in range(100):
            if os.path.exists(pidfile) and open(pidfile).read():
                break
            time.sleep(0.1)
        engine = int(open(pidfile).read())
        server = run.Server("http://127.0.0.1:1", "", proc)
        run.STOP_API_WAIT, run.STOP_WAIT = 3, (0.3, 0.3, 2)
        threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGINT)).start()  # (the second Ctrl-C, while the first one's stop waits)
        ui = run.Ui(Sink())
        try:
            server.stop(ui)
        except KeyboardInterrupt:
            raise AssertionError("a Ctrl-C meanwhile ended the stop before its sweep")
        time.sleep(0.3)
        assert not alive(proc.pid) and not alive(engine), "the server or its engine outlived the stop"
        assert "Stopping the server" in ui.f.text()
        assert signal.getsignal(signal.SIGINT) == signal.default_int_handler  # (and Ctrl-C works again after)
    finally:
        run.STOP_API_WAIT, run.STOP_WAIT = saved
        for pid in [proc.pid] + ([int(open(pidfile).read())] if os.path.exists(pidfile) and open(pidfile).read() else []):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        shutil.rmtree(d)


def test_a_server_still_loading_is_not_waited_for_as_long():
    """S7: Ctrl-C while the model loads took 50 s on the L4 (the engine does not answer SIGTERM until the load is over, and the API server's
    wait is 30 s): a server that has not answered /health is swept after a few seconds, and says it is still loading; a ready one gets its
    graceful stop."""
    if not sys.platform.startswith(("linux", "darwin")):
        return print("test_a_server_still_loading_is_not_waited_for_as_long: skipped (needs process groups)")
    import signal
    code = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(600)"  # (a server that is busy, and does not stop on SIGTERM)
    saved = run.STOP_API_WAIT, run.STOP_LOADING_WAIT, run.STOP_WAIT
    run.STOP_API_WAIT, run.STOP_LOADING_WAIT, run.STOP_WAIT = 2.0, 0.2, (0.2, 0.2, 2)
    try:
        took = {}
        for ready in (False, True):
            proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            time.sleep(0.3)
            server = run.Server("http://127.0.0.1:1", "", proc)
            server.ready = ready
            ui = run.Ui(Sink())
            t0 = time.time()
            server.stop(ui)
            took[ready] = time.time() - t0, ui.f.text()
            assert not alive(proc.pid)
        assert took[False][0] < 1.5 and "still loading" in took[False][1], took[False]
        assert took[True][0] >= 1.8 and "a few seconds" in took[True][1] and "still loading" not in took[True][1], took[True]  # (the full graceful wait, then the sweep)
    finally:
        run.STOP_API_WAIT, run.STOP_LOADING_WAIT, run.STOP_WAIT = saved


def test_signals_nohup_download_and_a_terminal_that_went():
    """S7c, S7d, nit 10: nohup's ignored SIGHUP stays ignored (ssh host 'nohup glyd serve MODEL &' is how a server outlives its login); Ctrl-C in a
    download ends the process at once (huggingface_hub's threads are not daemons); a terminal that has gone is not an error; a pipe closed by
    `| head` is not a traceback; no SIGHUP (Windows) is not an AttributeError."""
    import signal
    if hasattr(signal, "SIGHUP"):
        before_hup, before_term = signal.getsignal(signal.SIGHUP), signal.getsignal(signal.SIGTERM)
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            with Patched(run__cmd_doctor=lambda argv: 0):
                run.main("doctor", [])
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN  # (nohup)
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
            with Patched(run__cmd_doctor=lambda argv: 0):
                run.main("doctor", [])
            assert signal.getsignal(signal.SIGHUP) not in (signal.SIG_IGN, signal.SIG_DFL)  # (not ignored: a hangup stops the server)
        finally:
            signal.signal(signal.SIGHUP, before_hup)
            signal.signal(signal.SIGTERM, before_term)
    seen = []
    stub = types.SimpleNamespace(SIGTERM=15, SIGINT=2, SIG_IGN=1, signal=lambda *a: seen.append(a), getsignal=lambda n: None)  # (no SIGHUP in it)
    real_signal, run.signal = run.signal, stub
    try:
        run.install_signal_handlers()
    finally:
        run.signal = real_signal
    assert [a[0] for a in seen] == [15]
    calls, real_exit = [], os._exit
    os._exit = lambda code: calls.append(code)
    try:
        with Patched(run__main=lambda cmd, argv: 130):
            assert cli.main(["run", "M"]) == 130 and calls == [130]  # (S7c)
        with Patched(run__main=lambda cmd, argv: 0):
            assert cli.main(["doctor"]) == 0 and calls == [130]
    finally:
        os._exit = real_exit
    class Gone:  # (a terminal that was closed: writes fail with EIO)
        def write(self, t):
            raise OSError(5, "Input/output error")

        def flush(self):
            raise OSError(5, "Input/output error")

        def isatty(self):
            return True

    ui = run.Ui(Gone())
    ui.line("Stopping the server")
    ui.status("Loading")
    ui.note("x")
    err, real, fd1 = Sink(), sys.stderr, os.dup(1)  # (the handler points standard output at /dev/null, as a CLI that was cut off by `| head` should: put it back)
    sys.stderr = err
    try:
        with Patched(run__cmd_doctor=lambda argv: (_ for _ in ()).throw(BrokenPipeError())):
            assert run.main("doctor", []) == 0  # (glyd doctor | head -1)
    finally:
        sys.stderr = real
        os.dup2(fd1, 1)
        os.close(fd1)
    assert "unexpected error" not in err.text() and "Traceback" not in err.text()
    os.environ["TERM"] = "dumb"
    try:
        assert run.Ui(Sink(tty=True)).tty is False  # (nit 9: \r and ESC [ K do not work there)
    finally:
        os.environ.pop("TERM")
    assert run.Ui(Sink(tty=True)).tty is True


def test_a_failure_says_the_next_step():
    """S8: the disk full is not the network; a retry is diagnosed from its own lines; a failure with no retry still says what to do; a user's mistakes are not
    for the issue tracker."""
    import errno
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda *a, **k: (_ for _ in ()).throw(OSError(errno.ENOSPC, "No space left on device"))
    utils, lg = types.ModuleType("huggingface_hub.utils"), types.ModuleType("huggingface_hub.utils.logging")
    lg.set_verbosity_error = lambda: None
    utils.logging = lg
    utils.tqdm = type("tqdm", (), {"__init__": lambda self, *a, **k: None, "close": lambda self: None})
    fake.utils = utils
    mods = {"huggingface_hub": fake, "huggingface_hub.utils": utils, "huggingface_hub.utils.logging": lg}
    saved = {k: sys.modules.get(k) for k in mods}
    d = tempfile.mkdtemp()
    os.environ["HF_HUB_CACHE"] = d
    try:
        sys.modules.update(mods)
        mm = pf.model_of("some/Model", QWEN["Qwen/Qwen3-8B"], bf16=16_381_470_720, files=[("model.safetensors", 10**9)])
        r = raises(lambda: run.download(mm, run.Ui(Sink())), "is full")  # (S8a: not "check the network connection")
        assert "network" not in r.what + r.fix and "HF_HOME=/path/on/that/drive glyd run some/Model" in r.fix and d in r.what
    finally:
        os.environ.pop("HF_HUB_CACHE", None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        shutil.rmtree(d)
    log = os.path.join(tempfile.mkdtemp(), "x.log")  # (S8b: the lines of the first start are not the second's)
    open(log, "w").write("(EngineCore) ValueError: ... the estimated maximum model length is 3280. Try increasing\n")
    start = os.path.getsize(log)
    open(log, "a").write("(EngineCore) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB\n")
    assert run.diagnose(run.log_tail(log)).kind == "context" and run.diagnose(run.log_tail(log, start=start)).kind == "oom"
    f = run.diagnose("torch.OutOfMemoryError: CUDA out of memory")
    r = run.stopped(f, M8, OWNER_GPU, "run", log)  # (S8c)
    assert "Close programs that use the GPU" in r.fix and "The server's log" in r.fix and "Report it" not in r.fix
    f = run.diagnose("estimated maximum model length is 2000. Try increasing")  # (no retry possible: under 4,096: still says what to do)
    assert f.kind == "context" and "Close programs that use the GPU" in run.stopped(f, M8, OWNER_GPU, "run", log).fix
    raises(lambda: pf.settings(M8, L4_GPU, "run", context=100_000, environ={}), "own window is 40,960 tokens")  # (S8d: a mistake of the user's, said before the download)
    f = run.diagnose("ValueError: Model architectures ['FooForCausalLM'] are not supported for now.")
    assert f.kind == "arch" and "FooForCausalLM" in f.what and "Report it" not in f.fix
    f = run.diagnose("User-specified max_model_len (100000) is greater than the derived max_model_len (max_position_embeddings=40960 or model_max_length=None in model's config.json).")
    assert f.kind == "window" and "40,960" in f.what and "--context 40,960" in f.fix
    f = run.diagnose("", rc=-9)
    assert f.kind == "killed" and "ran out of memory" in f.what and "Report it" not in f.fix and run.diagnose("", rc=1).kind == "other"


def test_a_float16_checkpoint_is_refused_before_the_download():
    """S9: the plugin runs bfloat16, and vLLM stops a float16 checkpoint's load with 'float16 is not supported for quantization method glyd': 14 GB later."""
    f16 = pf.model_of("meta/Llama-2-7b", QWEN["Qwen/Qwen3-8B"], bf16=13_000_000_000, dtype="f16")
    r = raises(lambda: pf.check_dtype(f16, {}), "float16")
    assert "bf16 version" in r.fix and "--dtype bfloat16" in r.fix
    pf.check_dtype(f16, pf.flags_given(["--dtype", "bfloat16"]))
    pf.check_dtype(M8, {})
    hub = lambda name: (QWEN["Qwen/Qwen3-8B"], {"F16": 6_000_000_000, "BF16": 10}, [("model.safetensors", 12_000_000_000)])
    assert pf.load_model("meta/Llama-2-7b", hub=hub).dtype == "f16" and pf.load_model("meta/Llama-2-7b", hub=lambda n: (QWEN["Qwen/Qwen3-8B"], {"BF16": 6_000_000_000}, [("model.safetensors", 12_000_000_000)])).dtype == "bf16"
    called = []
    d = tempfile.mkdtemp()
    os.environ["XDG_STATE_HOME"] = d
    try:
        with Patched(pf__setup_checks=lambda gpus=None: (L4_GPU, []), pf__probe_gpus=lambda: [L4_GPU], pf__load_model=lambda name: f16, run__download=lambda m, ui: called.append(1)):
            raises(lambda: run.start(run.parse("run", ["meta/Llama-2-7b"]), [], "run", run.Ui(Sink())), "float16")
        assert called == []  # (before the download)
    finally:
        os.environ.pop("XDG_STATE_HOME", None)
        shutil.rmtree(d)


def test_glyd_login_asks_again_and_says_who():
    """S10: huggingface_hub 1.x skips login() where any token is saved, so glyd login did nothing and said nothing, and a stale token stayed; the token
    glyd reads for the Hub's metadata is huggingface_hub's own, and goes to the host it was meant for only."""
    calls = []
    fake = types.ModuleType("huggingface_hub")
    fake.login = lambda token=None, *, add_to_git_credential=False, skip_if_logged_in=True: calls.append((token, add_to_git_credential, skip_if_logged_in))
    fake.whoami = lambda: {"name": "octocat"}
    saved = sys.modules.get("huggingface_hub")
    out, real = Sink(), sys.stdout
    try:
        sys.modules["huggingface_hub"] = fake
        sys.stdout = out
        try:
            assert run.cmd_login([]) == 0
        finally:
            sys.stdout = real
        assert calls == [(None, False, False)] and "Logged in to Hugging Face as octocat." in out.text(), (calls, out.text())
        fake.login = lambda *a, **k: (_ for _ in ()).throw(ValueError("Invalid token passed!"))
        r = raises(lambda: run.cmd_login([]), "did not accept that token")
        assert "settings/tokens" in r.fix and "glyd login" in r.fix
        fake.get_token = lambda: "tok-from-the-hub"  # (S10b: its answer, not a file under HF_HOME that the hub may not use)
        d = tempfile.mkdtemp()
        os.environ["HF_HOME"] = d
        open(os.path.join(d, "token"), "w").write("stale-token")
        try:
            assert fit._token() == "tok-from-the-hub"
            del fake.get_token
            assert fit._token() == "stale-token"  # (no get_token: the file)
            sys.modules.pop("huggingface_hub")
            assert fit._token() == "stale-token"
        finally:
            os.environ.pop("HF_HOME")
            shutil.rmtree(d)
    finally:
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved
    import urllib.request
    h = fit._SameHostAuth()
    req = urllib.request.Request("https://huggingface.co/api/models/a/b", headers={"Authorization": "Bearer secret", "User-Agent": "glyd"})
    away = h.redirect_request(req, None, 302, "Found", {}, "https://cas-bridge.example.net/x")
    same = h.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/api/models/a/c")
    assert "Authorization" not in away.headers and "Authorization" not in away.unredirected_hdrs and same.headers.get("Authorization") == "Bearer secret"


def test_the_gpu_picked_is_one_glyd_can_use():
    """S11: the freest GPU was picked first and refused for its age (a P40 beside a 3060); MIG showed the whole GPU's memory to a process that sees a slice."""
    p40 = pf.Gpu(0, "Tesla P40", 24 * GiB, 23 * GiB, (6, 1), "570", (12, 8))
    r3060 = pf.Gpu(1, "NVIDIA GeForce RTX 3060", 12 * GiB, 11 * GiB, (8, 6), "570", (12, 8))
    assert pf.pick_gpu([p40, r3060], "").index == 1 and pf.setup_checks(False, gpus=[p40, r3060])[0].index == 1
    assert pf.pick_gpu([p40, r3060], "0").index == 0  # (CUDA_VISIBLE_DEVICES names the old one: refused, as asked for)
    r = raises(lambda: pf.setup_checks(False, gpus=[p40, pf.Gpu(1, "Tesla P4", 8 * GiB, 7 * GiB, (6, 1), "570", (12, 8))]), "too old")
    assert "none of this computer's 2 GPUs" in r.what
    mig = pf.Gpu(0, "NVIDIA A100 80GB", 80 * GiB, 79 * GiB, (8, 0), "595", (13, 2), mig=True)
    r = raises(lambda: pf.setup_checks(False, gpus=[mig]), "MIG is on")
    assert "--gpu-memory-utilization" in r.fix and "nvidia-smi -i 0 -mig 0" in r.fix
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-1234"
    try:
        assert pf.setup_checks(False, gpus=[mig])[0] is mig  # (a slice named: the user's own sizing)
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    raises(lambda: pf.setup_checks(False, gpus=[pf.Gpu(0, "NVIDIA A100 MIG 1g.10gb", 0, 0, (8, 0), "595", (13, 2))]), "did not say how much memory")  # ([N/A] memory)
    r4090 = pf.Gpu(1, "NVIDIA GeForce RTX 4090", 24 * GiB, 23 * GiB, (8, 9), "595", (13, 2))  # N7: a MIG-enabled card with more memory is not picked ahead of a usable one
    assert pf.pick_gpu([mig, r4090], "").index == 1 and pf.setup_checks(False, gpus=[mig, r4090])[0].index == 1
    assert pf.pick_gpu([mig, r4090], "0").index == 0 and pf.pick_gpu([mig, r4090], "MIG-1234").index == 0  # (named: as asked for)
    assert pf.pick_gpu([mig], "").index == 0  # (the only one: picked, for the MIG refusal)
    g = pf.probe_gpus(smi("0, NVIDIA A100 80GB, 81920, 80000, 400, 595.91.07, 8.0, Disabled, Enabled\n"))[0]
    assert g.mig and not pf.probe_gpus(smi("0, NVIDIA L4, 23034, 22566, 469, 595.91.07, 8.9, Disabled, Disabled\n"))[0].mig


def test_small_review_nits_in_the_settings_and_the_ports():
    """Nits 1, 2, 5, 6, 8: a context the 92% cap makes smaller than the smallest chat is refused; a port that is not one; the last ten logs by time; a
    port listening on [::1]; the zig script written whole."""
    gpu = pf.Gpu(0, "x", int(22.91 * GiB), int(22.76 * GiB), (8, 9), "595", (13, 2))
    raises(lambda: pf.settings(model("Qwen3-14B", 29_540_000_000), gpu, "serve", environ={}), "need more than the 92% of this GPU's memory")  # (it returned a 3,072-token context)
    for bad in ("abc", "99999", "0", "-1", "80.5"):
        raises(lambda bad=bad: pf.parse_port(bad), "is not a port number")
    assert pf.parse_port("8000") == 8000 and pf.parse_port("8k") == 8000
    try:
        run.parse("run", ["M", "--port", "abc"])
        raise AssertionError("a port that is not a number was taken")
    except SystemExit as e:
        assert e.code == 2
    d = tempfile.mkdtemp()
    os.environ["XDG_STATE_HOME"] = d
    try:
        logs = os.path.join(d, "glyd", "logs")
        os.makedirs(logs)
        for i, name in enumerate(["serve-20260101-000001", "run-20260101-000002"] + [f"serve-2026010{n}-000000" for n in range(3, 9)] + [f"run-2026011{n}-000000" for n in range(0, 4)]):
            path = os.path.join(logs, name + ".log")
            open(path, "w").close()
            os.utime(path, (1_000_000 + i, 1_000_000 + i))  # (by name, run-* sorts before serve-*: by time the oldest are the first written)
        new = run.new_log("run")
        left = sorted(os.listdir(logs))
        assert len(left) == 9 and "serve-20260101-000001.log" not in left and "run-20260101-000002.log" not in left and os.stat(logs).st_mode & 0o077 == 0, left
        assert new.startswith(logs)
        # nit 8: a zig script that is a link at its name is replaced, not written through; and it is whole at every moment
        elsewhere = os.path.join(d, "elsewhere")
        open(elsewhere, "w").write("not ours\n")
        os.symlink(elsewhere, os.path.join(d, "glyd", "zigcc"))
        path = run.zig_cc()
        assert open(elsewhere).read() == "not ours\n" and not os.path.islink(path) and os.access(path, os.X_OK) and open(path).read().startswith("#!/bin/sh")
        assert run.zig_cc() == path and not [f for f in os.listdir(os.path.join(d, "glyd")) if f.startswith(".zigcc-")]
    finally:
        os.environ.pop("XDG_STATE_HOME", None)
        shutil.rmtree(d)
    assert run.base_url("::1", 8000) == "http://[::1]:8000" and run.base_url("0.0.0.0", 8000) == "http://127.0.0.1:8000" and run.base_url("127.0.0.1", 9) == "http://127.0.0.1:9"
    import socket
    if socket.has_ipv6:  # (nit 6: a dev server on [::1]:8000 is what a browser opening http://localhost:8000 reaches)
        s6 = socket.socket(socket.AF_INET6)
        try:
            s6.bind(("::1", 0))
            s6.listen(1)
            port = s6.getsockname()[1]
            assert not pf.port_free("127.0.0.1", port), "a program listening on [::1] was not seen"
            assert pf.pick_port("127.0.0.1", port, False) != port
        except OSError:
            pass  # (no IPv6 loopback on this machine)
        finally:
            s6.close()
    assert pf._run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfeok')"]).endswith("ok")  # (nit 10: output in another encoding)


def stdin_at_its_end():
    """sys.stdin as a pipe nothing was written to and whose writer has gone: a script's `</dev/null`, a CI step's closed input."""
    r, w = os.pipe()
    os.close(w)
    return os.fdopen(r)


def test_an_empty_prompt_is_found_before_the_download_and_a_served_model_is_not_started_again():
    """Nits 3 and 4; N1 of the re-review: the empty prompt is found once this machine and the model were checked (a machine with no GPU says that
    first), and before the download and the load."""
    m = pf.model_of("fake/Model-1B", QWEN["Qwen/Qwen3-8B"], bf16=16_381_470_720, files=[("model.safetensors", 1)])
    calls = []
    with Patched(pf__setup_checks=lambda gpus=None: (L4_GPU, []), pf__probe_gpus=lambda: [L4_GPU], pf__load_model=lambda name: m, pf__check_disk=lambda m: None,
                 pf__pick_port=lambda host, port, explicit: 8123, run__download=lambda m, ui: calls.append("download"), run__launch=lambda *a, **k: calls.append("launch")):
        raises(lambda: run.cmd_run(["M", "--prompt", "  "]), "the prompt is empty")
        assert calls == []  # (an argument: found before anything is looked at)
        real = sys.stdin
        sys.stdin = stdin_at_its_end()  # (standard input at its end at once: nothing was piped)
        try:
            raises(lambda: run.cmd_run(["M"]), "standard input is empty")
        finally:
            sys.stdin.close()
            sys.stdin = real
    assert calls == []  # (found before any download or load)
    srv = Fake()
    d = tempfile.mkdtemp()
    out, real = Sink(), sys.stderr
    try:
        with Patched(run__start=lambda *a, **k: run.Server(srv.base, model="fake/Model-1B")):
            sys.stderr = out
            assert run.main("serve", ["M"]) == 0
    finally:
        sys.stderr = real
        srv.shutdown()
        shutil.rmtree(d)
    assert "Nothing to start" in out.text() and "Press Ctrl-C" not in out.text(), out.text()


def test_a_machine_with_no_gpu_says_so_before_it_looks_at_stdin():
    """N1 of the re-review: `glyd run` with standard input at its end (a CI step's, `ssh host glyd run M`, cron, `docker run` without -i) said
    "no prompt: standard input is empty" before any GPU was looked for, so a runner with no GPU never heard what glyd run needs, and
    release.yml's wheel test, which greps for the GPU message, failed."""
    for platform, said in (("linux", "no NVIDIA GPU answered"), ("darwin", "needs Linux and an NVIDIA GPU")):
        real = sys.stdin
        sys.stdin = stdin_at_its_end()
        try:
            with Patched(pf__probe_gpus=lambda: (_ for _ in ()).throw(pf.no_gpu_refusal(platform))):
                r = raises(lambda: run.cmd_run(["Qwen/Qwen3-8B"]), said)
                assert "standard input" not in r.what + r.fix
        finally:
            sys.stdin.close()
            sys.stdin = real
    # and as the CLI runs it, with nothing on PATH that is nvidia-smi and standard input closed: the step's own lines
    empty = tempfile.mkdtemp()
    try:
        p = subprocess.run([sys.executable, "-m", "glyd.cli", "run", "Qwen/Qwen3-8B"], cwd=HERE, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           env={**os.environ, "PATH": empty, "PYTHONPATH": HERE})
        assert p.returncode != 0 and "NVIDIA GPU" in p.stdout + p.stderr and "standard input" not in p.stdout + p.stderr, (p.returncode, p.stdout, p.stderr)
    finally:
        shutil.rmtree(empty)


def test_the_suite_runs_with_sigint_ignored():
    """N2 of the re-review: a background job of a non-interactive shell has SIGINT ignored, which Python then keeps, and the tests that send
    themselves a Ctrl-C (and one that asserts the default handler) failed. The module puts the default handler back."""
    p = subprocess.run([sys.executable, "-c", "import signal; signal.signal(signal.SIGINT, signal.SIG_IGN); import test_onboard; "
                        "assert signal.getsignal(signal.SIGINT) == signal.default_int_handler"], cwd=HERE, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-300:]


def test_several_api_keys_are_refused_and_a_hugging_face_token_stays_off_the_command_line():
    """N3 and N6 of the re-review: `-- --api-key K1 K2` stayed on the server's command line and in its log, and `glyd serve` died on its banner's
    401 ("unexpected error: ApiError"); `-- --hf-token T` was on the command line and in the log as `--api-key` had been."""
    r = raises(lambda: pf.take_api_key(["--api-key", "K1", "K2"], {}), "one API key")
    assert "VLLM_API_KEY" in r.fix and "vllm serve by hand" in r.fix
    raises(lambda: pf.take_api_key(["--api-key=K1", "--api-key", "K2", "K3"], {}), "one API key")
    assert pf.take_api_key(["--api-key", "K1", "--max-model-len", "4096"], {}) == ("K1", ["--max-model-len", "4096"])
    assert pf.take_api_key(["--api-key", "K1", "--api-key", "K2"], {}) == ("K2", [])  # (twice: the last one, as argparse takes it)
    assert pf.take_api_key([], {"VLLM_API_KEY": "E"}) == ("E", [])
    assert pf.take_secret(["--hf-token", "T", "--dtype", "bfloat16"], ("--hf-token", "--hf_token"), "token") == ("T", ["--dtype", "bfloat16"])
    assert pf.take_secret(["--hf_token=T"], ("--hf-token", "--hf_token"), "token") == ("T", [])
    assert pf.take_secret(["--hf-token", "--dtype", "bfloat16"], ("--hf-token",), "token") == (None, ["--hf-token", "--dtype", "bfloat16"])  # (alone: the saved login, no secret)
    with FakeVllmHome() as h:
        saved_wait, saved_env = run.wait_ready.__defaults__, os.environ.get("HF_TOKEN")
        run.wait_ready.__defaults__ = (8,)
        try:
            with Patched(**h.patches):
                out, err, port = Sink(), Sink(), h.free_port()
                code = h.go(["serve", h.model, "--port", str(port), "--", "--hf-token", "hf_SECRET123"], out, err, cancel_when="Press Ctrl-C to stop.")
        finally:
            run.wait_ready.__defaults__ = saved_wait
            if saved_env is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = saved_env
        assert code == 130 and "Press Ctrl-C to stop." in err.text(), err.text()
        started = json.load(open(os.environ["FAKE_DUMP"]))
        assert "--hf-token" not in started["argv"] and "hf_SECRET123" not in " ".join(started["argv"]) and started["env"]["HF_TOKEN"] == "hf_SECRET123", started
        logs = "".join(open(f).read() for f in glob.glob(os.path.join(h.state, "glyd", "logs", "*.log")))
        assert "hf_SECRET123" not in logs and "hf_SECRET123" not in err.text() and "Hugging Face token was taken off" in err.text()
        out, err = Sink(), Sink()
        with Patched(**h.patches):  # (several keys: refused before anything starts)
            code = h.go(["serve", h.model, "--port", str(h.free_port()), "--", "--api-key", "K1", "K2"], out, err, cancel_when="Press Ctrl-C to stop.")
        assert code == 1 and "one API key" in err.text() and "Loading" not in err.text(), (code, err.text())


def test_sigterm_during_a_download_does_not_wait_for_the_shard():
    """N4 of the re-review: Ctrl-C ended through os._exit, but SIGTERM and SIGHUP ended by SystemExit, and the interpreter then joined
    huggingface_hub's download threads (not daemons): 143 after 12 s for a shard of 12 s."""
    if not hasattr(__import__("signal"), "SIGHUP"):
        return print("test_sigterm_during_a_download_does_not_wait_for_the_shard: skipped (no SIGHUP)")
    import signal
    code = ("import sys, threading, time\n"
            "from glyd import cli\nfrom glyd.gpu import run\n"
            "threading.Thread(target=time.sleep, args=(12,)).start()  # a shard in flight: a thread that is not a daemon\n"
            "run.cmd_run = lambda argv: (print('waiting', flush=True), time.sleep(30))\n"
            "sys.exit(cli.main(['run', 'M']))\n")
    for sig, status in ((signal.SIGTERM, 143), (signal.SIGHUP, 129)):
        p = subprocess.Popen([sys.executable, "-c", code], cwd=HERE, env={**os.environ, "PYTHONPATH": HERE}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert p.stdout.readline().strip() == "waiting"
        t0 = time.time()
        p.send_signal(sig)
        rc = p.wait(timeout=30)
        assert rc == status and time.time() - t0 < 5, (sig, rc, time.time() - t0)


def test_a_loopback_address_that_was_chosen_says_what_is_off():
    """N5 of the re-review: `--host 127.0.0.1` (or localhost, ::1) is an address the user named, so the Host and Origin checks and the CORS limit are
    off for it, as the README says; nothing said so where the server starts. glyd's own default says nothing: the checks are on there."""
    with FakeVllmHome() as h:
        saved_wait = run.wait_ready.__defaults__
        run.wait_ready.__defaults__ = (8,)
        try:
            with Patched(**h.patches):
                said = {}
                for name, argv in (("default", []), ("chosen", ["--host", "127.0.0.1"]), ("flag", ["--", "--host", "localhost"])):
                    out, err = Sink(), Sink()
                    h.go(["serve", h.model, "--port", str(h.free_port())] + argv, out, err, cancel_when="Press Ctrl-C to stop.")
                    said[name] = "the checks that keep another web page's script" in err.text()
        finally:
            run.wait_ready.__defaults__ = saved_wait
    assert said == {"default": False, "chosen": True, "flag": True}, said


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    if "--serve" in sys.argv:  # a fake server with the chat page, for a browser
        srv = Fake(window=int(os.environ.get("WINDOW", 200)), field="reasoning", port=int(sys.argv[sys.argv.index("--serve") + 1]), start=False)
        print("fake server on", srv.base, flush=True)
        srv.serve_forever()
    failed = 0
    for t in TESTS:
        try:
            t()
            print("ok  ", t.__name__)
        except Exception:
            failed += 1
            import traceback
            print("FAIL", t.__name__)
            traceback.print_exc()
    print(f"{len(TESTS) - failed} of {len(TESTS)} passed")
    sys.exit(1 if failed else 0)
