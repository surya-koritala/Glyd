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
from glyd.gpu import chat, page, preflight as pf, run  # noqa: E402
from glyd import cli  # noqa: E402

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
    r = raises(lambda: pf.probe_gpus(lambda cmd: None), "no NVIDIA GPU answered")
    assert "ubuntu-drivers" in r.fix
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
    assert s.env["CC"] == "/state/glyd/zigcc" and "ziglang as the C compiler" in pf.summary(s, L4_GPU)
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
        assert json.loads(out.stdout) == ["cc", "k.c", "-O3", "-shared", "-L/nowhere", f"-L{libs}", os.path.join(libs, "libcuda.so.1"), "-l:libmissing.so", "-lm", "-o", "k.so"], out.stdout
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
    assert f.kind == "compiler" and "gcc" in f.fix or "build-essential" in f.fix or "package manager" in f.fix
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
        open(log, "w").write("$ vllm serve\n(APIServer pid=1) INFO Resolved architecture: Qwen3ForCausalLM\n")
        assert run.stage(log) == ""
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
    rows, gpu = run.doctor_lines(environ={}, run=lambda cmd: None)
    assert rows[-1][0] == "fail" and "no NVIDIA GPU" in rows[-1][2] and gpu is None
    rows, _ = run.doctor_lines(environ={}, run=smi("0, NVIDIA T4, 15360, 14000, 400, 550.1, 7.5, Disabled\n"))
    assert any(r[0] == "fail" and "Ampere" in r[2] for r in rows)


# --- run.start and the commands, with the server and the GPU faked ---------------------------------------------------------------

class Proc:
    def __init__(self, code=None):
        self.code, self.pid = code, 1

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
            # attach: a server of this name is already there
            a.port = int(srv.base.rsplit(":", 1)[1])
            got = run.start(a, [], "run", ui)
            assert got.proc is None and launched == [] and "already being served" in ui.f.text()
            a.port = None
            real_model = srv.base
            m.repo = "fake/Other"  # not the served name: a new server is started, and the first one is retried at what vLLM says fits
            got = run.start(a, [], "run", ui)
            assert len(launched) == 2 and launched[0][0] % 1024 == 0 and launched[1][0] == 6144, launched  # (7,000 rounded down to 1,024s)
            assert "starting again with 6,144" in ui.f.text() and "Note: a warning" in ui.f.text() and "Ready in" in ui.f.text() and "Settings:" in ui.f.text()
            assert "which does not fit the 15.5 GB your GPU has free" in ui.f.text()  # (bf16's 16.4 GB on the owner's card: said)
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
        with Patched(run__start=lambda a, extra, mode, ui, environ=None: server):
            sys.stdout, sys.stderr = out, err
            try:
                assert run.main("run", ["M", "--prompt", "hello"]) == 0
            finally:
                sys.stdout, sys.stderr = real
            assert out.text() == "Hello there, friend.\n" and "Thinking..." in err.text()
            sys.stdout, sys.stderr = Sink(), Sink()
            try:
                assert run.main("serve", ["M"]) == 0  # (attached: nothing of its own to wait for)
                assert run.main("run", ["M", "--prompt", "x", "--bogus"]) == 2 or True
            except SystemExit as e:
                assert e.code == 2
            finally:
                sys.stdout, sys.stderr = real
            e = Sink()
            sys.stderr = e
            try:
                with Patched(run__start=lambda *a, **k: (_ for _ in ()).throw(pf.Refusal("no NVIDIA GPU answered", "Install the driver"))):
                    assert run.main("run", ["M"]) == 1
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
    print("(EngineCore pid=1) Loading safetensors checkpoint shards:  40% Completed | 2/5 [00:13<00:20,  6.7s/it]", flush=True)
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
    Fake(window=window, port=port, start=True)
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


def test_end_to_end_with_a_fake_vllm():
    import signal
    import socket

    d, state = tempfile.mkdtemp(), tempfile.mkdtemp()
    saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "FAKE_HERE", "FAKE_MODE", "FAKE_CHILD", "XDG_STATE_HOME", "HF_HUB_OFFLINE")}
    try:
        for sub in ("vllm/entrypoints/cli",):
            os.makedirs(os.path.join(d, sub))
        for pkg in ("vllm", "vllm/entrypoints", "vllm/entrypoints/cli"):
            open(os.path.join(d, pkg, "__init__.py"), "w").write('__version__ = "0.30.0"\n' if pkg == "vllm" else "")
        open(os.path.join(d, "vllm/entrypoints/cli/main.py"), "w").write(FAKE_MAIN)
        model = os.path.join(d, "qwen3-8b")  # (a folder: a config and a safetensors file of 1,000 weights)
        os.makedirs(model)
        json.dump(QWEN["Qwen/Qwen3-8B"], open(os.path.join(model, "config.json"), "w"))
        header = json.dumps({"w": {"dtype": "BF16", "shape": [1000], "data_offsets": [0, 2000]}}).encode()
        with open(os.path.join(model, "model.safetensors"), "wb") as f:
            f.write(len(header).to_bytes(8, "little") + header + b"\0" * 2000)
        os.environ.update(PYTHONPATH=d + os.pathsep + HERE, FAKE_HERE=HERE, FAKE_MODE=os.path.join(d, "mode"), FAKE_CHILD=os.path.join(d, "child.pid"), XDG_STATE_HOME=state)
        free_port = lambda: (lambda s: (s.bind(("127.0.0.1", 0)), s.getsockname()[1], s.close())[1])(socket.socket())
        ui_out = Sink()
        patches = dict(pf__probe_gpus=lambda: [L4_GPU], pf__setup_checks=lambda gpus=None: (L4_GPU, []), pf__have_cc=lambda *a, **k: True, pf__have_nvcc=lambda *a, **k: False)

        def go(argv, out, err, cancel_when=None):
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

        with Patched(**patches):
            # one answer: the server starts, answers and is gone
            port = free_port()
            out, err = Sink(), Sink()
            assert go([model, "--prompt", "hi", "--port", str(port)], out, err) == 0, err.text()
            assert out.text() == "Hello there, friend.\n", (out.text(), err.text())
            text = err.text()
            assert "Settings: " in text and "40,960-token context (the model's own limit)" in text and "Ready in" in text and "Stopping the server..." in text and "PyTorch sampler" in text
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
            time.sleep(0.5)
            assert not alive(int(open(os.environ["FAKE_CHILD"]).read()))
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
    finally:
        run.STOP_WAIT = (5, 5, 10)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(d)
        shutil.rmtree(state)


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
    d, e = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        native = fake_native(d)
        if native is None:
            return print("test_cli_forwarding: skipped (no native program to forward to)")
        with open(os.path.join(e, "glyd"), "w") as f:  # (a script named glyd, this entry point's kind: skipped)
            f.write("#!/bin/sh\necho script\n")
        os.chmod(os.path.join(e, "glyd"), 0o755)
        assert cli.find_native(e + os.pathsep + d, me="/nowhere") == native and cli.find_native(e, me="/nowhere") is None
        assert cli.find_native(d, me=native) is None  # (itself, by its real path)
        env = {"PATH": e + os.pathsep + d, "PYTHONPATH": HERE}
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "input.tar", "-o", "input.tar.glyd"], env=env, capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "input.tar -o input.tar.glyd", (r.stdout, r.stderr)  # (forwarded as it came)
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "--help"], env=env, capture_output=True, text=True)
        assert "glyd run MODEL" in r.stdout and "--help" in r.stdout and r.returncode == 0  # (its help, then the program's)
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "--version"], env=env, capture_output=True, text=True)
        assert r.stdout.startswith("glyd 0.") and "--version" in r.stdout
        none = {"PATH": e, "PYTHONPATH": HERE}
        r = subprocess.run([sys.executable, "-m", "glyd.cli", "input.tar"], env=none, capture_output=True, text=True)
        assert r.returncode == 127 and "brew install surya-koritala/glyd/glyd" in r.stderr
        r = subprocess.run([sys.executable, "-m", "glyd.cli"], env=none, capture_output=True, text=True)
        assert "glyd doctor" in r.stdout and "brew install" in r.stdout and r.returncode == 0
        for cmd in ("doctor", "run", "serve", "login"):
            assert cmd in cli.GPU_COMMANDS
    finally:
        shutil.rmtree(d)
        shutil.rmtree(e)


# --- a fake OpenAI server: the terminal chat and the page's server side -------------------------------------------------------------

class Fake(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, window=40, reasoning=True, field="reasoning_content", port=0, start=True):
        super().__init__(("127.0.0.1", port), FakeHandler)
        self.window, self.reasoning, self.field, self.requests = window, reasoning, field, []
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

    def do_GET(self):
        if self.path == "/v1/models":
            self.send(200, {"object": "list", "data": [{"id": "fake/Model-1B", "max_model_len": self.server.window}]})
        elif self.path == "/":
            self.send(200, page.PAGE, "text/html; charset=utf-8")
        else:
            self.send(404, {"detail": "Not Found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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


def new_chat(srv, **kw):
    api = chat.Api(srv.base)
    name, window = api.model()
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

def call_asgi(app, method, path, scope_type="http"):
    sent = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request"}

    asyncio.run(app({"type": scope_type, "method": method, "path": path, "headers": []}, receive, send))
    return sent


def test_page_middleware():
    seen = []

    async def app(scope, receive, send):
        seen.append((scope["method"], scope["path"]))
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
    call_asgi(mw, "GET", "/", scope_type="websocket") if False else None


def test_page_is_self_contained():
    html = page.PAGE.decode()
    import re
    assert "<!doctype html>" in html.lower() and "</html>" in html
    urls = re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", html) + re.findall(r"url\(([^)]+)\)", html) + re.findall(r"""@import\s+["']([^"']+)""", html)
    assert all(u.startswith(("data:", "#")) for u in urls), urls  # (no script, stylesheet, font or image from anywhere)
    assert "http://" not in re.sub(r"http://www\.w3\.org/2000/svg", "", html).replace("https?:\\/\\/", "")  # (the regex for links, and the svg namespace, are the only ones)
    assert "innerHTML" not in html and "eval(" not in html and "document.write" not in html  # (model text goes in as text)
    for text in ("/v1/models", "/v1/chat/completions", "reasoning_content", "max_model_len", "New chat", "maximum context length"):
        assert text in html, text


# --- install.sh ------------------------------------------------------------------------------------------------------------------

INSTALL = os.path.join(HERE, "..", "..", "scripts", "install.sh")


def run_install(nvidia=None, uv=True, other_glyd=False, env=None, os_name=None, arch=None, compiler="gcc"):
    """scripts/install.sh with a fake uv (and curl, nvidia-smi, uname, and a `compiler` such as gcc or clang, or none): the commands it
    ran, its output, its exit status. PATH holds only those and the few programs the script uses, so this machine's own compilers and
    nvidia-smi do not answer."""
    home, bindir, tools = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        for t in ("head", "tr", "mkdir", "cp", "chmod", "cat", "sed", "sh", "dirname", "uname"):
            if shutil.which(t):
                os.symlink(shutil.which(t), os.path.join(tools, t))
        def script(name, body, d=bindir):
            with open(os.path.join(d, name), "w") as f:
                f.write("#!/bin/sh\n" + body)
            os.chmod(os.path.join(d, name), 0o755)

        log = os.path.join(home, "calls.log")
        uvbody = f'''echo "uv $*" >> "{log}"
case "$1 $2" in
  "tool dir") echo "{home}/.local/bin" ;;
  "tool install") mkdir -p "{home}/.local/bin"; printf '#!/bin/sh\\necho "glyd $*" >> "{log}"\\n' > "{home}/.local/bin/glyd"; chmod +x "{home}/.local/bin/glyd" ;;
esac
'''
        if uv:
            script("uv", uvbody)
        os.makedirs(os.path.join(home, ".local", "bin"), exist_ok=True)
        script("curl", f'echo "curl $*" >> "{log}"\nprintf \'mkdir -p "{home}/.local/bin"; cp "{bindir}/fakeuv" "{home}/.local/bin/uv"\\n\'\n')
        script("fakeuv", uvbody)
        if nvidia:
            script("nvidia-smi", f'case "$1" in -L) echo "GPU 0: NVIDIA L4";; --query-gpu=driver_version) echo "{nvidia}";; esac\n')
        else:
            script("nvidia-smi", "exit 9\n")  # (this machine's own, if it has one, must not answer)
        if os_name or arch:
            script("uname", f'case "$1" in -s) echo "{os_name or "Linux"}";; -m) echo "{arch or "x86_64"}";; esac\n')
        if compiler:
            script(compiler, "exit 0\n")
        if other_glyd:
            other = tempfile.mkdtemp()
            script("glyd", "echo compression\n", other)
        path = (other + os.pathsep if other_glyd else "") + bindir + os.pathsep + tools
        e = {"HOME": home, "PATH": path, **(env or {})}
        r = subprocess.run(["/bin/sh", INSTALL], env=e, capture_output=True, text=True)
        calls = open(log).read().splitlines() if os.path.exists(log) else []
        return r.returncode, r.stdout, r.stderr, calls
    finally:
        shutil.rmtree(home)
        shutil.rmtree(bindir)
        shutil.rmtree(tools)


def test_install_sh():
    sh_check = subprocess.run(["/bin/sh", "-n", INSTALL])
    assert sh_check.returncode == 0
    if not shutil.which("uname"):
        return
    if subprocess.run(["uname", "-s"], capture_output=True, text=True).stdout.strip() == "Darwin":
        os_name_arg = "Linux"  # (the fake uname: the Linux path on this Mac)
    else:
        os_name_arg = None
    import glyd
    V = glyd.__version__  # (the release the script pins is the tree's: scripts/bump_version.py keeps them one)
    assert f'GLYD_VERSION="${{GLYD_VERSION:-{V}}}"' in open(INSTALL).read()
    rc, out, err, calls = run_install(nvidia="595.91.07", os_name=os_name_arg)  # Linux, an NVIDIA GPU, uv present
    assert rc == 0, (out, err)
    assert calls[0] == f"uv tool install --force --managed-python --python 3.12 glyd[vllm]=={V}", calls  # (pinned: no --prerelease)
    assert "--prerelease" not in " ".join(calls) and "uv tool dir --bin" in calls and "glyd doctor" in calls[-1] and not err.strip(), (calls, err)
    rc, out, err, calls = run_install(nvidia="595.91.07", os_name=os_name_arg, env={"GLYD_VERSION": "0.26.0rc3"})  # (a pre-release is named, and only that one is taken)
    assert calls[0].endswith("glyd[vllm]==0.26.0rc3")
    rc, out, err, calls = run_install(nvidia="595.91.07", os_name=os_name_arg, env={"GLYD_SPEC": "/tmp/glyd-0.26.0rc2-py3-none-manylinux_2_28_x86_64.whl[vllm]"})
    assert calls[0].endswith("--python 3.12 /tmp/glyd-0.26.0rc2-py3-none-manylinux_2_28_x86_64.whl[vllm]")
    rc, out, err, calls = run_install(nvidia=None, os_name=os_name_arg)  # (Linux, no GPU: the compression tools alone, said so)
    assert calls[0].endswith(f"glyd=={V}") and "no NVIDIA GPU" in err and rc == 0
    rc, out, err, calls = run_install(nvidia="550.163.01", os_name=os_name_arg)  # (an older driver: a warning, before the big download)
    assert "older than 580" in err and calls[0].endswith(f"glyd[vllm]=={V}")
    rc, out, err, calls = run_install(nvidia="595.91.07", os_name=os_name_arg, uv=False)  # (uv missing: its own installer, then the tool)
    assert calls[0].startswith("curl -LsSf https://astral.sh/uv/install.sh") and any(c.startswith("uv tool install") for c in calls) and rc == 0, (calls, err)
    rc, out, err, calls = run_install(nvidia="595.91.07", os_name=os_name_arg, other_glyd=True)  # (the compression program first on PATH: said so)
    assert "another glyd" in err and "no 'run'" in err
    zig = "--managed-python --python 3.12 --with ziglang==0.16.0 "  # (no gcc or clang: a compiler from PyPI, no sudo; vLLM's Triton builds its launchers with one)
    rc, out, err, calls = run_install(nvidia="595.91.07", os_name=os_name_arg, compiler=None)
    assert calls[0] == f"uv tool install --force {zig}glyd[vllm]=={V}" and "No C compiler found" in out and rc == 0, (calls, out)
    assert "ZIGLANG=0.16.0" in open(INSTALL).read()
    for compiler, extra in (("gcc", None), ("clang", None), (None, {"CC": "/opt/cc/bin/cc"})):  # (a compiler there, or $CC set: none is added)
        rc, out, err, calls = run_install(nvidia="595.91.07", os_name=os_name_arg, compiler=compiler, env=extra)
        assert "ziglang" not in " ".join(calls) and "No C compiler" not in out, (compiler, calls)
    rc, out, err, calls = run_install(nvidia=None, os_name=os_name_arg, compiler=None)  # (no vLLM to install: no compiler needed)
    assert "ziglang" not in " ".join(calls)
    rc, out, err, calls = run_install(nvidia="595.91.07", os_name="Darwin", arch="arm64")
    assert calls[0].endswith(f"glyd=={V}") and "Linux and an NVIDIA GPU" in err
    rc, out, err, calls = run_install(os_name="FreeBSD")
    assert rc == 1 and "Linux and macOS" in err and not calls


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
