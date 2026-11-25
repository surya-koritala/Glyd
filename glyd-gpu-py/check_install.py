#!/usr/bin/env python3
"""What an installed glyd-gpu wheel has, looked at from outside the source tree (run it as python -I, so that no checkout is on the path):
its version; that importing it pulls in no torch; both CUDA libraries load without a GPU (their CUDA runtime is linked in: the driver is
looked for at the first call) and answer the C API version of the header the wheel ships (the package's calls are written for it:
test_gpu.py's header tests tie the two; a stale library does not) and export the functions the package calls by their names; the engine's program (glyd-engine) is executable, is this
version, finds the library beside it and says which models it runs (with no GPU, and no network); vLLM's entry point is registered.

    python -I check_install.py VERSION [--no-libs]        (--no-libs: a wheel built without the CUDA libraries)"""
import ctypes
import ctypes.util
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from importlib.metadata import entry_points, version


want = sys.argv[1]
import glyd_gpu  # noqa: E402

d = os.path.dirname(glyd_gpu.__file__)
assert "site-packages" in d, f"{d}: not an installed package"
assert version("glyd-gpu") == want == glyd_gpu.__version__, (version("glyd-gpu"), want, glyd_gpu.__version__)
assert "torch" not in sys.modules, "import glyd_gpu pulled in torch"
assert {e.name: e.value for e in entry_points(group="vllm.general_plugins")}.get("glyd") == "glyd_gpu.vllm_entry:register"
libs = sorted(glob.glob(d + "/libglyd_gpu_cuda*.so"))
if "--no-libs" in sys.argv:
    assert not libs, libs
else:
    assert len(libs) == 2, libs
    api = int(re.search(r"#define GLYD_GPU_API_VERSION (\d+)", open(d + "/glyd_gpu.h").read()).group(1))
    for f in libs:
        lib = ctypes.CDLL(f)
        print(os.path.basename(f), os.path.getsize(f), "bytes, CUDA runtime", lib.glyd_gpu_cuda_version(), "C API", lib.glyd_gpu_api_version())
        assert lib.glyd_gpu_api_version() == api, f"{f}: its C API is not the header's ({api})"
        assert hasattr(lib, "glyd_gpu_cuda_version") and hasattr(lib, "glyd_gpu_mma_gemm"), f"{f}: a function of the C API is not exported by its name"
eng = os.path.join(d, "glyd-engine")
if "--no-libs" in sys.argv:
    assert not os.path.exists(eng), "an engine's program in a wheel built without the libraries"
else:
    assert os.path.isfile(eng) and os.access(eng, os.X_OK), f"{eng} is not an executable program"
    env = dict(os.environ, HF_HUB_OFFLINE="1")  # (the engine looks at nothing on the network)
    said = subprocess.run([eng, "--version"], capture_output=True, text=True, env=env).stdout.strip()
    assert said == f"glyd-engine {want}", said
    doctor = subprocess.run([eng, "doctor"], capture_output=True, text=True, env=env).stdout
    assert re.search(r"GPU library\s+" + re.escape(d) + r"/libglyd_gpu_cuda1[23]\.so\s", doctor + "\n"), doctor  # (the library beside it, of the C API it is written for)
    with tempfile.TemporaryDirectory() as m:
        for arch, code in (("Qwen3ForCausalLM", 0), ("LlamaForCausalLM", 3)):
            with open(os.path.join(m, "config.json"), "w") as f:
                json.dump({"architectures": [arch], "rope_scaling": None}, f)
            got = subprocess.run([eng, "supports", m], capture_output=True, text=True, env=env).returncode
            assert got == code, (arch, got)
        # on a computer with no NVIDIA driver (this one, where there is none): the engine's check says no, in words, before it looks at a model's files, with status 3 (the vLLM way); `run` and `serve` leave
        # the reason in the status file the program that starts them names (one line, whatever its status) and say nothing on the terminal, and standard input is as it was: nothing is read before they are up
        if not ctypes.util.find_library("cuda"):
            with open(os.path.join(m, "config.json"), "w") as f:
                json.dump({"architectures": ["Qwen3ForCausalLM"], "rope_scaling": None, "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                           "intermediate_size": 3072, "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "max_position_embeddings": 40960}, f)
            r = subprocess.run([eng, "preflight", m], capture_output=True, text=True, env=env)
            assert r.returncode == 3 and "no NVIDIA driver" in r.stderr and r.stdout == "", (r.returncode, r.stdout, r.stderr)
            why = os.path.join(m, "why")
            r = subprocess.run([eng, "run", "--status-file", why, m, "--prompt", "hi"], capture_output=True, text=True, env=env)
            assert r.returncode == 3 and r.stderr == "" and open(why).read() == "no NVIDIA driver was found on this computer\n", (r.returncode, r.stderr)
            r = subprocess.run([eng, "serve", "--status-file", why, m, "--users", "99"], capture_output=True, text=True, env=env)
            # (the engine's limit is its own to change: the line says 1 to N, N a number)
            assert r.returncode == 1 and r.stderr == "" and re.fullmatch(r"--users 99: 1 to \d+ sequences at once\n", open(why).read()), (r.returncode, r.stderr, open(why).read())
            r = subprocess.run(["sh", "-c", '"$0" run --status-file "$1" "$2"; cat', eng, why, m], input="Say hello\n", capture_output=True, text=True, env=env)  # (a pipe: the engine stops at the driver and leaves it to `cat`)
            assert r.stdout == "Say hello\n" and open(why).read() == "no NVIDIA driver was found on this computer\n", (r.stdout, r.stderr)
            r = subprocess.run([eng, "run", "--status-file", why, m], stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env)  # (a standard input known to be empty is refused at once)
            assert r.returncode == 1 and r.stderr == "" and open(why).read() == "no prompt: standard input is empty\n", (r.returncode, r.stderr)
    from glyd_gpu import engine  # (the hand-off's own reading of where the program is, in the compiled module: beside the library, not on any PATH)

    assert engine.find() == eng and engine.model_of(["Qwen/Qwen3.8-27B", "--prompt", "hi"]) == "Qwen/Qwen3.8-27B", engine.find()
    print("glyd-engine", said.split()[1], "runs here:", os.path.getsize(eng), "bytes")
print("glyd-gpu", want, "installed in", d)
