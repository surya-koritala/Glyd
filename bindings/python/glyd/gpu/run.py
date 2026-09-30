"""`glyd run MODEL`, `glyd serve MODEL`, `glyd doctor` and `glyd login`: a model on the GPU with no flags to find.

    glyd run Qwen/Qwen3-8B                   pre-flight, download, settings, a quiet server, then a chat in the terminal and at one URL
    glyd run Qwen/Qwen3-8B --prompt "Hi"     one answer on stdout, for scripts (a prompt on stdin works too)
    glyd serve Qwen/Qwen3-8B                 the same, left up as an OpenAI API
    glyd run MODEL -- --max-model-len 4096   any vLLM flag after a lone --, which wins over the settings chosen here

vLLM runs as a subprocess (`python -m vllm.entrypoints.cli.main serve ...`) with its output in a log file; the terminal shows a
status line and, if the server stops, what went wrong in plain words. preflight.py decides what is checked and chosen; chat.py is the
terminal chat; page.py the chat page the server also shows.
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of Glyd's GPU code.
"""
import argparse
import ctypes
import glob
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from . import preflight as pf
from .chat import Api, Chat
from .. import __version__

ISSUES = "https://github.com/surya-koritala/Glyd/issues"


# --- the terminal ------------------------------------------------------------------------------------------------------------

class Ui:
    """What the terminal shows while things start (stderr: stdout is the answer's): lines, a status line that updates in a terminal, a
    download bar."""

    def __init__(self, f=None):
        self.f = f or sys.stderr
        self.tty = self.f.isatty()
        self.live = False

    def _clear(self):
        if self.live:
            self.f.write("\r\x1b[K")
            self.live = False

    def line(self, text=""):
        self._clear()
        self.f.write(text + "\n")
        self.f.flush()

    def note(self, text):
        self.line("Note: " + text)

    def status(self, text):
        """A line that is replaced by the next (a terminal only; elsewhere nothing is printed until line())."""
        if self.tty:
            self.f.write("\r\x1b[K" + text[: shutil.get_terminal_size().columns - 1])
            self.f.flush()
            self.live = True


class Bar:
    """One download bar: what is done of the total, at what rate, how long to go (a line at each 10% where there is no terminal)."""

    def __init__(self, ui, title, total):
        self.ui, self.title, self.total, self.t0, self.tick, self.tenth, self.done = ui, title, max(total, 1), time.time(), [], 0, 0

    def update(self, done, final=False):
        now = time.time()
        self.done = done = min(max(done, self.done), self.total)  # (never backwards)
        self.tick = [t for t in self.tick if now - t[0] < 8] + [(now, done)]
        rate = (done - self.tick[0][1]) / (now - self.tick[0][0]) if now > self.tick[0][0] else 0
        frac = done / self.total
        if self.ui.tty:
            width = 24
            if done >= self.total and not final:
                tail = "  finishing..."
            else:
                tail = (f"  {rate / 1e6:.0f} MB/s" if rate > 1e5 else "") + (f", {fmt_time((self.total - done) / rate)} left" if rate > 1e5 and done < self.total else "")
            self.ui.status(f"{self.title}  [{'#' * int(frac * width):<{width}}] {frac * 100:3.0f}%  {pf.gb(done)} of {pf.gb(self.total)}{tail}")
        elif int(frac * 10) > self.tenth:
            self.tenth = int(frac * 10)
            self.ui.line(f"{self.title}: {self.tenth * 10}% ({pf.gb(done)} of {pf.gb(self.total)})")


def fmt_time(s):
    s = int(s)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def say_refusal(r, ui):
    ui.line(f"glyd: {r.what}.")
    for line in r.fix.splitlines():
        ui.line(f"  {line}")


# --- the download ------------------------------------------------------------------------------------------------------------

def progress_bytes(repo):
    """Bytes of a repo in the hub cache now, complete files and the blocks written of each download (for a hub that reports none itself)."""
    total = 0
    for f in glob.glob(os.path.join(pf.hub_cache(), "models--" + repo.replace("/", "--"), "blobs", "*")):
        try:
            st = os.stat(f)
            total += min(st.st_size, st.st_blocks * 512) if f.endswith(".incomplete") else st.st_size
        except OSError:
            pass
    return total


def download(m, ui):
    """The model's files in the Hub cache, with one bar. A model already there (every file, at its size) is left alone. The bar is
    fed by the byte counts huggingface_hub reports (its own bars are kept quiet); a hub that reports none is read from the cache's blocks."""
    total = sum(n for _, n in m.files)
    base = pf.cached_bytes(m.repo, m.files)
    if not m.files or base >= total:
        if m.files:
            ui.line(f"{m.name} is downloaded already.")
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import logging as hub_logging, tqdm as hub_tqdm

    hub_logging.set_verbosity_error()  # (no advice about tokens in the middle of the bar)
    bars, failed = [], []

    class Quiet(hub_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            bars.append(self)

        def display(self, *args, **kwargs):
            pass

    def work():
        try:
            snapshot_download(m.repo, allow_patterns=[p for p, _ in m.files], tqdm_class=Quiet)
        except BaseException as e:  # (raised again in the main thread)
            failed.append(e)

    def done():  # (the slowest stage that has begun: xet's network bytes lead its disk bytes by seconds)
        seen = [b.n for b in bars if getattr(b, "unit", "") == "B" and b.n]
        return base + min(seen) if seen else progress_bytes(m.repo)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    bar = Bar(ui, f"Downloading {m.name}", total)
    while worker.is_alive():
        bar.update(done())
        worker.join(0.25)
    if failed:
        e = failed[0]
        if type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError") or "401" in str(e):
            raise pf.hub_refusal(m.repo, type("E", (), {"status": 401})())
        if isinstance(e, KeyboardInterrupt):
            raise e
        raise pf.Refusal(f"the download of {m.repo} stopped ({type(e).__name__}: {e})", "Check the network connection and run the same command again: it resumes where it stopped")
    bar.update(total, final=True)
    ui.line(f"Downloaded {m.name} ({pf.gb(total)}).")


# --- the server --------------------------------------------------------------------------------------------------------------

def state_dir():
    return os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state"), "glyd")


def new_log(kind):
    """A log file for this run (the last ten are kept)."""
    d = os.path.join(state_dir(), "logs")
    os.makedirs(d, exist_ok=True)
    for old in sorted(glob.glob(os.path.join(d, "*.log")))[:-9]:
        try:
            os.remove(old)
        except OSError:
            pass
    return os.path.join(d, f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.log")


def _with_parent():
    """In the child, before exec: die with this process (Linux), so a killed glyd leaves no server holding the GPU."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


@dataclass
class Server:
    base: str  # http://127.0.0.1:8000
    log: str = ""
    proc: object = None  # None where an existing server was attached to

    def stop(self):
        """Stop the server this run started (its process group: the engine too), waiting for the GPU's memory to be let go."""
        if self.proc is None or self.proc.poll() is not None or self.proc.pid <= 1:  # (pid <= 1: never signal a group that is not ours)
            return
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(self.proc.pid, sig)
                self.proc.wait(wait)
                return
            except subprocess.TimeoutExpired:
                continue
            except (ProcessLookupError, PermissionError):
                return


def launch(m, s, host, port, extra, log, environ=None):
    """vLLM's server, started with these settings: its output appended to `log`, in a session of its own."""
    env = dict(os.environ if environ is None else environ)
    env.update(s.env)
    if not m.local:
        env["HF_HUB_OFFLINE"] = "1"  # (downloaded: vLLM does not ask the Hub again)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"] + pf.vllm_args(m, s, host, port, extra)
    with open(log, "ab") as f:
        f.write(("$ " + " ".join(f"{k}={shlex.quote(v)}" for k, v in s.env.items()) + " " + " ".join(shlex.quote(c) for c in cmd) + "\n").encode())
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                preexec_fn=_with_parent if sys.platform.startswith("linux") else None)
    return Server(f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}", log, proc)


STAGES = (("Capturing CUDA graph", "capturing CUDA graphs"), ("torch.compile", "compiling"), ("Model loading took", "warming up"),
          ("Loading safetensors checkpoint shards", "loading the weights"), ("Loading weights", "loading the weights"), ("Starting vLLM", "starting"))


def stage(log):
    """What the server's log says it is doing (its last 8 KB)."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    best = max(((tail.rfind(key), what) for key, what in STAGES), default=(-1, ""))
    return best[1] if best[0] >= 0 else ""


@dataclass
class Failure:
    kind: str  # context, free, kv, oom, port, compiler, headers, nvcc, driver, other
    what: str
    fix: str = ""
    value: object = None


def diagnose(text):
    """What stopped a server, from the tail of its log: a Failure in plain words, with (where vLLM says it) the number that fixes it."""
    m = re.search(r"estimated maximum model length is (\d+)", text)
    if m:
        return Failure("context", "the conversation cache did not fit the context chosen", "", int(m.group(1)))
    m = re.search(r"Free memory on device \S+ \(([\d.]+)/([\d.]+) GiB\) on startup is less than desired", text)
    if m:
        return Failure("free", f"another program holds GPU memory (only {float(m.group(1)) * 2**30 / 1e9:.1f} GB was free when the server started)", "Close programs that use the GPU (nvidia-smi lists them), or run a smaller model", (float(m.group(1)), float(m.group(2))))
    if "No available memory for the cache blocks" in text:
        return Failure("kv", "no GPU memory was left for the conversation cache after the model's weights", "Close programs that use the GPU, or run a smaller model")
    if re.search(r"CUDA out of memory|OutOfMemoryError", text):
        return Failure("oom", "the GPU ran out of memory while loading the model", "Close programs that use the GPU (nvidia-smi lists them), or run a smaller model")
    if re.search(r"address already in use", text, re.I):
        return Failure("port", "the port is taken by another program", "Choose another with --port")
    if "Failed to find C compiler" in text:
        return Failure("compiler", "vLLM needs a C compiler, and this machine has none", pf.compiler_hint())
    if re.search(r"Python\.h", text):
        return Failure("headers", "this Python has no C headers (Python.h)", pf.compiler_hint())
    if re.search(r"Could not find nvcc|nvcc.*not found|CUDA_HOME", text):
        return Failure("nvcc", "a CUDA compiler was asked for and is not installed", "Report this at " + ISSUES + " with the log")
    if re.search(r"driver on your system is too old|Insufficient driver|driver version is insufficient", text, re.I):
        return Failure("driver", "the NVIDIA driver is older than this PyTorch needs", "Update the driver to 580 or newer (Ubuntu: sudo ubuntu-drivers install, then reboot)")
    last = [ln for ln in text.splitlines() if re.search(r"\b\w*(Error|Exception)\b: ", ln)]
    what = re.sub(r"^\(\w+ pid=\d+\)\s*", "", last[-1]).strip()[:240] if last else "it exited without saying why"
    return Failure("other", what, "Report it at " + ISSUES + " with the log if it persists")


def log_tail(log, n=60000):
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def wait_ready(server, ui, name, detail, timeout=1800):
    """Until the server answers /v1/models; a Failure (diagnosed from its log) where it exits first. The seconds and the stage on one line."""
    api, t0 = Api(server.base), time.time()
    if not ui.tty:
        ui.line(f"Loading {name} with Glyd: {detail}...")
    while True:
        if server.proc.poll() is not None:
            time.sleep(0.5)
            return diagnose(log_tail(server.log))
        try:
            api.get("/v1/models", timeout=2)
            return None
        except Exception:
            pass
        elapsed = time.time() - t0
        if elapsed > timeout:
            server.stop()
            return Failure("other", f"the server did not come up in {fmt_time(timeout)}", "See the log")
        ui.status(f"Loading {name} with Glyd: {detail}... {fmt_time(elapsed)}" + (f" ({stage(server.log)})" if stage(server.log) else ""))
        time.sleep(0.5)


# --- starting ----------------------------------------------------------------------------------------------------------------

def split_passthrough(argv):
    """(the arguments before a lone --, the vLLM flags after it)."""
    return (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])


def parse(cmd, argv):
    p = argparse.ArgumentParser(prog=f"glyd {cmd}", allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
                                description={"run": "Download a model, start it with Glyd, and chat with it in the terminal and at a web page.",
                                             "serve": "Download a model, start it with Glyd, and leave it up as an OpenAI API (and the chat page)."}[cmd],
                                epilog="MODEL is a Hugging Face name (Qwen/Qwen3-8B) or a folder. Memory, context and parsers are chosen from your GPU; any vLLM\nflag after a lone -- is passed on and wins:  glyd %s MODEL -- --max-model-len 4096" % cmd)
    p.add_argument("model", help="a Hugging Face name such as Qwen/Qwen3-8B, or a folder")
    p.add_argument("--context", type=int, metavar="TOKENS", help="the longest conversation, in tokens (default: the most that fits the GPU)")
    p.add_argument("--port", type=int, help="the port (default 8000; glyd run takes the next free one if it is taken)")
    if cmd == "run":
        p.add_argument("-p", "--prompt", metavar="TEXT", help="answer this once on stdout and exit (a prompt on stdin works too)")
        p.add_argument("--no-think", action="store_true", help="ask the model not to think before it answers")
    else:
        p.add_argument("--host", default="127.0.0.1", help="the address to listen on (default 127.0.0.1: this computer only)")
    a = p.parse_args(argv)
    a.host = getattr(a, "host", "127.0.0.1")
    return a


def start(a, extra, mode, ui, environ=None):
    """The checks, the download, the settings and the server up: a Server, or a Refusal. `a` is the parsed arguments."""
    env = os.environ if environ is None else environ
    given = pf.flags_given(extra)
    gpus = pf.probe_gpus()
    gpu, warnings = pf.setup_checks(gpus=gpus)
    for w in warnings:
        ui.note(w)
    ui.status("Reading the model's details...")
    m = pf.load_model(a.model)
    host = str(given["host"]) if "host" in given else a.host
    port = int(pf.number(given["port"])) if "port" in given else (a.port or 8000)
    base = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    try:  # (a Glyd server for this model already up there: use it)
        name, _ = Api(base).model()
        if name == m.repo:
            ui.line(f"{m.repo} is already being served at {base}: using that server.")
            return Server(base)
    except Exception:
        pass
    settings = lambda g: pf.settings(m, g, mode, a.context, nvcc=pf.have_nvcc(env), cc=pf.have_cc(env), environ=env, given=given)
    try:
        s = settings(gpu)
    except pf.Refusal as r:
        if r.fix:
            raise
        raise pf.refusal_with_fix(m, gpu, r, mode, pf.other_users()) from None
    pf.check_disk(m)
    if "port" not in given:
        want = port
        port = pf.pick_port(host, port, a.port is not None or mode == "serve")  # (glyd run moves to a free port; glyd serve's apps expect theirs)
        if port != want:
            ui.note(f"port {want} is in use; using {port}")
    bf16 = pf.gb(m.bf16)
    ui.line(f"{m.name}: {pf.gb(s.weights)} on the GPU with Glyd, instead of {bf16}; your GPU has {pf.gb(gpu.free)} free.")
    download(m, ui)
    if len(gpus) > 1 and "CUDA_VISIBLE_DEVICES" not in env:
        s.env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    ui.line("Settings: " + pf.summary(s, gpu))
    if host not in ("127.0.0.1", "localhost", "::1"):
        ui.note(f"listening on {host}: anyone who can reach this computer can use the model. Add  -- --api-key SECRET  to require a key.")
    log = new_log(mode)
    detail = f"{pf.gb(s.weights)} instead of {bf16}"
    t0 = time.time()
    for attempt in (1, 2):
        server = launch(m, s, host, port, extra, log, env)
        failure = None
        try:
            failure = wait_ready(server, ui, m.name, detail)
        except BaseException:
            server.stop()
            raise
        if failure is None:
            ui.line(f"Ready in {fmt_time(time.time() - t0)}. (The server's log: {log})")
            return server
        server.stop()
        s2 = None
        if attempt == 1 and failure.kind == "context" and "max-model-len" not in given:
            ctx = int(failure.value) // 1024 * 1024
            if ctx >= pf.MIN_CONTEXT and (not a.context or a.context > ctx):
                s2 = replace(s, context=ctx)
                ui.line(f"vLLM has room for {int(failure.value):,} tokens of context here, not {s.context:,}: starting again with {ctx:,}.")
        elif attempt == 1 and failure.kind == "free" and "gpu-memory-utilization" not in given:
            free = int(failure.value[0] * 2**30) + int(pf.CTX)
            try:
                s2 = pf.settings(m, replace(gpu, free=free), mode, a.context, nvcc=pf.have_nvcc(env), cc=pf.have_cc(env), environ=env, given=given)
                ui.line(f"Only {pf.gb(int(failure.value[0] * 2**30))} of GPU memory was free when the server started: starting again with less ({s2.util * 100:.0f}%, {s2.context:,}-token context).")
            except pf.Refusal:
                s2 = None
        if s2 is None:
            raise pf.Refusal(f"the server stopped while starting: {failure.what}", (failure.fix + "\n" if failure.fix else "") + f"The server's log: {log}")
        s = s2
        time.sleep(2)
    raise pf.Refusal("the server stopped twice while starting", f"The server's log: {log}")


# --- the commands ------------------------------------------------------------------------------------------------------------

def cmd_run(argv):
    argv, extra = split_passthrough(argv)
    a = parse("run", argv)
    ui = Ui()
    server = start(a, extra, "run", ui)
    try:
        api = Api(server.base)
        name, window = api.model()
        chat = Chat(api, name, window, think=not a.no_think, log=server.log)
        if a.prompt is not None:
            return chat.once(a.prompt)
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
            if not prompt:
                raise pf.Refusal("no prompt: standard input is empty", "Give one with --prompt TEXT, or run glyd in a terminal to chat")
            return chat.once(prompt)
        ui.line(f"Chat here, or open {server.base} in a browser.")
        return chat.loop()
    finally:
        server.stop()


def cmd_serve(argv):
    argv, extra = split_passthrough(argv)
    a = parse("serve", argv)
    ui = Ui()
    server = start(a, extra, "serve", ui)
    name, window = Api(server.base).model()
    ui.line(f"\nServing {name} with Glyd" + (f" ({window:,}-token window)" if window else "") + f".\n  OpenAI API   {server.base}/v1   (model name: {name})\n  Chat page    {server.base}\n  Log          {server.log}\nPress Ctrl-C to stop.")
    try:
        while server.proc is not None and server.proc.poll() is None:
            time.sleep(1)
        if server.proc is None:
            return 0
        f = diagnose(log_tail(server.log))
        raise pf.Refusal(f"the server stopped: {f.what}", f"The server's log: {server.log}")
    finally:
        server.stop()


def doctor_lines(environ=None, run=pf._run):
    """[(status, label, text)] for this machine: ok, warn or fail; then the table of models that fit and the verdict."""
    env = os.environ if environ is None else environ
    rows = [("ok", "Glyd", f"{__version__}, Python {sys.version_info[0]}.{sys.version_info[1]} ({sys.executable})")]
    try:
        gpus = pf.probe_gpus(run)
    except pf.Refusal as r:
        rows.append(("fail", "GPU", f"{r.what}. {r.fix}"))
        return rows, None
    gpu = pf.pick_gpu(gpus, env.get("CUDA_VISIBLE_DEVICES", ""))
    for g in gpus:
        ok = g.cc >= pf.MIN_CAPABILITY
        rows.append(("ok" if ok else "fail", "GPU", f"{g.name}: {pf.gb(g.total)}, {pf.gb(g.free)} free, compute capability {g.cc[0]}.{g.cc[1]}" + (", a display is attached" if g.display else "")
                     + ("" if ok else ". Glyd needs Ampere or newer (RTX 30 series, A10, A100, L4, RTX 40 series, H100 and later)")))
    built = pf.torch_cuda()
    driver = f"{gpu.driver}" + (f", runs CUDA {gpu.cuda[0]}.{gpu.cuda[1]}" if gpu.cuda else "")
    try:
        warn = pf.check_driver(gpu, built)
        rows.append(("warn" if warn else "ok", "Driver", driver + (f"\n    {warn}" if warn else "")))
    except pf.Refusal as r:
        rows.append(("fail", "Driver", f"{driver}\n    {r.what}.\n    {r.fix}"))
    tv = pf.package_version("torch")
    rows.append(("ok" if tv and built else "fail", "PyTorch", f"{tv}, built for CUDA {built[0]}.{built[1]}" if tv and built else f"{tv or 'not installed'}: {'built without CUDA' if tv else 'run the installer: ' + pf.INSTALLER}"))
    vv = pf.package_version("vllm")
    if vv and pf.tested(vv):
        rows.append(("ok", "vLLM", vv))
    else:
        rows.append(("fail", "vLLM", f"{vv} is not the vLLM {pf.TESTED} that Glyd is tested with" if vv else f"not installed (Glyd is tested with vLLM {pf.TESTED}): run the installer: {pf.INSTALLER}"))
    lib = pf.gpu_library(built, env)
    rows.append(("ok" if lib else "fail", "Glyd GPU library", os.path.basename(lib) if lib else f"libglyd_gpu_cuda{built[0] if built else 'N'}.so is not in this install: run the installer again"))
    cc = shutil.which(env.get("CC") or "gcc") or shutil.which("clang")
    if cc and pf.python_headers():
        rows.append(("ok", "C compiler", f"{cc}, and Python.h (vLLM builds its Triton launchers with them)"))
    else:
        rows.append(("fail", "C compiler", ("no C compiler" if not cc else f"{cc}, but no Python.h") + ": vLLM cannot start without them. " + pf.compiler_hint()))
    nvcc = pf.have_nvcc(env)
    rows.append(("ok", "CUDA compiler", "found" if nvcc else "not found: not needed (glyd run turns FlashInfer's sampler off, which would compile)"))
    cache = pf.hub_cache()
    free = pf.disk_free(cache)
    rows.append(("ok" if free > 20e9 else "warn", "Disk", f"{pf.gb(free)} free for models, in {cache}"))
    return rows, gpu


def cmd_doctor(argv):
    argparse.ArgumentParser(prog="glyd doctor", description="Check this machine for glyd run: the GPU, driver, compilers, and which models fit.").parse_args(argv)
    rows, gpu = doctor_lines()
    mark = {"ok": " ok ", "warn": " !! ", "fail": " NO "}
    for status, label, text in rows:
        print(f"{mark[status]} {label:<17}{text}")
    ready = all(r[0] != "fail" for r in rows)
    best = None
    if gpu:
        print("\nModels that fit this GPU with Glyd (a chat as long as the GPU allows):")
        for repo in pf.COMMON:
            m = pf.ladder_model(repo)
            weights, needs = pf.footprint(m, gpu, ["mma", "mma12"] if gpu.cc != (8, 9) else ["mma"])
            try:
                s = pf.settings(m, gpu, "run")
                print(f"  {repo:<16} {pf.gb(s.weights):>8} on the GPU (bf16 {pf.gb(m.bf16)}): fits, a {s.context:,}-token context")
            except pf.Refusal:
                print(f"  {repo:<16} {pf.gb(weights):>8} on the GPU (bf16 {pf.gb(m.bf16)}): does not fit; it needs {pf.gb(needs)} free")
        for repo, cfg in pf.LADDERS["qwen"]:
            try:
                pf.settings(pf.model_of(repo, cfg), gpu, "run")
                best = repo
            except pf.Refusal:
                pass
    print()
    if not ready:
        print("Not ready: fix the lines marked NO.")
    elif best is None:
        print("This GPU has room for no model Glyd suggests.")
    else:
        print(f"Ready: glyd run {best}")
    return 0 if ready else 1


def cmd_login(argv):
    argparse.ArgumentParser(prog="glyd login", description="Save a Hugging Face token, for models that ask you to accept a licence first (Llama, Gemma).").parse_args(argv)
    try:
        from huggingface_hub import login
    except ImportError:
        raise pf.Refusal("huggingface_hub is not installed", "Run the installer: " + pf.INSTALLER) from None
    print("Open https://huggingface.co/settings/tokens, create a token with read access, and paste it here (nothing shows as you type).")
    login(add_to_git_credential=False)
    return 0


def main(cmd, argv):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # (a killed glyd stops its server: Server.stop runs in finally)
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(129))
    ui = Ui()
    try:
        return {"run": cmd_run, "serve": cmd_serve, "doctor": cmd_doctor, "login": cmd_login}[cmd](argv)
    except pf.Refusal as r:
        say_refusal(r, ui)
        return 1
    except KeyboardInterrupt:
        ui.line("\nStopped.")
        return 130
    except SystemExit:
        raise
    except Exception as e:
        if os.environ.get("GLYD_DEBUG"):
            raise
        ui.line(f"glyd: unexpected error: {type(e).__name__}: {e}")
        ui.line(f"  Set GLYD_DEBUG=1 for the traceback, and report it at {ISSUES}")
        return 1
