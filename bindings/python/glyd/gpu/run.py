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
import errno
import glob
import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from . import preflight as pf
from .chat import Api, ApiError, Chat, clean
from .. import __version__

ISSUES = "https://github.com/surya-koritala/Glyd/issues"
STOP_API_WAIT = 30  # seconds the API server gets to stop itself, and its engine, after SIGTERM
STOP_LOADING_WAIT = 5  # ... where the server is still loading its model: the engine does not answer a SIGTERM until the load is over (50 s on an L4 with Qwen3-1.7B at 30 s), so the group is swept sooner
STOP_WAIT = (5, 5, 10)  # then: seconds the engine gets to follow it out, to obey SIGTERM, and to die of SIGKILL, when a server is stopped
LOOPBACK = ("127.0.0.1", "localhost", "::1")
SECRET_ENV = ("VLLM_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")  # (never written to the server's log)


# --- the terminal ------------------------------------------------------------------------------------------------------------

class Ui:
    """What the terminal shows while things start (stderr: stdout is the answer's): lines, a status line that updates in a terminal, a
    download bar. Text goes out cleaned of control characters (a log line quoted in a refusal, a server's error), and a terminal that has
    gone away (EIO after a hangup, a closed pipe) is not an error here."""

    def __init__(self, f=None):
        self.f = f or sys.stderr
        self.tty = self.f.isatty() and os.environ.get("TERM") != "dumb"  # (a dumb terminal does not do \r and ESC [ K)
        self.live = False

    def _write(self, text):
        try:
            self.f.write(text)
            self.f.flush()
        except (OSError, ValueError):
            pass

    def _clear(self):
        if self.live:
            self._write("\r\x1b[K")
            self.live = False

    def line(self, text=""):
        self._clear()
        self._write(clean(text) + "\n")

    def note(self, text):
        self.line("Note: " + text)

    def status(self, text):
        """A line that is replaced by the next (a terminal only; elsewhere nothing is printed until line())."""
        if self.tty:
            self._write("\r\x1b[K" + clean(text)[: shutil.get_terminal_size().columns - 1])
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
            self.glyd_name = kwargs.get("name") or ""
            kwargs["disable"] = False  # (the hub's bars are off where stderr is not a terminal, and would count nothing)
            kwargs["leave"] = False  # (a closed tqdm bar writes a newline, which left the last drawn bar behind as a line of its own)
            super().__init__(*args, **kwargs)
            bars.append(self)

        def display(self, *args, **kwargs):
            pass

    def work():
        try:
            snapshot_download(m.repo, allow_patterns=[p for p, _ in m.files], tqdm_class=Quiet)
        except BaseException as e:  # (raised again in the main thread)
            failed.append(e)

    def done():  # (the bytes of the files: xet also reports its network bytes, which count what it had to fetch, so not those)
        seen = [b for b in bars if getattr(b, "unit", "") == "B" and b.n]
        whole = [b.n for b in seen if not b.glyd_name.endswith("transfer")] or [b.n for b in seen]
        return base + max(whole) if whole else progress_bytes(m.repo)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    bar = Bar(ui, f"Downloading {m.name}", total)
    try:
        while worker.is_alive():
            bar.update(done())
            worker.join(0.25)
    except KeyboardInterrupt:
        raise KeyboardInterrupt("download") from None  # (main() says that the download resumes; the hub's threads are not ours to wait for)
    if failed:
        e = failed[0]
        if type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError") or "401" in str(e):
            raise pf.hub_refusal(m.repo, type("E", (), {"status": 401})())
        if isinstance(e, KeyboardInterrupt):
            raise e
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            raise pf.Refusal(f"the disk holding {pf.hub_cache()} is full ({pf.gb(pf.disk_free(pf.hub_cache()))} left)",
                             f"Free some space ({pf.gb(max(total - base, 0))} to download), or keep the models on another drive: HF_HOME=/path/on/that/drive glyd run {m.repo}")
        raise pf.Refusal(f"the download of {m.repo} stopped ({type(e).__name__}: {clean(str(e))})", "Check the network connection and run the same command again: it resumes where it stopped")
    bar.update(total, final=True)
    ui.line(f"Downloaded {m.name} ({pf.gb(total)}).")


# --- the server --------------------------------------------------------------------------------------------------------------

def state_dir():
    return os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state"), "glyd")


def new_log(kind):
    """A log file for this run (the last ten, by time, are kept; the folder and its files are the user's alone)."""
    d = os.path.join(state_dir(), "logs")
    os.makedirs(d, mode=0o700, exist_ok=True)
    try:
        os.chmod(d, 0o700)  # (a folder an earlier version made with the user's umask)
    except OSError:
        pass
    for old in sorted(glob.glob(os.path.join(d, "*.log")), key=os.path.getmtime)[:-9]:
        try:
            os.remove(old)
        except OSError:
            pass
    return os.path.join(d, f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.log")


def marker_path(port):
    return os.path.join(state_dir(), "serving", f"{port}.json")


def write_marker(port, model, base):
    """Record that this user's `glyd serve` is up on a port: only such a server is one `glyd run` chats with (another user can bind the
    port first and answer to the same model name)."""
    path = marker_path(port)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"pid": os.getpid(), "model": model, "base": base}, f)


def read_marker(port):
    """The record of a running `glyd serve` of this user on the port, or None."""
    try:
        with open(marker_path(port)) as f:
            d = json.load(f)
        os.kill(int(d["pid"]), 0)
        return d
    except (OSError, ValueError, KeyError, TypeError):
        return None


def drop_marker(port):
    try:
        os.remove(marker_path(port))
    except OSError:
        pass


def _with_parent():
    """In the child, before exec: die with this process (Linux), so a killed glyd leaves no server holding the GPU."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def base_url(host, port):
    """The address glyd's own connections use: 127.0.0.1 where the server listens on every interface; an IPv6 address in brackets."""
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return f"http://[{host}]:{port}" if ":" in host and not host.startswith("[") else f"http://{host}:{port}"


@dataclass
class Server:
    base: str  # http://127.0.0.1:8000
    log: str = ""
    proc: object = None  # None where an existing server was attached to
    log_start: int = 0  # where this start's lines begin in the log (a retry appends to it)
    model: str = ""
    token: str = ""  # the API key the server asks for on /v1, if any
    ready: bool = False  # it has answered /health: it is stopped by its own graceful stop, not by the sweep that a server in the middle of loading needs

    @property
    def url(self):
        """The address as a person types it: localhost where the server is on this computer only."""
        return self.base.replace("//127.0.0.1:", "//localhost:")

    def stop(self, ui=None):
        """Stop the server this run started and wait for its GPU memory to be let go. The API server gets SIGTERM alone: it stops the engine
        itself (a SIGTERM to the engine as well made the API server log the dead engine as an error, with a traceback). What is left of
        the process group after that is terminated, then killed. A Ctrl-C meanwhile is ignored: it would skip the sweep, and the engine
        would keep the GPU's memory."""
        if self.proc is None or self.proc.pid <= 1:  # (pid <= 1: never signal a group that is not ours)
            return
        pgid = self.proc.pid
        try:
            before = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):  # (not the main thread)
            before = None
        try:
            if self.proc.poll() is None:
                if ui is not None:
                    ui.line("Stopping the server (a few seconds)..." if self.ready else "Stopping the server (it is still loading: up to half a minute)...")
                try:
                    os.kill(pgid, signal.SIGTERM)
                    self.proc.wait(STOP_API_WAIT if self.ready else STOP_LOADING_WAIT)
                except subprocess.TimeoutExpired:
                    pass
                except ProcessLookupError:
                    pass
            for sig, seconds in zip((None, signal.SIGTERM, signal.SIGKILL), STOP_WAIT):
                if sig is not None:
                    try:
                        os.killpg(pgid, sig)
                    except (ProcessLookupError, PermissionError):
                        break
                end = time.time() + seconds
                while time.time() < end:
                    try:
                        os.killpg(pgid, 0)
                    except (ProcessLookupError, PermissionError):
                        sig = "gone"
                        break
                    time.sleep(0.1)
                if sig == "gone":
                    break
            try:
                self.proc.wait(1)
            except subprocess.TimeoutExpired:
                pass
        finally:
            if before is not None:
                signal.signal(signal.SIGINT, before)


ZIGCC = """#!/bin/sh
exec {python} -c 'import os, sys
a = sys.argv[1:]
d = [x[2:] for x in a if x.startswith("-L")]
o = [next((os.path.join(p, x[3:]) for p in d if os.path.exists(os.path.join(p, x[3:]))), x) if x.startswith("-l:") else x for x in a]
os.execv(sys.executable, [sys.executable, "-m", "ziglang", "cc", "-w"] + o)' "$@"
"""


def zig_cc():
    """A C compiler for vLLM's Triton on a machine that has none: a `cc` script in the state directory that runs ziglang's (installed
    by install.sh where there is no gcc or clang). Triton links with `-l:libcuda.so.1`, which zig's linker does not take, so the script
    gives that one as the library's path, found in the -L directories, and -w keeps clang's warnings about Python's own headers out of
    the server's log. Written whole and put in place in one step: a second `glyd run` never executes it half written, and a link left
    at its name is replaced, not written through."""
    d = state_dir()
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".zigcc-")
    with os.fdopen(fd, "w") as f:
        f.write(ZIGCC.format(python=shlex.quote(sys.executable)))
    os.chmod(tmp, 0o755)
    path = os.path.join(d, "zigcc")
    os.replace(tmp, path)
    return path


def launch(m, s, host, port, extra, log, environ=None):
    """vLLM's server, started with these settings: its output appended to `log` (readable by this user alone), in a session of its own."""
    env = dict(os.environ if environ is None else environ)
    env.update(s.env)
    env.setdefault("HF_HUB_OFFLINE", "1")  # (downloaded, or cached, or a folder: vLLM does not ask the Hub again, and does not wait on a network that is not there)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"] + pf.vllm_args(m, s, host, port, extra)
    try:
        start = os.path.getsize(log)
    except OSError:
        start = 0
    shown = " ".join(f"{k}={'***' if k in SECRET_ENV else shlex.quote(v)}" for k, v in s.env.items())
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "ab") as f:
        f.write(("$ " + shown + " " + " ".join(shlex.quote(c) for c in cmd) + "\n").encode())
        f.flush()
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                preexec_fn=_with_parent if sys.platform.startswith("linux") else None)
    return Server(base_url(host, port), log, proc, start, m.repo)


STAGES = (("Capturing CUDA graph", "capturing CUDA graphs"), ("Dynamo bytecode transform", "compiling"), ("Model loading took", "warming up"),
          ("Loading safetensors checkpoint shards", "loading the weights"), ("Loading weights", "loading the weights"),
          ("Initializing a V1 LLM engine", "starting the engine"), ("Resolved architecture", "reading the model's details"), ("Starting vLLM", "starting"))


def stage(log):
    """What the server's log says it is doing (its last 8 KB): the stage, and for the weights the shard it is at."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    best = max(((tail.rfind(key), what) for key, what in STAGES), default=(-1, ""))
    if best[0] < 0:
        return ""
    if best[1] == "loading the weights":
        shards = re.findall(r"Loading safetensors checkpoint shards:\s+\d+% Completed \| (\d+)/(\d+)", tail)
        if shards:
            return f"loading the weights, {shards[-1][0]} of {shards[-1][1]} parts"
    return best[1]


@dataclass
class Failure:
    kind: str  # context, free, args, kv, oom, port, compiler, headers, nvcc, driver, window, arch, killed, other
    what: str
    fix: str = ""
    value: object = None


def diagnose(text, rc=None):
    """What stopped a server, from its log (the lines of this start): a Failure in plain words, with (where vLLM says it) the number that
    fixes it. `rc` is the server's exit status."""
    m = re.search(r"estimated maximum model length is (\d+)", text)
    if m:
        return Failure("context", "the conversation cache did not fit the context chosen", "", int(m.group(1)))
    m = re.search(r"Free memory on device \S+ \(([\d.]+)/([\d.]+) GiB\) on startup is less than desired", text)
    if m:
        return Failure("free", f"another program holds GPU memory (only {float(m.group(1)) * 2**30 / 1e9:.1f} GB was free when the server started)", "Close programs that use the GPU (nvidia-smi lists them), or run a smaller model", (float(m.group(1)), float(m.group(2))))
    m = re.search(r"\b(?:main\.py|vllm(?: serve)?): error: (.+)", text)
    if m:  # (a flag after -- that vLLM does not take)
        return Failure("args", f"vLLM did not accept the flags: {m.group(1).strip()[:200]}", "Check what follows the lone -- (vllm serve --help lists the flags), or leave it out")
    m = re.search(r"User-specified max_model_len \((\d+)\) is greater than the derived max_model_len \([^)]*?=(\d+)", text)
    if m:
        return Failure("window", f"the context asked for ({int(m.group(1)):,} tokens) is longer than the model's own window ({int(m.group(2)):,})", f"Use --context {int(m.group(2)):,} or less")
    m = re.search(r"Model architectures \[([^\]]*)\] (?:are not supported|failed to be inspected)", text)
    if m:
        return Failure("arch", f"vLLM cannot run this model's architecture ({m.group(1).replace(chr(39), '')[:80]})", "Glyd runs the models vLLM runs: https://docs.vllm.ai/en/latest/models/supported_models.html lists them")
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
        return Failure("driver", "the NVIDIA driver is older than this PyTorch needs", "Update the driver to 580 or newer (Ubuntu: " + pf.DRIVER_PACKAGE.format(want="580") + ")")
    last = [ln for ln in text.splitlines() if re.search(r"\b\w*(Error|Exception)\b: ", ln)]
    if rc == -9 and not last:
        return Failure("killed", "the system killed the server (exit status -9), which usually means this computer ran out of memory (RAM, not the GPU's)",
                       "Close other programs, or run a smaller model. As an administrator, dmesg | grep -i 'killed process' shows what was killed")
    what = re.sub(r"^\(\w+ pid=\d+\)\s*", "", last[-1]).strip()[:240] if last else "it exited without saying why"
    return Failure("other", what, "Report it at " + ISSUES + " with the log if it persists")


def log_tail(log, n=60000, start=0):
    """The last n bytes of a log, but not before `start` (the lines of an earlier start are not this one's)."""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(start, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def wait_ready(server, ui, name, detail, timeout=1800):
    """Until the server answers /health (open whether or not it has an API key; /v1/models is not); a Failure (diagnosed from the lines of
    its log that this start wrote) where it exits first. The seconds and the stage on one line."""
    api, t0 = Api(server.base), time.time()
    if not ui.tty:
        ui.line(f"Loading {name} with Glyd: {detail}...")
    while True:
        if server.proc.poll() is not None:
            time.sleep(0.5)
            return diagnose(log_tail(server.log, start=server.log_start), server.proc.returncode)
        try:
            if api.status("/health", timeout=2) == 200:
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


def port_arg(v):
    try:
        return pf.parse_port(v)
    except pf.Refusal as r:
        raise argparse.ArgumentTypeError(r.what + ". " + r.fix) from None


def parse(cmd, argv):
    p = argparse.ArgumentParser(prog=f"glyd {cmd}", allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
                                description={"run": "Download a model, start it with Glyd, and chat with it in the terminal and at a web page.",
                                             "serve": "Download a model, start it with Glyd, and leave it up as an OpenAI API (and the chat page)."}[cmd],
                                epilog="MODEL is a Hugging Face name (Qwen/Qwen3-8B) or a folder. Memory, context and parsers are chosen from your GPU; any vLLM\nflag after a lone -- is passed on and wins:  glyd %s MODEL -- --max-model-len 4096" % cmd)
    p.add_argument("model", help="a Hugging Face name such as Qwen/Qwen3-8B, or a folder")
    p.add_argument("--context", type=int, metavar="TOKENS", help="the longest conversation, in tokens (default: the most that fits the GPU)")
    p.add_argument("--port", type=port_arg, help="the port (default 8000; glyd run takes the next free one if it is taken)")
    if cmd == "run":
        p.add_argument("-p", "--prompt", metavar="TEXT", help="answer this once on stdout and exit (a prompt on stdin works too)")
        p.add_argument("--no-think", action="store_true", help="ask the model not to think before it answers")
    else:
        p.add_argument("--host", default=None, help="the address to listen on (default 127.0.0.1: this computer only, and only its own page and programs). "
                       "Another address opens the server to the network: set an API key with VLLM_API_KEY=KEY glyd serve ...")
    a = p.parse_args(argv)
    a.host = getattr(a, "host", None)  # (None: not given)
    return a


def open_note(host, key):
    """What a server on an address other than this computer's own leaves open, said where it is started."""
    note = (f"listening on {host}: anyone who can reach this computer's network address can send the model prompts, and the traffic is plain HTTP, not encrypted "
            "(use a VPN or an SSH tunnel, or add  -- --ssl-keyfile FILE --ssl-certfile FILE ).")
    open_paths = "The chat page, /health, /metrics, /version, /tokenize, /detokenize and the API's /docs stay open without a key."
    if key:
        return note + " The API key guards /v1 only. " + open_paths
    return note + " There is no API key: start it with  VLLM_API_KEY=KEY glyd serve ...  (the key goes in the environment, not on the command line, where ps and a log show it). " + open_paths


def start(a, extra, mode, ui, environ=None):
    """The checks, the download, the settings and the server up: a Server, or a Refusal. `a` is the parsed arguments."""
    env = os.environ if environ is None else environ
    typed = list(extra)
    key, extra = pf.take_api_key(extra, env)  # (a key given as --api-key goes to the server in VLLM_API_KEY: not on its command line or in its log)
    if extra != typed:
        ui.note("the API key was taken off the server's command line and goes to it in VLLM_API_KEY. It is still in glyd's own command line, "
                "which ps shows: next time set VLLM_API_KEY=KEY yourself.")
    given = pf.flags_given(extra)
    gpus = pf.probe_gpus()
    gpu, warnings = pf.setup_checks(gpus=gpus)
    for w in warnings:
        ui.note(w)
    ui.status("Reading the model's details...")
    m = pf.load_model(a.model)
    pf.check_dtype(m, given)
    host = str(given["host"]) if "host" in given else (a.host or "127.0.0.1")
    port = pf.parse_port(given["port"]) if "port" in given else (a.port or 8000)
    strict = "host" not in given and not a.host  # (the address is glyd's own choice: this computer, its own page and programs only)
    base = base_url(host, port)
    served = read_marker(port)
    if served and served.get("model") == m.repo:  # (this user's own glyd serve of this model, up there: use it)
        try:
            name, _ = Api(base, token=key).model()
            if name == m.repo:
                ui.line(f"{m.repo} is already being served at {base}: using that server.")
                return Server(base, model=m.repo, token=key or "")
        except ApiError as e:
            if e.status == 401:
                raise pf.Refusal(f"the glyd server for {m.repo} at {base} asks for an API key", "Give it the key: VLLM_API_KEY=KEY glyd run ...") from None
        except Exception:
            pass
    zig = zig_cc() if "CC" not in env and not pf.have_cc(env) and pf.have_zig() else ""

    def settings(g):  # (every Settings, the retry's included: the GPU it was sized for, and the guard)
        s = pf.settings(m, g, mode, a.context, nvcc=pf.have_nvcc(env), environ=env, given=given, cc=zig)
        if len(gpus) > 1 and "CUDA_VISIBLE_DEVICES" not in env:  # (nvidia-smi's index is the PCI bus order; CUDA's default is the fastest first)
            s.env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
            if "CUDA_DEVICE_ORDER" not in env:
                s.env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if strict:
            s.local_only = True
            s.env["GLYD_LOCAL_ONLY"] = "1"
        return s

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
            base = base_url(host, port)
    elif not pf.port_free(host, port):
        raise pf.Refusal(f"port {port} is in use by another program", f"Choose another: -- --port {port + 1}")
    bf16 = pf.gb(m.bf16)
    fits_bf16 = pf.need(m.bf16, m, gpu) <= gpu.free
    ui.line(f"{m.name}: {pf.gb(s.weights)} on the GPU with Glyd, instead of {bf16}" + (f"; your GPU has {pf.gb(gpu.free)} free." if fits_bf16 else f", which does not fit the {pf.gb(gpu.free)} your GPU has free."))
    download(m, ui)
    ui.line("Settings: " + pf.summary(s, gpu))
    if host not in LOOPBACK:
        ui.note(open_note(host, key))
    log = new_log(mode)
    detail = f"{pf.gb(s.weights)} instead of {bf16}"
    child = dict(env)
    if key:
        child["VLLM_API_KEY"] = key
    t0 = time.time()
    for attempt in (1, 2):
        server = launch(m, s, host, port, extra, log, child)
        server.token = key or ""
        failure = None
        try:
            failure = wait_ready(server, ui, m.name, detail)
        except BaseException:
            server.stop(ui)
            raise
        if failure is None:
            server.ready = True
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
                s2 = settings(replace(gpu, free=free))
                ui.line(f"Only {pf.gb(int(failure.value[0] * 2**30))} of GPU memory was free when the server started: starting again with less ({s2.util * 100:.0f}%, {s2.context:,}-token context).")
            except pf.Refusal:
                s2 = None
        if s2 is None:
            raise stopped(failure, m, gpu, mode, log)
        s = s2
        time.sleep(2)
    raise pf.Refusal("the server stopped twice while starting", f"The server's log: {log}")


def stopped(failure, m, gpu, mode, log):
    """The Refusal for a server that stopped while starting: what it was, and the next step (where memory is the matter: what holds the GPU's
    memory, and a smaller model)."""
    fixes = [failure.fix or "Close programs that use the GPU (nvidia-smi lists them), or run a smaller model"] if failure.fix or failure.kind in ("context", "free", "kv", "oom") else []
    if failure.kind in ("context", "free", "kv", "oom"):
        fixes += pf.memory_fixes(m, gpu, mode, pf.other_users())
    return pf.Refusal(f"the server stopped while starting: {failure.what}", "\n".join(fixes + [f"The server's log: {log}"]))


# --- the commands ------------------------------------------------------------------------------------------------------------

def stdin_now():
    """Standard input read now where it can be (it holds data, or is at its end): None where it is a pipe that has nothing in it yet."""
    try:
        ready = select.select([sys.stdin], [], [], 0)[0]
    except (OSError, ValueError, TypeError):
        return None
    return sys.stdin.read() if ready else None


def cmd_run(argv):
    argv, extra = split_passthrough(argv)
    a = parse("run", argv)
    ui = Ui()
    prompt = a.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = stdin_now()  # (before the download and the load: an empty one is found now, not in minutes; a pipe with nothing in it yet is read after)
        if prompt is not None and not prompt.strip():
            raise pf.Refusal("no prompt: standard input is empty", "Give one with --prompt TEXT, or run glyd in a terminal to chat")
    elif prompt is not None and not prompt.strip():
        raise pf.Refusal("the prompt is empty", "Give one: --prompt \"Say hello\"")
    server = start(a, extra, "run", ui)
    try:
        api = Api(server.base, token=server.token or None)
        try:
            name, window = api.model()
        except ApiError as e:
            if e.status == 401:
                raise pf.Refusal(f"the server at {server.base} asks for an API key", "Give it the key: VLLM_API_KEY=KEY glyd run ...") from None
            raise
        chat = Chat(api, name, window, think=not a.no_think, log=server.log)
        if prompt is None and not sys.stdin.isatty():
            prompt = sys.stdin.read()
            if not prompt.strip():
                raise pf.Refusal("no prompt: standard input is empty", "Give one with --prompt TEXT, or run glyd in a terminal to chat")
        if prompt is not None:
            return chat.once(prompt.strip())
        ui.line(f"Chat here, or open {server.url} in a browser.")
        return chat.loop()
    finally:
        server.stop(ui)


def cmd_serve(argv):
    argv, extra = split_passthrough(argv)
    a = parse("serve", argv)
    ui = Ui()
    server = start(a, extra, "serve", ui)
    if server.proc is None:  # (this user's glyd serve of the model was up already: nothing to start, and nothing to wait for)
        ui.line(f"Nothing to start: {server.url} is up already.")
        return 0
    port = int(server.base.rsplit(":", 1)[1])
    write_marker(port, server.model, server.base)
    try:
        name, window = Api(server.base, token=server.token or None).model()
        ui.line(f"\nServing {clean(name)} with Glyd" + (f" ({window:,}-token window)" if window else "") + f".\n  OpenAI API   {server.url}/v1   (model name: {clean(name)})"
                + ("   (asks for the API key)" if server.token else "") + f"\n  Chat page    {server.url}\n  Log          {server.log}\nPress Ctrl-C to stop.")
        while server.proc.poll() is None:
            time.sleep(1)
        f = diagnose(log_tail(server.log, start=server.log_start), server.proc.returncode)
        raise pf.Refusal(f"the server stopped: {f.what}", f"The server's log: {server.log}")
    finally:
        drop_marker(port)
        server.stop(ui)


def doctor_lines(environ=None, run=pf._run, platform=None, mine=None):
    """[(status, label, text)] for this machine: ok, warn, fail or info; then the table of models that fit and the verdict."""
    env = os.environ if environ is None else environ
    rows = [("ok", "Glyd", f"{__version__}, Python {sys.version_info[0]}.{sys.version_info[1]} ({sys.executable})")]
    shadow = pf.path_shadow(env, mine)
    if shadow:
        first, own = shadow
        rows.append(("warn", "glyd on PATH", f"another program named glyd comes first on your PATH ({first}): the compression program, which takes `run` for a file name.\n"
                     f"    Run this one as {own}, or put its folder first: export PATH=\"{os.path.dirname(own)}:$PATH\" (and add that line to your shell's startup file)"))
    try:
        gpus = pf.probe_gpus(run, platform)
    except pf.Refusal as r:
        rows.append(("info", "GPU", f"{r.what}.\n    {r.fix}".replace("\n", "\n    ") if r.fix else f"{r.what}."))
        return rows, None
    gpu = pf.pick_gpu(gpus, env.get("CUDA_VISIBLE_DEVICES", ""))
    for g in gpus:
        ok = g.cc >= pf.MIN_CAPABILITY and not g.mig
        rows.append(("ok" if ok else "fail", "GPU", f"{g.name}: {pf.gb(g.total)}, {pf.gb(g.free)} free, compute capability {g.cc[0]}.{g.cc[1]}" + (", a display is attached" if g.display else "")
                     + ("" if g.cc >= pf.MIN_CAPABILITY else ". Glyd needs Ampere or newer (RTX 30 series, A10, A100, L4, RTX 40 series, H100 and later)")
                     + (". MIG is on: CUDA sees one slice, and glyd does not size a slice; use the whole GPU, or set the memory by hand with -- --gpu-memory-utilization" if g.mig else "")))
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
    if (cc or pf.have_zig()) and pf.python_headers():
        rows.append(("ok", "C compiler", f"{cc}, and Python.h (vLLM builds its Triton launchers with them)" if cc else "ziglang (a C compiler from PyPI), and Python.h: vLLM builds its Triton launchers with them"))
    else:
        rows.append(("fail", "C compiler", ("no C compiler" if not (cc or pf.have_zig()) else f"{cc or 'ziglang'}, but no Python.h") + ": vLLM cannot start without them. " + pf.compiler_hint()
                     + (". Or run the installer again, which adds a compiler from PyPI where there is none" if not cc else "")))
    nvcc = pf.have_nvcc(env)
    rows.append(("ok", "CUDA compiler", "found" if nvcc else "not found: not needed (glyd run turns FlashInfer's sampler off, which would compile)"))
    if gpu.free < 0.8 * gpu.total:
        users = pf.other_users(run)
        rows.append(("warn", "GPU in use", (", ".join(f"{n} ({pf.gb(b)})" for n, b in users[:4]) + ": " if users else "") + f"{pf.gb(gpu.total - gpu.free)} of the GPU's {pf.gb(gpu.total)} is held by other programs, and is not free for a model"))
    cache = pf.hub_cache()
    free = pf.disk_free(cache)
    rows.append(("ok" if free > 20e9 else "warn", "Disk", f"{pf.gb(free)} free for models, in {cache}"))
    return rows, gpu


def cmd_doctor(argv):
    argparse.ArgumentParser(prog="glyd doctor", description="Check this machine for glyd run: the GPU, driver, compilers, and which models fit.").parse_args(argv)
    rows, gpu = doctor_lines()
    mark = {"ok": " ok ", "warn": " !! ", "fail": " NO ", "info": " -- "}
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
                try:  # (would it fit if nothing else held the GPU?)
                    s = pf.settings(m, replace(gpu, free=gpu.total), "run")
                    print(f"  {repo:<16} {pf.gb(weights):>8} on the GPU (bf16 {pf.gb(m.bf16)}): not now ({pf.gb(gpu.free)} free, it needs {pf.gb(needs)}); on an idle GPU a {s.context:,}-token context")
                except pf.Refusal:
                    print(f"  {repo:<16} {pf.gb(weights):>8} on the GPU (bf16 {pf.gb(m.bf16)}): does not fit; it needs {pf.gb(needs)} free")
        idle = None
        for repo, cfg in pf.LADDERS["qwen"]:
            for g in (gpu, replace(gpu, free=gpu.total)):
                try:
                    pf.settings(pf.model_of(repo, cfg), g, "run")
                except pf.Refusal:
                    continue
                if g is gpu:
                    best = repo
                idle = repo
    print()
    if gpu is None:
        print("glyd run cannot run models on this computer (it needs Linux and an NVIDIA GPU). " + pf.COMPRESSION)
    elif not ready:
        print("Not ready: fix the lines marked NO.")
    elif best is None and idle:
        print(f"Only {pf.gb(gpu.free)} of the GPU's {pf.gb(gpu.total)} is free now: other programs hold the rest (nvidia-smi lists them). With the GPU to itself: glyd run {idle}")
    elif best is None:
        print("This GPU has room for no model Glyd suggests.")
    else:
        print(f"Ready: glyd run {best}")
    return 0 if ready else 1


def cmd_login(argv):
    argparse.ArgumentParser(prog="glyd login", description="Save a Hugging Face token, for models that ask you to accept a licence first (Llama, Gemma).").parse_args(argv)
    try:
        from huggingface_hub import login, whoami
    except ImportError:
        raise pf.Refusal("huggingface_hub is not installed", "Run the installer: " + pf.INSTALLER) from None
    print("Hugging Face's own login follows. A token with read access comes from https://huggingface.co/settings/tokens (nothing shows as you paste it).")
    try:
        login(add_to_git_credential=False, skip_if_logged_in=False)  # (False: a token already saved, stale or without access to the model, is asked about again)
        who = whoami()
    except KeyboardInterrupt:
        raise
    except Exception as e:  # (a malformed token is a ValueError, a refused one an HTTP error)
        raise pf.Refusal(f"Hugging Face did not accept that token ({type(e).__name__})", "Make a new one at https://huggingface.co/settings/tokens with read access, and run: glyd login") from None
    print(f"Logged in to Hugging Face as {clean(str(who.get('name') or who.get('fullname') or 'your account'))}.")
    return 0


def install_signal_handlers():
    """A killed glyd stops its server (Server.stop runs in finally): SIGTERM, and SIGHUP where the terminal goes away, unless the hangup is
    ignored (nohup: `ssh host 'nohup glyd serve MODEL &'` is how a server is left running after logout)."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    hup = getattr(signal, "SIGHUP", None)  # (none on Windows)
    if hup is not None and signal.getsignal(hup) != signal.SIG_IGN:
        signal.signal(hup, lambda *_: sys.exit(129))


def main(cmd, argv):
    install_signal_handlers()
    ui = Ui()
    try:
        return {"run": cmd_run, "serve": cmd_serve, "doctor": cmd_doctor, "login": cmd_login}[cmd](argv)
    except pf.Refusal as r:
        say_refusal(r, ui)
        return 1
    except KeyboardInterrupt as e:
        ui.line("\nStopped." + (" The download resumes where it stopped, next time." if "download" in str(e) else ""))
        return 130
    except BrokenPipeError:  # (glyd doctor | head: the reader went away)
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 0
    except SystemExit:
        raise
    except Exception as e:
        if os.environ.get("GLYD_DEBUG"):
            raise
        ui.line(f"glyd: unexpected error: {type(e).__name__}: {e}")
        ui.line(f"  Set GLYD_DEBUG=1 for the traceback, and report it at {ISSUES}")
        return 1
