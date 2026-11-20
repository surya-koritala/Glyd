"""glyd run on Glyd's own engine, glyd-engine (the program that ships beside the library in this package), on the GPUs where it is measured at least as fast as vLLM and will certainly run the model,
and on vLLM everywhere else, as the earlier release did: THE SPEED BAR. The engine serves up to 8 users at once, so `glyd serve` of a lossless model is vLLM's (GLYD_ENGINE=1 asks for the engine there), and which GPUs
the engine is used on for `glyd run` is one list in the engine (machine.rs: GPUS), which measurements fill and which is empty until they are made. One line on stderr says which runs the model.

The engine is asked first, before anything is downloaded (`glyd-engine preflight MODEL`: the GPU that would be used, whether the engine is measured at least as fast as vLLM on it, its driver, the kind
of model, the memory, the disks, the KV cache). Where it says yes the engine runs the command as this process's child (the signals this process gets are its), with a status file (`--status-file`) in which it shows it is alive while it starts, writes
`ready` when it is up (serving, or about to answer), and leaves the reason where it refuses. Until `ready` nothing irreversible is done (it does not read standard input, it holds no port), and
every way the start can end before it, a refusal, a non-zero exit, a crash (a panic, SIGSEGV, SIGKILL, an out-of-memory kill) or going silent for START_STALL seconds, runs the same command on vLLM,
with one line saying why. After `ready` the engine's exit status is the command's: a crash is said in one line and the model is not started again.
An API key or a token given as a flag (--api-key, --hf-token) is taken off the engine's command line and goes to it in its environment (GLYD_API_KEY, HF_TOKEN): a command line is in `ps` and in a log.

    GLYD_ENGINE=0      the vLLM path, whatever the engine would run (off, no, false and vllm too)
    GLYD_ENGINE=1      the engine on any GPU it runs on, measured or not, for glyd run and glyd serve (on, yes and true too; its checks of the GPU, the driver and the model still apply)
    GLYD_ENGINE=PATH   another engine program in place of the one beside the library (the PATH's glyd-engine otherwise); for glyd serve it asks for the engine as 1 does

`glyd run MODEL:kestrel` (also `glyd run MODEL --lossy`: a Qwen3.5-family model at about 6.5 bits per weight) and `MODEL:swift` (about 5.5) run on the engine alone, as vLLM has no such path: where the engine
does not run them, the reason is said and the command ends with status 3, and nothing runs on vLLM. The engine reads these two tags itself: they go to it as they were typed. `MODEL:penguin` is the lossless
default: its tag is taken off before the engine and vLLM see the model (vLLM's path never sees a tag). `MODEL:hummingbird` (Nemotron 3 Super's experts at about 3 bits, on a 16 to 32 GB card) is one of these too. `glyd serve` takes
the same tags and --lossy, and a level, a glyd-v7 directory and a glyd/ repo vLLM cannot read go to the engine without GLYD_ENGINE=1 (nothing else serves them). A tag that names no level built (wren is named, not built in a release) and --lossy with another
level are refused with status 1, and nothing runs.

The settings the vLLM path reads and the engine does not (GLYD_KV other than auto, GLYD_LAYOUT, GLYD_EXACT, GLYD_VERIFY, GLYD_FRACTION) keep the command on the vLLM path where one is set. The
Rust `glyd` (the compression program) passes run and serve to this tool, with GLYD_ENGINE naming the engine program it found where the user named none, and GLYD_ENGINE_ASKED=1 where the user's
GLYD_ENGINE was 1 (the tool tells the engine's preflight so in the same variable).
Under the Apache License 2.0 (LICENSE), as the rest of Glyd.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

NAME = "glyd-engine"
OFF = ("0", "off", "no", "false", "vllm")
ON = ("1", "on", "yes", "true", "auto")
ASK = ("1", "on", "yes", "true")  # (of ON, what asks for the engine by name: `auto` and nothing leave it to the default)
CANNOT = 3  # the engine's status for a model it will not run on this computer, from `preflight` and from `run` and `serve` until they are up
PANIC = 101  # a Rust program's status where it panics
LEVELS = ("penguin", "kestrel", "swift", "hummingbird")  # the levels built, as a tag on the model names them; kestrel (which --lossy is), swift and hummingbird are lossy, which the engine alone runs
NOT_BUILT = ("wren",)  # named, not built yet (Glyd's levels are penguin, kestrel, swift, wren and hummingbird), but for Gemma 4: its wren is Google's QAT weights, which the engine says where it has
WITH_VALUE = ("--prompt", "-p", "--context", "--ctx", "--port", "--host", "--temp", "--layout", "--users", "--tables", "--api-key", "--hf-token", "--allowed-origins", "--allowed-hosts")
# the flags that carry a secret, which go to the engine in its environment (a command line is in `ps` and in a log): (the flag's names, the variable, what it is)
SECRETS = ((("--api-key", "--api_key"), "GLYD_API_KEY", "API key"), (("--hf-token", "--hf_token"), "HF_TOKEN", "Hugging Face token"))
VLLM_SETTINGS = {"GLYD_KV": ("", "auto"), "GLYD_LAYOUT": ("", "auto"), "GLYD_FRACTION": ("", "1", "1.0"), "GLYD_EXACT": ("", "0", "false", "no", "off"), "GLYD_VERIFY": ("", "0", "false", "no", "off")}
PREFLIGHT_SECONDS = 120  # (the Hub's answers and the GPU's first look: seconds; a wedged driver is not waited for)
START_STALL = 900  # seconds the engine may show no sign of life before it is up: a download, a packing or the KV cache's tables move its status file, a hang does not (a fixed limit would stop a slow first download)
POLL = 0.5  # seconds between looks at the status file
SERVE_USERS = 8  # the most users the engine serves at once (its --users): above that glyd serve is vLLM's
USER_STOPS = (129, 130, 131, 143)  # the shell's numbers for SIGHUP, SIGINT, SIGQUIT and SIGTERM: what an engine that is stopped by the user exits with
PASSED_ON = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")


def choice(environ=None):
    """(wanted, path): wanted is False for GLYD_ENGINE=0 and its spellings; path is the program GLYD_ENGINE names, else None."""
    v = (os.environ if environ is None else environ).get("GLYD_ENGINE", "").strip()
    if v.lower() in OFF:
        return False, None
    if not v or v.lower() in ON:
        return True, None
    return True, v


def asked_for(environ=None):
    """Whether the user asked for the engine by name: GLYD_ENGINE=1 (on, yes, true) or the path of a program. `auto` and nothing leave it to the default, which is the speed bar's: for `glyd serve`, vLLM."""
    wanted, named = choice(environ)
    v = (os.environ if environ is None else environ).get("GLYD_ENGINE", "").strip().lower()
    return wanted and (v in ASK or named is not None)


def forced(cmd, environ=None):
    """Whether the engine is used past its list of GPUs measured at least as fast as vLLM (its checks of the GPU, the driver and the model still apply): where the user asked for it by name, GLYD_ENGINE=1
    (on, yes, true), which the Rust glyd passes on as GLYD_ENGINE_ASKED=1, and for glyd serve a program's path too (the ask there, as 1 is)."""
    env = os.environ if environ is None else environ
    v = env.get("GLYD_ENGINE", "").strip().lower()
    return choice(env)[0] and (v in ASK or env.get("GLYD_ENGINE_ASKED", "").strip() == "1" or (cmd == "serve" and asked_for(env)))


def find(path=None):
    """The engine's program: `path` (GLYD_ENGINE's), else the one beside the library in this package, else the PATH's; None where there is none."""
    if path:
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    beside = os.path.join(os.path.dirname(os.path.abspath(__file__)), NAME)
    if os.path.isfile(beside) and os.access(beside, os.X_OK):
        return beside
    return shutil.which(NAME)


def model_at(argv):
    """Where the model of `glyd run MODEL ...` is in argv: the first word that is not a flag or a flag's value (None where there is none)."""
    it = enumerate(argv)
    for i, a in it:
        if a == "--":
            return None
        if a in WITH_VALUE:
            next(it, None)
        elif not a.startswith("-"):
            return i
    return None


def model_of(argv):
    """The model of `glyd run MODEL ...` (None where there is none)."""
    i = model_at(argv)
    return None if i is None else argv[i]


def level_of(argv):
    """(argv with the level tag taken off the model, the tag in lower case): `Qwen/Qwen3.8-27B:swift` has the tag `swift`. The tag is the text after the model's last ':' where that is all ASCII letters
    (a Hub name has no ':'; in a path the text after one has a '/' or more); a model with none is as it is, and the tag is None. Whether the tag names a level is `handoff`'s to say."""
    i = model_at(argv)
    if i is None:
        return argv, None
    model, colon, tag = argv[i].rpartition(":")
    if not (colon and tag.isascii() and tag.isalpha()):
        return argv, None
    return [*argv[:i], model, *argv[i + 1 :]], tag.lower()


def is_gemma4(model):
    """Whether MODEL is a Gemma 4 model: its directory's config.json says so, else its name does (google/gemma-4-31B-it, glyd/gemma-4-31B-wren), as the engine's is_gemma4_model reads it. Glyd's engine alone
    runs Gemma 4 here: at :wren (Google's QAT weights) and at penguin, which vLLM has no route for."""
    cfg = os.path.join(model, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg) as f:
                c = json.load(f)
            return any(str(d.get("model_type", "")).startswith("gemma4") for d in (c, c.get("text_config") or {}))
        except (OSError, ValueError, AttributeError):
            return False
    return model.rstrip("/").rsplit("/", 1)[-1].lower().startswith("gemma-4")


def engine_only(model):
    """Whether MODEL is a glyd/ repo that vLLM cannot read: Gemma 4's (penguin, wren) and every lossy level's; or a directory of one, a glyd-v7 save (its glyd.json says so). A Qwen penguin repo (glyd-v4) loads in
    the vLLM plugin."""
    if model and os.path.isdir(model):
        try:
            with open(os.path.join(model, "glyd.json")) as f:
                return json.load(f).get("format") == "glyd-v7"
        except (OSError, ValueError, AttributeError):
            return False
    if not model or not model.startswith("glyd/") or model.count("/") != 1:
        return False
    return is_gemma4(model) or model.lower().endswith(tuple(f"-{level}" for level in ("kestrel", "swift", "hummingbird", "wren")))


def packed_repo(model, level):
    """The Hub repo of MODEL (`Qwen/Qwen3.8-27B`) at `level` (`swift`) packed ahead: `glyd/Qwen3.8-27B-swift`. The engine's rule too (cli.rs packed_repo), and the publish workflow's; the engine resolves it, the tag goes to it as typed."""
    name = model.rstrip('/').rsplit('/', 1)[-1]
    return f"glyd/{REPO_NAMES.get(name, name)}-{level}"


REPO_NAMES = {"NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nemotron-3", "gemma-4-31B-it-qat-q4_0-unquantized": "gemma-4-31B", "gemma-4-31B-it": "gemma-4-31B", "gemma-4-31B": "gemma-4-31B-base", "gemma-4-12B-it-qat-q4_0-unquantized": "gemma-4-12B", "gemma-4-12B-it": "gemma-4-12B", "gemma-4-12B": "gemma-4-12B-base"}


def take_flag(argv, names):
    """(the value, the flags without it): `--api-key KEY` or `--api-key=KEY` out of the flags before a lone `--` (what follows is vLLM's). None where the flag is not there or has no value after it
    (`--hf-token` alone means the saved login: it stays). The value of another flag is not a flag of ours, whatever it looks like."""
    out, value, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            out.extend(argv[i:])
            break
        name, eq, v = a.partition("=")
        if name in names:
            if eq:
                value, i = v, i + 1
                continue
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                value, i = argv[i + 1], i + 2
                continue
        elif name in WITH_VALUE and not eq and i + 1 < len(argv):
            out.extend(argv[i : i + 2])
            i += 2
            continue
        out.append(a)
        i += 1
    return value, out


def vllm_setting(environ):
    """NAME=value of a setting that only the vLLM path has, where one is set to something other than what it is without it; else None."""
    for name, unset in VLLM_SETTINGS.items():
        value = environ.get(name, "").strip()
        if value.lower() not in unset:
            return f"{name}={value}"
    return None


def reason(text):
    """The first line of an engine's refusal as a clause: `glyd: X is a Y model.` as `X is a Y model`."""
    line = (text.strip().splitlines() or [""])[0].strip()
    return (line[len("glyd: ") :] if line.startswith("glyd: ") else line).rstrip(".")


def _how(rc):
    """How a program ended, in words: a Popen status as `exit status 101` or `signal 11 (SIGSEGV)`."""
    if rc < 0:
        try:
            return f"signal {-rc} ({signal.Signals(-rc).name})"
        except ValueError:
            return f"signal {-rc}"
    return f"exit status {rc}"


def _span(seconds):
    return f"{round(seconds / 60)} minutes" if seconds >= 120 else f"{seconds:g} seconds"


def _look(path):
    """(the status file's modification time, what it says): (None, "") where it cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        return os.stat(path).st_mtime_ns, text
    except OSError:
        return None, ""


def _follow_parent():
    """The engine's program, after the fork and before it starts: where this process dies of a signal that cannot be caught, so does the engine, as it would if it were this process (Linux; None elsewhere)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        prctl = ctypes.CDLL(None).prctl
    except (ImportError, OSError, AttributeError):
        return None
    return lambda: prctl(1, int(signal.SIGTERM))  # (PR_SET_PDEATHSIG)


def _stop(child):
    """A program asked to stop, and made to if it does not."""
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def launch(exe, cmd, argv, start=subprocess.Popen, stall=START_STALL, poll=POLL, model="", environ=None):
    """The engine run as this process's child with the terminal, standard input and standard output as they are: (kind, exit status, words). The signals this process gets (Ctrl-C included: the terminal
    gives it to the engine as well) are passed to it, so that it stops as it would if it were this process; a hangup that is ignored (nohup) stays ignored. kind is
      "failed"   the engine did not get up (it refused, exited, crashed, was killed, or showed no sign of life for `stall` seconds): the words say why, and the command goes on with vLLM;
      "stopped"  the user stopped it before it was up (Ctrl-C, a kill): nothing more is done;
      "ran"      it was up, or it ended well: its exit status is the command's, and the words, where it ended badly after it was up, say how (it is not started again)."""
    fd, path = tempfile.mkstemp(prefix="glyd-engine-")
    os.close(fd)
    held, told, child = {}, [], None
    try:
        try:
            child = start([exe, cmd, "--status-file", path, *argv], preexec_fn=_follow_parent(), env=environ)
        except OSError as e:
            return "failed", CANNOT, f"{NAME} did not start ({e.strerror or e})"

        def pass_on(sig, _frame):
            told.append(sig)
            try:
                child.send_signal(sig)
            except OSError:
                pass

        for sig in (getattr(signal, n) for n in PASSED_ON if hasattr(signal, n)):
            if signal.getsignal(sig) != signal.SIG_IGN:
                held[sig] = signal.signal(sig, pass_on)
        up, hung, seen, moved = False, False, None, time.monotonic()
        while True:
            try:
                rc = child.wait(poll)
                break
            except subprocess.TimeoutExpired:
                pass
            mtime, text = _look(path)
            if text == "ready":
                up = True
                rc = child.wait()  # (up: nothing is looked at any more)
                break
            if mtime != seen:
                seen, moved = mtime, time.monotonic()
            elif time.monotonic() - moved > stall:
                hung = True
                _stop(child)
                rc = child.returncode
                break
        text = _look(path)[1]
        up = up or text == "ready"
        code = rc if rc >= 0 else 128 - rc  # (the shell's number for a signal)
        if hung:
            return "failed", code, f"{NAME} gave no sign of life for {_span(stall)} before {model} was up"
        if rc == 0:
            return "ran", 0, ""
        if up:
            badly = (rc < 0 and code not in USER_STOPS) or rc == PANIC
            return "ran", code, f"{NAME} stopped after {model} was up ({_how(rc)}); it is not started again" if badly else ""
        if told or code in USER_STOPS:
            return "stopped", code, ""
        if text and text != "ready":
            return "failed", code, reason(text)
        if rc < 0:
            return "failed", code, f"{NAME} was stopped by {_how(rc)} before {model} was up"
        return "failed", code, f"{NAME} stopped before {model} was up ({_how(rc)}{'; its message is above' if rc == CANNOT else ''})"
    finally:
        for sig, old in held.items():
            signal.signal(sig, old)
        if child is not None:  # (this process is leaving with the engine still up: it is stopped)
            _stop(child)
        try:
            os.unlink(path)
        except OSError:
            pass


def handoff(cmd, argv, environ=None, say=None, run=subprocess.run, start=subprocess.Popen, stall=START_STALL, poll=POLL):
    """None where the vLLM path goes on, after one line saying why. Where the engine will run the model, a line, and the engine run as this process's child: its exit status, which is the command's, once
    it is up; and None again (the vLLM path too, with the reason as the line) where it does not get up, whatever the way. The model's level tag is read here, for run and serve alike: kestrel and swift are --lossy's case (they
    go to the engine as they were typed), penguin's is taken off the words the engine is given (and the caller takes it off for vLLM's path, with level_of), and a tag that is no level built is refused."""
    env = os.environ if environ is None else environ
    say = say or (lambda text: print(text, file=sys.stderr, flush=True))
    wanted, named = choice(env)

    flags = argv[: argv.index("--")] if "--" in argv else argv
    lossy = "--lossy" if "--lossy" in flags else ""  # (the words that ask for a lossy level: --lossy, or the model's tag, below)

    def vllm(why, out=""):
        alone = lossy or (model if engine_only(model) else "")
        if alone:  # (vLLM has no lossy levels, nor a route for a Gemma 4 or lossy glyd/ repo: the reason, and nothing more)
            say(f"glyd {cmd}: {why}; {alone} runs on Glyd's own engine alone.")
            return CANNOT
        say(f"glyd {cmd}: {why}; using vLLM{out}.")
        return None

    def refuse(what, fix):  # (a level asked for that cannot be had is the user's to fix: status 1, and the engine's own words for it)
        say(f"glyd: {what}.\n  {fix}")
        return 1

    model = model_of(argv)
    if "--help" in argv or "-h" in argv or model is None:  # (the usage text is the vLLM path's, and so is a missing model's refusal)
        return None
    words, tag = level_of(argv)
    if tag and tag not in LEVELS and not (tag == "wren" and is_gemma4(model_of(words))):  # (Gemma 4's wren: the engine says which of its models have one)
        return refuse(f":{tag} is not built yet" if tag in NOT_BUILT else f":{tag} is not a level", "The levels available now: " + ", ".join(LEVELS) + " (and wren for Gemma 4)")
    if lossy and tag not in (None, "kestrel"):
        return refuse(f"--lossy is :kestrel, not :{tag}", "Give one or the other")
    if tag == "penguin":  # (the engine's preflight and run, and vLLM, have the model without its tag)
        argv, model = words, model_of(words)
    elif tag:  # (kestrel, swift or hummingbird: the engine is given them, and reads them)
        lossy = lossy or f":{tag}"
    if not wanted:
        return vllm("GLYD_ENGINE=0")
    if "--" in argv:
        return vllm("the flags after -- are vLLM's")
    setting = vllm_setting(env)
    if setting:
        return vllm(f"{setting} is a setting of the vLLM path")
    if cmd == "serve" and not asked_for(env) and not (lossy or is_gemma4(model) or engine_only(model)):  # (the speed bar: the engine serves up to 8 users at once, and vLLM is at least as fast at every load above that; a level, Gemma 4 and a glyd-v7 save are the engine's alone)
        return vllm(f"the engine serves up to {SERVE_USERS} users at once and vLLM more (GLYD_ENGINE=1 runs the engine)")
    exe = find(named)
    if exe is None:
        return vllm(f"GLYD_ENGINE={named} is not a program" if named else f"{NAME} is not installed here")
    secrets, run_argv = [], list(argv)  # (a key or a token given as a flag goes to the engine in its environment: a command line is in `ps` and in a log)
    for names, var, what in SECRETS:
        value, run_argv = take_flag(run_argv, names)
        if value:
            secrets.append((var, value, what))
    probe = dict(os.environ, **{var: value for var, value, _ in secrets if var == "HF_TOKEN"})  # (a gated model's config is read with the token)
    probe.pop("GLYD_ENGINE_ASKED", None)
    if forced(cmd, env):
        probe["GLYD_ENGINE_ASKED"] = "1"  # (the preflight skips its list of measured GPUs, and nothing else)
    try:  # (before anything is downloaded: whether the engine will certainly run this model on this computer)
        asked = run([exe, "preflight", *run_argv], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace", timeout=PREFLIGHT_SECONDS, env=probe)
    except OSError as e:
        return vllm(f"{NAME} did not start ({e.strerror or e})")
    except subprocess.TimeoutExpired:
        return vllm(f"{NAME} did not answer in {PREFLIGHT_SECONDS} seconds")
    if asked.returncode != 0:
        why = reason(asked.stderr) or f"{NAME} did not start (exit status {asked.returncode})"
        if asked.returncode == CANNOT and is_gemma4(model) and "of GPU memory" in why:  # (vLLM's bf16 Gemma 4 takes more memory than the engine's penguin: the engine's reason and its way out, not a second estimate from vLLM)
            say("\n".join([f"glyd {cmd}: {why}."] + [line.rstrip() for line in asked.stderr.strip().splitlines()[1:]]))
            return CANNOT
        return vllm(why)
    say(f"glyd {cmd}: {model} on {NAME} (GLYD_ENGINE=0: vLLM instead).")
    for var, value, what in secrets:
        say(f"glyd {cmd}: the {what} goes to the engine in {var}, not on its command line (next time set {var} yourself: a shell's history keeps what you type).")
    kind, code, why = launch(exe, cmd, run_argv, start, stall, poll, model, dict(os.environ, **{var: value for var, value, _ in secrets}))
    if kind == "failed":
        return vllm(why, " (GLYD_ENGINE=0 skips the engine next time)")
    if why:
        say(f"glyd {cmd}: {why}.")
    return code
