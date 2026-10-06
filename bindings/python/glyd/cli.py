"""The `glyd` command.

    glyd run Qwen/Qwen3.5-9B        # download, start and chat: the GPU half (the glyd-gpu package)
    glyd serve Qwen/Qwen3.5-9B      # the same, left up as an OpenAI API
    glyd doctor                     # what this machine has, and what fits
    glyd login                      # a Hugging Face token, for gated models
    glyd pack MODEL OUT             # a model's weights packed on the GPU and saved; glyd verify PATH checks a save
    glyd input.tar -o input.tar.glyd   # anything else: the compression program (the Rust `glyd`), found on PATH

The wheel carries the libraries, not the Rust program, so a compression command is forwarded to the `glyd` executable that is
not this one: the first file named glyd on PATH that is executable, is not this entry point (by its real path) and is not another
copy of this Python tool (a console script, which imports glyd.cli); an empty PATH entry (the current directory) is skipped. A
wrapper script around the real program (Nix, asdf, mise) counts. The forwarded program is started with GLYD_FORWARDED=1 and a glyd
that finds it set does not forward again, whatever sits on PATH. Where there is none, the message says how to get it.
"""
import os
import subprocess
import sys
from . import __version__

GPU_COMMANDS = ("run", "serve", "doctor", "login", "pack", "verify")
HELP = """glyd {version}

Run a model on your NVIDIA GPU with its weights packed by Glyd (about a third less GPU memory, the same bits):
  glyd run MODEL              download, start and chat, in the terminal and at http://localhost:8000
  glyd serve MODEL            the same, left up as an OpenAI API for other apps
  glyd doctor                 check this machine: GPU, driver, compilers, which models fit
  glyd login                  save a Hugging Face token (for gated models)
  glyd pack MODEL OUT         MODEL's weights packed on the GPU, each pack checked, saved in OUT; glyd verify PATH checks a save
  MODEL is a Hugging Face name (Qwen/Qwen3.5-9B) or a folder. glyd run --help shows the options;
  any vLLM flag can follow a lone --:  glyd run MODEL -- --max-model-len 4096

Compress files and object-storage data:
"""
MISSING = """  The compression program is not installed here. Get it with one of:
    brew install surya-koritala/glyd/glyd
    cargo install --git https://github.com/surya-koritala/Glyd glyd glyd-store
    a release from https://github.com/surya-koritala/Glyd/releases
"""


def _is_python_entry(path):
    """Whether a file is a console script of this Python tool (any copy of it, in another environment): it imports glyd.cli."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return True
    return head.startswith(b"#!") and b"glyd.cli" in head


def find_native(path=None, me=None):
    """The Rust `glyd`: the first executable file named glyd on PATH that is not this entry point (by its real path) and not another copy
    of this Python tool; an empty entry of PATH is skipped."""
    me = os.path.realpath(sys.argv[0] if me is None else me)
    for d in (os.environ.get("PATH", "") if path is None else path).split(os.pathsep):
        if not d:
            continue
        exe = os.path.join(d, "glyd")
        if os.path.isfile(exe) and os.access(exe, os.X_OK) and os.path.realpath(exe) != me and not _is_python_entry(exe):
            return exe
    return None


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in GPU_COMMANDS:
        try:
            from glyd_gpu import _cli  # (the GPU half is the glyd-gpu package, imported where it is asked for)
        except ImportError as e:
            sys.stderr.write(f'glyd {args[0]}: the GPU half of glyd is the glyd-gpu package, which is not installed here: pip install "glyd[gpu]" ({e})\n')
            return 1
        return _cli.main(args, prog="glyd")
    forwarded = bool(os.environ.get("GLYD_FORWARDED"))  # (a glyd that was started by a glyd does not start another)
    native = None if forwarded else find_native()
    env = dict(os.environ, GLYD_FORWARDED="1")
    if not args or args[0] in ("-h", "--help", "help"):
        sys.stdout.write(HELP.format(version=__version__))
        sys.stdout.flush()
        if native:
            return subprocess.call([native, "--help"], stdout=sys.stdout.fileno(), stderr=subprocess.STDOUT, env=env)
        sys.stdout.write(MISSING)
        return 0
    if args[0] in ("-v", "--version"):
        print(f"glyd {__version__}")
        sys.stdout.flush()
        if native:
            return subprocess.call([native, "--version"], stdout=sys.stdout.fileno(), env=env)
        return 0
    if native is None:
        sys.stderr.write("glyd: " + MISSING.strip() + ("\n  (another glyd forwarded this command here: the compression program is not the one it found.)" if forwarded else "") + "\n")
        return 127
    try:
        os.execve(native, [native] + args, env)  # (its stdin, stdout and exit status are the program's)
    except OSError as e:  # (a file that is not a program for this machine, or not allowed to run)
        sys.stderr.write(f"glyd: cannot run {native}: {e.strerror or e}\n  It is the file named glyd that comes first on your PATH after this one.\n")
        return 126


if __name__ == "__main__":
    sys.exit(main())
