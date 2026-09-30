"""The `glyd` command.

    glyd run Qwen/Qwen3-8B          # download, start and chat: the GPU half (glyd.gpu.run)
    glyd serve Qwen/Qwen3-8B        # the same, left up as an OpenAI API
    glyd doctor                     # what this machine has, and what fits
    glyd login                      # a Hugging Face token, for gated models
    glyd input.tar -o input.tar.glyd   # anything else: the compression program (the Rust `glyd`), found on PATH

The wheel carries the libraries, not the Rust program, so a compression command is forwarded to the `glyd` executable that is
not this one: the first native file named glyd on PATH (a `#!` script, or this entry point itself, is skipped). Where there is none,
the message says how to get it.
"""
import os
import subprocess
import sys
from . import __version__

GPU_COMMANDS = ("run", "serve", "doctor", "login")
HELP = """glyd {version}

Run a model on your NVIDIA GPU with its weights packed by Glyd (about a third less GPU memory, the same bits):
  glyd run MODEL              download, start and chat, in the terminal and at http://localhost:8000
  glyd serve MODEL            the same, left up as an OpenAI API for other apps
  glyd doctor                 check this machine: GPU, driver, compilers, which models fit
  glyd login                  save a Hugging Face token (for gated models such as Llama)
  MODEL is a Hugging Face name (Qwen/Qwen3-8B) or a folder. glyd run --help shows the options;
  any vLLM flag can follow a lone --:  glyd run MODEL -- --max-model-len 4096

Compress files and object-storage data:
"""
MISSING = """  The compression program is not installed here. Get it with one of:
    brew install surya-koritala/glyd/glyd
    cargo install glyd
    a release from https://github.com/surya-koritala/Glyd/releases
"""


def _is_script(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"#!"
    except OSError:
        return True


def find_native(path=None, me=None):
    """The Rust `glyd`: the first executable file named glyd on PATH that is not this entry point (by its real path) and not a script."""
    me = os.path.realpath(sys.argv[0] if me is None else me)
    for d in (os.environ.get("PATH", "") if path is None else path).split(os.pathsep):
        exe = os.path.join(d or ".", "glyd")
        if os.path.isfile(exe) and os.access(exe, os.X_OK) and os.path.realpath(exe) != me and not _is_script(exe):
            return exe
    return None


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in GPU_COMMANDS:
        from .gpu import run  # (the GPU half is imported where it is asked for)

        return run.main(args[0], args[1:])
    native = find_native()
    if not args or args[0] in ("-h", "--help", "help"):
        sys.stdout.write(HELP.format(version=__version__))
        sys.stdout.flush()
        if native:
            return subprocess.call([native, "--help"], stdout=sys.stdout.fileno(), stderr=subprocess.STDOUT)
        sys.stdout.write(MISSING)
        return 0
    if args[0] in ("-v", "--version"):
        print(f"glyd {__version__}")
        sys.stdout.flush()
        if native:
            return subprocess.call([native, "--version"], stdout=sys.stdout.fileno())
        return 0
    if native is None:
        sys.stderr.write("glyd: " + MISSING.strip() + "\n")
        return 127
    os.execv(native, [native] + args)  # (its stdin, stdout and exit status are the program's)


if __name__ == "__main__":
    sys.exit(main())
