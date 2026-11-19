"""glyd-gpu's wheel: every module of glyd_gpu compiled (Cython, Python's semantics kept), and no .py in it but the two files
Python reads as source (and, in a Linux wheel, glyd-engine, the engine's program, a binary that build.sh puts in the package). It is a release build: what a module holds between its `# dev` and `# end dev` lines (the tuning
variables, which a run from the sources reads and no release reads) is not compiled."""
import glob
import os
import re
import shutil
import sys

from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
# python -m glyd_gpu: runpy runs a module from its code object, which a compiled one has none of; _trace.py: torch.compile
# (Dynamo, vLLM's too) traces the bytecode of the functions it reaches, which a compiled function has none of (its header says why)
SOURCE = {"__main__", "_trace"}

Options.docstrings = False  # (a compiled module's docstrings are strings in the binary)
DIRECTIVES = {
    "language_level": 3,
    "annotation_typing": False,  # (annotations stay what Python makes of them: not types enforced at a call)
    "binding": True,  # (functions are as Python's: signatures, introspection)
    "emit_code_comments": False,
    "embedsignature": False,
    "profile": False,
    "linetrace": False,
}
# the stable ABI (CPython 3.10's Py_LIMITED_API): a wheel an architecture, for every CPython from 3.10 on (cp310-abi3);
# GLYD_GPU_ABI3=0: one for the CPython that builds it
ABI3 = os.environ.get("GLYD_GPU_ABI3", "1") != "0"
LIMITED = [("Py_LIMITED_API", "0x030A0000"), ("CYTHON_LIMITED_API", "1")] if ABI3 else []
CC = ["-O2", "-g0", "-fvisibility=hidden", "-DCYTHON_CLINE_IN_TRACEBACK=0"]  # (no C line in a traceback, whatever a later Cython defaults to)
LD = ["-Wl,-x"] if sys.platform == "darwin" else ["-Wl,--strip-all"]  # (no symbols but the module's entry point)

modules = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(HERE, "glyd_gpu", "*.py")))
STAGE = os.path.join("build", "src", "glyd_gpu")  # the modules as they are compiled, without their dev blocks


def strip_dev(text, name):
    """text without its dev blocks: the lines from `# dev` (or `# dev: ...`) to `# end dev`, in which a module reads the tuning variables,
    blank (the lines of what is left keep their numbers)."""
    out, inside = [], False
    for line in text.splitlines(keepends=True):
        mark = line.strip()
        if re.fullmatch(r"# dev(:.*)?", mark):
            if inside:
                raise SystemExit(f"{name}: a dev block inside a dev block")
            inside = True
        elif mark == "# end dev":
            if not inside:
                raise SystemExit(f"{name}: `# end dev` with no `# dev` before it")
            inside = False
        if inside or mark == "# end dev":
            out.append("\n")
        else:
            out.append(line)
    if inside:
        raise SystemExit(f"{name}: a dev block that is not ended")
    return "".join(out)


shutil.rmtree(os.path.join(HERE, "build", "src"), ignore_errors=True)
os.makedirs(os.path.join(HERE, STAGE))
for m in modules:
    with open(os.path.join(HERE, "glyd_gpu", f"{m}.py"), encoding="utf-8") as f:
        text = f.read()
    with open(os.path.join(HERE, STAGE, f"{m}.py"), "w", encoding="utf-8") as f:
        f.write(strip_dev(text, f"glyd_gpu/{m}.py"))


_here = os.getcwd()
os.chdir(os.path.join(HERE, "build", "src"))  # (the paths a compiled module names in a traceback: glyd_gpu/kvcal.py)
try:
    EXTENSIONS = cythonize(
        [Extension(f"glyd_gpu.{m}", [f"glyd_gpu/{m}.py"], extra_compile_args=CC, extra_link_args=LD, define_macros=LIMITED, py_limited_api=ABI3) for m in modules if m not in SOURCE],
        compiler_directives=DIRECTIVES,
        build_dir=os.path.join(HERE, "build", "cython"),  # (no nthreads: its worker processes do not have Options.docstrings; the C compiler runs in parallel below)
    )
finally:
    os.chdir(_here)


class BuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        return [m for m in super().find_package_modules(package, package_dir) if m[1] in SOURCE]

    def build_package_data(self):
        super().build_package_data()
        for f in self.get_outputs(include_bytecode=False):
            if os.path.basename(f) == "glyd-engine":  # (the engine's program is run: package data is copied without its mode, and an installer takes a file's mode from its wheel)
                os.chmod(f, 0o755)


class BuildExt(build_ext):
    def initialize_options(self):
        super().initialize_options()
        self.parallel = int(os.environ.get("GLYD_GPU_JOBS", 0)) or os.cpu_count()


setup(
    ext_modules=EXTENSIONS,
    cmdclass={"build_py": BuildPy, "build_ext": BuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp310"}} if ABI3 else {},
)
