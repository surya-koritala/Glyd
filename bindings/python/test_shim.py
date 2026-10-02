"""glyd.gpu and the `glyd` command's GPU half are the glyd-gpu package's, found where they are asked for: with the package missing, one
clear message; with a stand-in for it, the same objects, and the commands handed to it.

    python test_shim.py              (or pytest test_shim.py)"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = '''
__version__ = "0.0.0"
fit = lambda *a, **k: "fit"
Fit = type("Fit", (), {})
from_pretrained = lambda *a, **k: "from_pretrained"
compress = lambda *a, **k: "compress"
save_pretrained = lambda *a, **k: "save_pretrained"
'''
FAKE_CLI = '''
calls = []
def main(argv, prog="x"):
    calls.append((argv, prog))
    return 7
'''


def run(code, fake=False):
    """code run in a Python that has glyd (this directory) and, with fake, a stand-in for glyd_gpu: its output, else the error."""
    with tempfile.TemporaryDirectory() as d:
        if fake:
            os.makedirs(os.path.join(d, "glyd_gpu"))
            open(os.path.join(d, "glyd_gpu", "__init__.py"), "w").write(FAKE)
            open(os.path.join(d, "glyd_gpu", "_cli.py"), "w").write(FAKE_CLI)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = os.pathsep.join([HERE, d])
        env["GLYD_LIB"] = os.path.join(d, "no-such-library")  # (the codec's library is not what is looked at here)
        r = subprocess.run([sys.executable, "-c", code], cwd=d, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_without_glyd_gpu():
    out = run('''
import sys
sys.modules["glyd_gpu"] = None  # as if not installed
import glyd, glyd.gpu
for call in (lambda: glyd.from_pretrained("m"), lambda: glyd.fit("m"), lambda: glyd.gpu.compress(None)):
    try:
        call()
        raise AssertionError("the GPU half without glyd-gpu")
    except ImportError as e:
        assert 'pip install "glyd[gpu]"' in str(e) and "glyd-gpu" in str(e), e
try:
    glyd.gpu.no_such_name
    raise AssertionError("AttributeError")
except AttributeError:
    pass
from glyd import cli
import io
err, sys.stderr = sys.stderr, io.StringIO()
code = cli.main(["pack", "m", "out"])
msg, sys.stderr = sys.stderr.getvalue(), err
assert code == 1 and 'pip install "glyd[gpu]"' in msg, (code, msg)
print("ok")
''')
    assert out == "ok", out


def test_with_glyd_gpu():
    out = run('''
import glyd, glyd.gpu, glyd_gpu
assert glyd.fit is glyd_gpu.fit and glyd.gpu.fit is glyd_gpu.fit and glyd.gpu.Fit is glyd_gpu.Fit
assert glyd.from_pretrained is glyd_gpu.from_pretrained and glyd.save_pretrained is glyd_gpu.save_pretrained and glyd.gpu.compress is glyd_gpu.compress
from glyd import cli
from glyd_gpu import _cli
for cmd in cli.GPU_COMMANDS:
    assert cli.main([cmd, "a", "b"]) == 7
assert [c[0][0] for c in _cli.calls] == ["run", "serve", "doctor", "login", "pack", "verify"] and all(c[1] == "glyd" for c in _cli.calls) and cli.GPU_COMMANDS[-2:] == ("pack", "verify")
print("ok")
''', fake=True)
    assert out == "ok", out


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
