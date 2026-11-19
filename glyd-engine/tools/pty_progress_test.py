"""What a terminal shows during a first run's download: python pty_progress_test.py ENGINE MODEL

Runs `ENGINE fetch MODEL` in a pseudo-terminal with a fresh Hugging Face cache (HF_HOME=a new directory) and checks the screen: a bar that fills (lines rewritten with a carriage return, a bar of # and its
percentage, the rate and the time left), the line that ends it, and nothing left over."""
import os
import pty
import re
import select
import shutil
import sys
import tempfile
import time

ENGINE, MODEL = sys.argv[1], sys.argv[2]
home = tempfile.mkdtemp(prefix="hf-pty-", dir=os.environ.get("TMPDIR", "/tmp"))
os.environ["HF_HOME"] = home
os.environ["TERM"] = "xterm"
pid, fd = pty.fork()
if pid == 0:
    os.execv(ENGINE, [ENGINE, "fetch", MODEL])
out = b""
end = time.time() + 900
while time.time() < end:
    r, _, _ = select.select([fd], [], [], 0.5)
    if r:
        try:
            d = os.read(fd, 65536)
        except OSError:
            break
        if not d:
            break
        out += d
    p, status = os.waitpid(pid, os.WNOHANG)
    if p:
        try:
            while True:
                d = os.read(fd, 65536)
                if not d:
                    break
                out += d
        except OSError:
            pass
        break
shutil.rmtree(home, ignore_errors=True)
text = out.decode(errors="replace")
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (" -- " + str(detail)[-300:].replace("\r", "<CR>").replace("\x1b", "<ESC>") if detail != "" else ""))
    if not ok:
        fails.append(name)


frames = [f for f in text.split("\r") if "[" in f and "#" in f or "[                        ]" in f]
check("a bar is drawn and rewritten", len(frames) > 3, len(frames))
check("it fills: the last frame is at 100%", any("100%" in f for f in frames), frames[-1] if frames else "")
check("with a rate and a time left in between", any("MB/s" in f and "left" in f for f in frames), [f for f in frames if "MB/s" in f][:1])
check("the line that ends it", re.search(r"Downloaded \S+ \([\d.]+ GB\)\.", text) is not None, text[-200:])
check("no line is wider than the terminal (80 columns)", all(len(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", f)) < 80 for f in frames), max((len(f) for f in frames), default=0))
check("nothing after the last line but the shell's", text.rstrip().endswith(".") or text.rstrip().endswith(")."), text[-60:])
print("FAILED: " + ", ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
