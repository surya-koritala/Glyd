"""runtests.py TESTFILE_DIR [NAME ...]: each test of test_onboard.py in a process of its own, with a time limit (default 60 s; TIMEOUT=N).
Prints ok / FAIL (with the first line of the error) / TIMEOUT for each, and the totals."""
import os, re, subprocess, sys

d = sys.argv[1]
src = open(os.path.join(d, "test_onboard.py")).read()
names = sys.argv[2:] or re.findall(r"^def (test_\w+)\(", src, re.M)
limit = int(os.environ.get("TIMEOUT", "60"))
bad = 0
for n in names:
    code = f"import sys; sys.argv=['x']\nimport test_onboard as t\nt.{n}()\nprint('ok')"
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True, timeout=limit)
        out = r.stdout.strip().splitlines()
        err = r.stderr.strip().splitlines()
        status = "ok" if r.returncode == 0 and out and out[-1] == "ok" else "FAIL " + ((err or out or [""])[-1][:150])
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    bad += status != "ok"
    print(f"{n:<76} {status}", flush=True)
print(f"{len(names) - bad} of {len(names)} ok")
sys.exit(1 if bad else 0)
