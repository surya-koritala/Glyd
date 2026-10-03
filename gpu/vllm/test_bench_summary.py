"""bench_summary.py on made-up results (no GPU): each server's highest prefix cache hit rate is printed, and above 1% the
run is flagged NOT COMPARABLE.

    python test_bench_summary.py"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = "INFO 10-02 12:00:00 [loggers.py:259] Engine 000: Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: {}%\n"


def summary(bf16, glyd):
    """bench_summary.py's output for servers whose logs have these hit rates."""
    with tempfile.TemporaryDirectory() as d:
        for mode, rates in (("bf16", bf16), ("glyd", glyd)):
            json.dump({"request_throughput": 1.0, "completed": 1}, open(f"{d}/{mode}-rate1.json", "w"))
            open(f"{d}/serve-{mode}.txt", "w").write("".join(LINE.format(r) for r in rates))
        return subprocess.run([sys.executable, f"{HERE}/bench_summary.py", d], capture_output=True, text=True, check=True).stdout


clean = summary(["0.0", "0.0"], ["0.0", "0.9"])
assert "bf16 0.0%, glyd 0.9%" in clean and "NOT COMPARABLE" not in clean, clean
flagged = summary(["0.0"], ["17.7", "48.3", "21.8"])
assert "bf16 0.0%, glyd 48.3%" in flagged and flagged.count("NOT COMPARABLE") == 2 and "above 1%: glyd 48.3%" in flagged, flagged
assert "bf16 n/a" in summary([], ["0.0"]), "a log without the line"
print("test_bench_summary: 3 of 3 passed")
