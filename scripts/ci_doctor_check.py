#!/usr/bin/env python3
"""The GPU check of `glyd doctor --bundle FILE`: on the runner's GPU the bundle says everything is fine, the CUDA backend is accepted, the sample plan loads, and
nothing secret or personal is in it.

usage: ci_doctor_check.py BUNDLE.json
"""
import json
import os
import sys

text = open(sys.argv[1]).read()
b = json.loads(text)
bad = []
checks = {c["id"]: c for c in b["checks"]}
if b["schema"] != "glyd-doctor/1":
    bad.append(f"schema {b['schema']}")
if not b["ok"]:
    bad.append("not ok: " + "; ".join(f"{c['id']}: {c['found']}" for c in b["checks"] if c["status"] == "fail"))
if "accepted" not in checks["backend-cuda"]["found"]:
    bad.append("CUDA: " + checks["backend-cuda"]["found"])
if not b["gpus"]:
    bad.append("no GPU in the bundle")
if b["sample_plan"].get("level") is None:
    bad.append(f"the sample plan does not load here: {b['sample_plan']}")
home = os.path.expanduser("~")
if len(home) > 1 and home in text:
    bad.append("the home folder is in the bundle")
for secret in (os.environ.get("HF_TOKEN"), os.environ.get("GLYD_API_KEY")):
    if secret and secret in text:
        bad.append("a secret is in the bundle")
for c in b["checks"]:
    if c["status"] != "ok" and c["fix"] is None and c["status"] == "fail":
        bad.append(f"{c['id']} fails with no fix")
print(f"{len(b['checks'])} checks, GPU {b['gpus'][0]['name'] if b['gpus'] else None}, sample plan {b['sample_plan'].get('level')}")
if bad:
    print("the bundle is not what it should be:\n  " + "\n  ".join(bad))
    sys.exit(1)
