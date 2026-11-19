#!/usr/bin/env bash
# The owner's hard Runpod cap (CAP, default $100) for a day's work; run one per machine in the background. Every 60 s: the pods named glyd-* are read, each pod's
# cost so far is costPerHr x the longest uptime seen for it (kept per pod id in ledger.tsv, so deleted pods still count),
# the total goes to spent.txt. At $92 it writes STOP (rp_run.sh then refuses new pods); at $100 it deletes every glyd-* pod.
# The API key is read into each call's environment only.
D=${GLYD_RUNS:-$HOME/glyd-runs}/rp_budget; mkdir -p "$D"; CAP=${CAP:-100}; SOFT=${SOFT:-92}; L=$D/ledger.tsv; touch "$L"
rp() { RUNPOD_API_KEY="$(tr -d '[:space:]' < ~/.runpod/api_key)" runpodctl "$@"; }
while true; do
  rp pod list 2> /dev/null > "$D/now.json"
  python3 - "$D/now.json" "$L" > "$D/spent.txt" <<'PY'
import json, sys
now, led = sys.argv[1], sys.argv[2]
rows = {}
for line in open(led):
    i, n, c, u = line.rstrip("\n").split("\t"); rows[i] = [n, float(c), float(u)]
try: pods = json.load(open(now))
except Exception: pods = []
for p in pods:
    n = p.get("name") or ""
    if not n.startswith("glyd-"): continue
    i, c, u = p["id"], float(p.get("costPerHr") or 0), float(p.get("uptimeSeconds") or 0) + 120  # (+2 min: creation and the poll gap)
    r = rows.setdefault(i, [n, c, 0.0]); r[1] = max(r[1], c); r[2] = max(r[2], u)
with open(led, "w") as f:
    for i, (n, c, u) in rows.items(): f.write(f"{i}\t{n}\t{c}\t{u}\n")
print(f"{sum(c * u / 3600 for n, c, u in rows.values()):.2f}")
PY
  spent=$(cat "$D/spent.txt"); echo "$(date -u +%H:%M) spent \$$spent" >> "$D/guard.log"
  awk -v s="$spent" -v t="$SOFT" 'BEGIN{exit !(s >= t)}' && [ ! -f "$D/STOP" ] && { echo "soft cap \$$SOFT reached at \$$spent" > "$D/STOP"; echo "$(date -u +%H:%M) STOP written" >> "$D/guard.log"; }
  if awk -v s="$spent" -v t="$CAP" 'BEGIN{exit !(s >= t)}'; then
    python3 -c 'import json,sys; [print(p["id"]) for p in json.load(open(sys.argv[1])) if (p.get("name") or "").startswith("glyd-")]' "$D/now.json" 2> /dev/null | while read -r id; do
      rp pod delete "$id" > /dev/null 2>&1; echo "$(date -u +%H:%M) cap \$$CAP reached: deleted $id" >> "$D/guard.log"; done
  fi
  sleep 60
done
