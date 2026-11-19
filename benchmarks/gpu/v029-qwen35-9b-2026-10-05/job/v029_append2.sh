
# ---- appended 2: glyd serve with vLLM compiled (vLLM's own default) in place of the eager default, the same models ----
gvlog() { local f; f=$(ls -t /root/.local/state/glyd/logs/serve-*.log 2> /dev/null | head -1); grep -h -o -E "Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Enforce eager set|glyd: [a-z0-9]+ layout" "$f" 2> /dev/null | sort -u | tr '\n' ';'; }
serve glydc8b 8006 glyd serve Qwen/Qwen3-8B --port 8006 -- --no-enforce-eager
[ -n "${SRV:-}" ] && { step speed_glydc8b 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8006 Qwen/Qwen3-8B "$R/speed-glydcompiled-qwen3-8b.json"; echo "glyd serve compiled Qwen3-8B: $(gvlog)" >> "$R/numbers.txt"; stop; }
serve glydc_q35 8007 glyd serve Qwen/Qwen3.5-9B --port 8007 -- --no-enforce-eager
[ -n "${SRV:-}" ] && { step speed_glydc_q35 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8007 Qwen/Qwen3.5-9B "$R/speed-glydcompiled-qwen3.5-9b.json"; echo "glyd serve compiled Qwen3.5-9B: $(gvlog)" >> "$R/numbers.txt"; stop; }
check "the GPU is idle after the compiled runs" "$(gpu_idle)"
cp /root/.local/state/glyd/logs/*.log "$R/" 2> /dev/null
python3 - "$R" > "$R/speed.txt" 2>&1 <<'EOF'
import json, os, sys
R = sys.argv[1]
for f in sorted(os.listdir(R)):
    if f.startswith("speed-") and f.endswith(".json"):
        d = json.load(open(os.path.join(R, f))); print(f"{f[6:-5]:32} TTFT {d['ttft_ms_median']:8.1f} ms  decode {d['decode_tok_s_median']:7.2f} tok/s")
EOF
{ echo "v0.29 pre-release run on $GPU (with both appended blocks)"; echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"; for f in "${FAIL[@]}"; do echo "FAIL $f"; done
  echo "--- one user's speed"; cat "$R/speed.txt"; echo "--- numbers"; cat "$R/numbers.txt" 2> /dev/null; echo "--- times"; cat "$R/times.txt" 2> /dev/null; } > "$R/summary2.txt"
cat "$R/summary2.txt" | tee -a "$R/job.log"
