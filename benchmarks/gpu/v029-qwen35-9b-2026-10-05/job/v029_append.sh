
# ---- appended during the run: the Qwen3-8B baselines again (port 8001 is Runpod's nginx; vLLM run by path needs its bin on PATH for ninja) ----
serve vllm8b_b 8002 env PATH="$TOOL/bin:$PATH" "$TOOL/bin/vllm" serve Qwen/Qwen3-8B --port 8002 --max-model-len 8192
[ -n "${SRV:-}" ] && { step speed_vllm8b_b 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8002 Qwen/Qwen3-8B "$R/speed-vllm-qwen3-8b.json"; echo "vllm bf16 Qwen3-8B: $(vlog vllm8b_b)" >> "$R/numbers.txt"; stop; }
serve glydvllm8b_b 8005 glyd serve Qwen/Qwen3-8B --port 8005
[ -n "${SRV:-}" ] && { check "glyd serve Qwen3-8B (default, port 8005): vLLM runs it" "$(has 'vLLM' "$R/glydvllm8b_b.serve.err")" "$(grep '^glyd serve:' "$R/glydvllm8b_b.serve.err" | head -2)"
    step speed_glydvllm8b_b 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8005 Qwen/Qwen3-8B "$R/speed-glydvllm-qwen3-8b.json"; echo "glyd serve Qwen3-8B: $(vlog glydvllm8b_b)" >> "$R/numbers.txt"; stop; }
check "the GPU is idle at the very end" "$(gpu_idle)"
python3 - "$R" > "$R/speed.txt" 2>&1 <<'EOF'
import json, os, sys
R = sys.argv[1]
for f in sorted(os.listdir(R)):
    if f.startswith("speed-") and f.endswith(".json"):
        d = json.load(open(os.path.join(R, f))); print(f"{f[6:-5]:28} TTFT {d['ttft_ms_median']:8.1f} ms  decode {d['decode_tok_s_median']:7.2f} tok/s  prompt {d['rounds'][0].get('prompt_tokens')}  rounds {[r['decode_tok_s'] for r in d['rounds']]}")
EOF
{ echo "v0.29 pre-release run on $GPU (with the appended baselines)"; echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"; for f in "${FAIL[@]}"; do echo "FAIL $f"; done; for p in "${PASS[@]}"; do echo "ok   $p"; done
  echo "--- one user's speed"; cat "$R/speed.txt"
  echo "--- numbers"; cat "$R/numbers.txt" 2> /dev/null; echo "--- sizes"; cat "$R/sizes.txt" 2> /dev/null; echo "--- times"; cat "$R/times.txt" 2> /dev/null; } > "$R/summary.txt"
cat "$R/summary.txt" | tee -a "$R/job.log"
