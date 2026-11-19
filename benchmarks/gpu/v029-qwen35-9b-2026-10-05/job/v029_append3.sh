
# ---- appended 3: plain vLLM where it did not start (its FlashInfer sampler on this pip-only setup): VLLM_USE_FLASHINFER_SAMPLER=0, as glyd sets it ----
if [ ! -s "$R/speed-vllm-qwen3-8b.json" ]; then
  serve vllm8b_c 8008 env VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$TOOL/bin:$PATH" "$TOOL/bin/vllm" serve Qwen/Qwen3-8B --port 8008 --max-model-len 8192
  [ -n "${SRV:-}" ] && { step speed_vllm8b_c 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8008 Qwen/Qwen3-8B "$R/speed-vllm-qwen3-8b.json"; echo "vllm bf16 Qwen3-8B (sampler off): $(vlog vllm8b_c)" >> "$R/numbers.txt"; stop; }
fi
if [ ! -s "$R/speed-vllm-qwen3.5-9b.json" ]; then
  serve vllm_q35_c 8009 env VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$TOOL/bin:$PATH" "$TOOL/bin/vllm" serve Qwen/Qwen3.5-9B --port 8009 --max-model-len 4096
  [ -n "${SRV:-}" ] && { step speed_vllm_q35_c 900 "$PY" "$HOME/speed1.py" http://127.0.0.1:8009 Qwen/Qwen3.5-9B "$R/speed-vllm-qwen3.5-9b.json"; echo "vllm bf16 Qwen3.5-9B (sampler off): $(vlog vllm_q35_c)" >> "$R/numbers.txt"; stop; }
fi
python3 - "$R" > "$R/speed.txt" 2>&1 <<'EOF'
import json, os, sys
R = sys.argv[1]
for f in sorted(os.listdir(R)):
    if f.startswith("speed-") and f.endswith(".json"):
        d = json.load(open(os.path.join(R, f))); print(f"{f[6:-5]:32} TTFT {d['ttft_ms_median']:8.1f} ms  decode {d['decode_tok_s_median']:7.2f} tok/s")
EOF
{ echo "v0.29 pre-release run on $GPU (all three appended blocks)"; echo "passed ${#PASS[@]}, failed ${#FAIL[@]}"; for f in "${FAIL[@]}"; do echo "FAIL $f"; done
  echo "--- one user's speed"; cat "$R/speed.txt"; echo "--- numbers"; cat "$R/numbers.txt" 2> /dev/null; echo "--- times"; cat "$R/times.txt" 2> /dev/null; } > "$R/summary3.txt"
cat "$R/summary3.txt" | tee -a "$R/job.log"
