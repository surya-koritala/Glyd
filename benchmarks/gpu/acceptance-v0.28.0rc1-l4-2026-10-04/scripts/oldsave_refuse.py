"""With glyd-gpu 0.27.0: a glyd-v4 save refused, and how."""
import sys
import glyd

try:
    glyd.from_pretrained(sys.argv[1])
except Exception as e:
    msg = f"{type(e).__name__}: {e}"
    print("refused:", msg, flush=True)
    sys.exit(0 if isinstance(e, ValueError) and "glyd-v4" in str(e) and "this glyd reads" in str(e) else 1)
print("NOT REFUSED", flush=True)
sys.exit(2)
