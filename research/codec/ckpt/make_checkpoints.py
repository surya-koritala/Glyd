"""Real training checkpoints with optimizer state, for measuring: a
causal LM fine-tuned with AdamW (fp32 master weights, both moments) on
real text, `torch.save`d every `--every` steps as training code does:
{"model": state_dict, "optimizer": state_dict, "step": n}.

    python make_checkpoints.py MODEL_DIR TEXT_FILE OUT_DIR [--steps 250] [--every 50]
"""
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model"), ap.add_argument("text"), ap.add_argument("out")
ap.add_argument("--steps", type=int, default=250)
ap.add_argument("--every", type=int, default=50)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=4)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(args.model)
# fp32 master weights, as mixed-precision training keeps them; bf16 compute.
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).cuda()
model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.1)
text = open(args.text, errors="ignore").read(64 << 20)
ids = tok(text, return_tensors="pt").input_ids[0]
per = args.seq * args.batch
for step in range(1, args.steps + 1):
    at = ((step - 1) * per) % (ids.numel() - per - 1)
    batch = ids[at : at + per].view(args.batch, args.seq).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch, labels=batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 10 == 0:
        print(f"step {step} loss {loss.item():.3f}", flush=True)
    if step % args.every == 0:
        path = os.path.join(args.out, f"step{step:05d}.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}, path)
        print(f"saved {path} {os.path.getsize(path) / 1e9:.2f} GB", flush=True)
