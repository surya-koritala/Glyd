"""Frames (PNG) into a GIF: makegif.py FRAMEDIR OUT.gif FPS WIDTH [KEEP_EVERY] — scaled to WIDTH, one palette for all frames, the last held 2 s."""
import sys, glob
from PIL import Image
d, out, fps, width = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]); every = int(sys.argv[5]) if len(sys.argv) > 5 else 1
files = sorted(glob.glob(f"{d}/f*.png"))[::every]
frames = []
for f in files:
    im = Image.open(f).convert("RGB")
    frames.append(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS))
pal = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
dur = [round(1000 * every / fps)] * len(q); dur[-1] = 2000
q[0].save(out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True, disposal=1)
import os; print(f"{out}: {len(q)} frames, {os.path.getsize(out) / 1e6:.1f} MB")
