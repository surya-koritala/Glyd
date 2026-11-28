#!/usr/bin/env python3
"""Image input checks of a running server (tools/serve.sh): pictures drawn here, sent the ways the APIs take them (OpenAI image_url as a base64 data URL,
Anthropic image blocks), each with a question whose answer is known; a second turn about an image already read; an http URL on this computer refused
(the server fetches image URLs from public hosts only). Standard library only.
   tests/test_vision.py [--url http://127.0.0.1:8080] [--save DIR]   (--save: the pictures as PNG files too)"""
import argparse
import base64
import http.server
import json
import os
import struct
import threading
import urllib.error
import urllib.request
import zlib


def png(w, h, px):
    """A PNG of w x h RGB pixels, px(x, y) -> (r, g, b)."""
    rows = b"".join(b"\x00" + bytes(c for x in range(w) for c in px(x, y)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")


WHITE, BLACK = (255, 255, 255), (0, 0, 0)
SEVEN = ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."]   # a 5 x 7 digit


def circles(x, y):
    return BLACK if any((x - cx) ** 2 + (y - cy) ** 2 < 28 ** 2 for cx, cy in [(64, 80), (192, 70), (128, 190)]) else WHITE


def digit(x, y):
    c, r = (x - 48) // 32, (y - 16) // 32
    return BLACK if 0 <= c < 5 and 0 <= r < 7 and SEVEN[r][c] == "#" else WHITE


PICTURES = {
    "red": (256, 256, lambda x, y: (220, 20, 30)),
    "halves": (320, 256, lambda x, y: (30, 60, 220) if x < 160 else (240, 220, 20)),
    "circles": (256, 256, circles),
    "seven": (256, 256, digit),
}


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--save"); a = ap.parse_args()
    u, ok = a.url, []
    pics = {k: png(*v) for k, v in PICTURES.items()}
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for k, b in pics.items():
            open(os.path.join(a.save, k + ".png"), "wb").write(b)

    def check(name, cond, detail=""):
        ok.append(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}"[:200], flush=True)

    def data_url(k):
        return "data:image/png;base64," + base64.b64encode(pics[k]).decode()

    def ask(content, max_tokens=24, history=()):
        r = post(u, "/v1/chat/completions", {"messages": [*history, {"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0})
        return r["choices"][0]["message"]["content"] or "", r

    def picture(k, q):
        return [{"type": "image_url", "image_url": {"url": data_url(k)}}, {"type": "text", "text": q}]

    t, _ = ask(picture("red", "What color is this image? Answer with one word."))
    check("openai data URL: a color", "red" in t.lower(), t)
    t, _ = ask(picture("halves", "What color is the left half of this image? Answer with one word."))
    check("left of right", "blue" in t.lower(), t)
    t, _ = ask(picture("circles", "How many circles are in this image? Answer with a number."))
    check("counting", "3" in t or "three" in t.lower(), t)
    t, _ = ask(picture("seven", "What digit is drawn in this image? Answer with the digit only."))
    check("a digit", "7" in t, t)
    # an http URL on this computer: refused, not fetched (the server fetches image URLs from public hosts only, since any client can name one)
    class One(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.hits += 1; self.send_response(200); self.send_header("Content-Type", "image/png"); self.end_headers(); self.wfile.write(pics["halves"])

        def log_message(self, *args):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), One); srv.hits = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ask([{"type": "image_url", "image_url": {"url": f"http://127.0.0.1:{srv.server_port}/halves.png"}}, {"type": "text", "text": "What color is the right half of this image? Answer with one word."}])
        status, msg = 200, "answered"
    except urllib.error.HTTPError as e:
        status, msg = e.code, e.read().decode(errors="replace")
    srv.shutdown()
    check("a local http URL is refused", status == 400 and "public" in msg and srv.hits == 0, f"{status} hits={srv.hits} {msg[:80]}")
    # two pictures in one message, in order
    t, _ = ask([{"type": "text", "text": "First picture:"}, {"type": "image_url", "image_url": {"url": data_url("seven")}}, {"type": "text", "text": "Second picture:"},
                {"type": "image_url", "image_url": {"url": data_url("red")}}, {"type": "text", "text": "What is the main color of the second picture? Answer with one word."}])
    check("two pictures", "red" in t.lower(), t)
    # a second turn about the picture of the first (the server reads the conversation again; its earlier turn reused where it can)
    first = picture("circles", "How many circles are in this image? Answer with a number.")
    t1, r1 = ask(first)
    t, _ = ask("What color are they? Answer with one word.", history=[{"role": "user", "content": first}, {"role": "assistant", "content": t1}])
    check("second turn", "black" in t.lower(), t)
    r = post(u, "/v1/messages", {"model": "x", "max_tokens": 24, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(pics["red"]).decode()}},
        {"type": "text", "text": "What color is this image? Answer with one word."}]}]})
    t = "".join(b.get("text", "") for b in r.get("content", []))
    check("anthropic image block", "red" in t.lower(), t)
    req = urllib.request.Request(u + "/v1/chat/completions", data=json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,bm90IGFuIGltYWdl"}}]}], "max_tokens": 4}).encode(), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60)
        check("a bad image is refused", False)
    except urllib.error.HTTPError as e:
        check("a bad image is refused", e.code == 400, f"{e.code} {e.read()[:100]!r}")
    print(f"{sum(ok)} of {len(ok)} passed")


if __name__ == "__main__":
    main()
