"""The chat page `glyd run` and `glyd serve` show at http://localhost:8000: one self-contained HTML file (chat.html: no build step,
no external resource), served by vLLM's own app so that it is on the API's origin. vLLM 0.30 adds a class given as --middleware to
its app; this one answers GET and HEAD of / with the page and hands every other request on, untouched:

    vllm serve MODEL --quantization glyd --middleware glyd.gpu.page.ChatPage

A pure ASGI class (not Starlette's BaseHTTPMiddleware), so streamed answers and a client's disconnect, which aborts a request in vLLM,
pass through as they do without it. The page is sent with a Content-Security-Policy that lets it talk to its own origin only.
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of Glyd's GPU code.
"""
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html"), "rb") as _f:
    PAGE = _f.read()

HEADERS = [
    (b"content-type", b"text/html; charset=utf-8"),
    (b"cache-control", b"no-store"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"content-security-policy", b"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
]


class ChatPage:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ("GET", "HEAD") and scope["path"] == "/":
            await send({"type": "http.response.start", "status": 200, "headers": HEADERS + [(b"content-length", str(len(PAGE)).encode())]})
            await send({"type": "http.response.body", "body": PAGE if scope["method"] == "GET" else b""})
            return
        await self.app(scope, receive, send)
