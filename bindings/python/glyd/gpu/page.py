"""The chat page `glyd run` and `glyd serve` show at http://localhost:8000: one self-contained HTML file (chat.html: no build step,
no external resource), served by vLLM's own app so that it is on the API's origin. vLLM 0.30 adds a class given as --middleware to
its app; this one answers GET and HEAD of / with the page and hands every other request on, untouched:

    vllm serve MODEL --quantization glyd --middleware glyd.gpu.page.ChatPage

A pure ASGI class (not Starlette's BaseHTTPMiddleware), so streamed answers and a client's disconnect, which aborts a request in vLLM,
pass through as they do without it. The page is sent with a Content-Security-Policy that lets it talk to its own origin only.

It is also the server's guard on this computer (GLYD_LOCAL_ONLY=1, which `glyd run` and `glyd serve` set unless the user chose a --host).
It is the outermost middleware, so it sees every request before vLLM's CORS and key checks do. A web page in the user's browser can reach
a server on 127.0.0.1, by CORS (vLLM's default allows any origin) or, with a name that resolves to 127.0.0.1, by DNS rebinding, which
needs no CORS at all. Then a request whose Host is not localhost, 127.0.0.1 or [::1] is refused with 421 (the rebinding page's name),
and a request with an Origin that is not this server's own (http:// and its Host) with 403 (another page's script, a preflight too);
a WebSocket's handshake is checked the same way, as CORS does not cover it. Open WebUI's server sends no Origin and a Host that is an
address, so it passes; so do curl and the OpenAI libraries.
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of Glyd's GPU code.
"""
import json
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
LOOPBACK = ("localhost", "127.0.0.1", "[::1]")


def own_origins(port):
    """The origins this server's own page has, for vLLM's --allowed-origins."""
    return [f"http://{h}:{port}" for h in LOOPBACK]


def _hostname(host):
    """The name in a Host header: localhost:8000 is localhost, [::1]:8000 is [::1]."""
    host = host.strip().lower()
    return host[: host.index("]") + 1] if host.startswith("[") and "]" in host else host.split(":")[0]


def refusal(headers):
    """(status, why) for a request this server answers only to its own page and local programs, or None: a Host that is not a loopback
    name is a DNS-rebinding page's (421), an Origin that is not http:// and the Host is another page's script (403)."""
    host = headers.get(b"host", b"").decode("latin-1")
    origin = headers.get(b"origin", b"").decode("latin-1")
    if host and _hostname(host) not in LOOPBACK:
        return 421, f"this server answers to localhost, 127.0.0.1 and [::1], not to {host[:80]!r} (glyd serve --host sets another address)"
    if origin and origin.lower() != "http://" + host.strip().lower():
        return 403, (f"this server answers to its own page, not to a script from {origin[:80]!r} (to let a web app of your own in: glyd serve MODEL --host 127.0.0.1 "
                     "-- --allowed-origins '[\"http://localhost:5173\"]', which leaves the origins to vLLM)")
    return None


class ChatPage:
    def __init__(self, app):
        self.app = app
        self.local_only = os.environ.get("GLYD_LOCAL_ONLY") == "1"

    async def __call__(self, scope, receive, send):
        if self.local_only and scope["type"] in ("http", "websocket"):
            bad = refusal(dict(scope.get("headers") or ()))
            if bad:
                status, why = bad
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                body = json.dumps({"error": {"message": why, "type": "Forbidden", "code": status}}).encode()
                await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        if scope["type"] == "http" and scope["method"] in ("GET", "HEAD") and scope["path"] == "/":
            await send({"type": "http.response.start", "status": 200, "headers": HEADERS + [(b"content-length", str(len(PAGE)).encode())]})
            await send({"type": "http.response.body", "body": PAGE if scope["method"] == "GET" else b""})
            return
        await self.app(scope, receive, send)
