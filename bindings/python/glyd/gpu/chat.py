"""A terminal chat with an OpenAI-compatible server (the one `glyd run` starts): streamed answers, thinking shown apart, /clear and
/bye, and a plain message where the conversation has outgrown the model's window. The standard library only.

    api = Api("http://127.0.0.1:8000")
    chat = Chat(api, *api.model())      # the model's name and window, from /v1/models
    chat.loop()                         # the terminal's chat
    chat.once("Say hello")              # one answer on stdout (thinking on stderr)
Under the Business Source License 1.1 (LICENSE-glyd-gpu), as the rest of Glyd's GPU code.
"""
import http.client
import json
import os
import re
import sys
import urllib.parse

HELP = """  /bye           leave (or Ctrl-D)
  /clear         start a new chat
  /think         turn the model's thinking on or off
  \"\"\"            start and end a message of several lines"""


def full_message(window, chat=True):
    """A conversation (or, for one answer, a prompt) longer than the model's window, and what to do."""
    size = f" ({window:,} tokens)" if window else ""
    if not chat:
        return f"This prompt is longer than the model's window{size}. Shorten it, or start glyd with a larger --context."
    return f"This conversation is longer than the model's window{size}. Start a new chat with /clear."


class ApiError(Exception):
    """The server's refusal (an HTTP status and its message)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message

    @property
    def context_full(self):
        return bool(re.search(r"maximum context length|max_model_len|context (length|window)", self.message, re.I))


class Api:
    def __init__(self, base, timeout=3600):
        u = urllib.parse.urlsplit(base)
        self.host, self.port, self.timeout = u.hostname, u.port or 80, timeout

    def _conn(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)

    def get(self, path, timeout=5):
        c = self._conn(timeout)
        try:
            c.request("GET", path)
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                raise ApiError(r.status, _message(body))
            return json.loads(body)
        finally:
            c.close()

    def model(self):
        """(the served model's name, its window in tokens: 0 where the server does not say)."""
        d = self.get("/v1/models")["data"][0]
        return d["id"], int(d.get("max_model_len") or 0)

    def stream(self, model, messages, think=True):
        """The server's answer to a chat, streamed: ("reasoning" | "content", text), then ("usage", total tokens) and ("finish", reason).
        An ApiError where the server refuses; the connection is closed where the generator is (Ctrl-C aborts the request)."""
        body = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": bool(think)}}
        c = self._conn()
        try:
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                raise ApiError(r.status, _message(r.read()))
            while True:
                line = r.readline()
                if not line:
                    return
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                d = json.loads(data)
                if d.get("usage"):
                    yield "usage", int(d["usage"].get("total_tokens") or 0)
                for ch in d.get("choices") or ():
                    delta = ch.get("delta") or {}
                    think_text = delta.get("reasoning_content") or delta.get("reasoning")
                    if think_text:
                        yield "reasoning", think_text
                    if delta.get("content"):
                        yield "content", delta["content"]
                    if ch.get("finish_reason"):
                        yield "finish", ch["finish_reason"]
        finally:
            c.close()


def _message(body):
    """The message in a server's error body (vLLM's {"error": {"message": ...}}, or the older top-level "message")."""
    try:
        d = json.loads(body)
        return (d.get("error") or {}).get("message") or d.get("message") or str(d)
    except (ValueError, AttributeError):
        return body.decode("utf-8", "replace")[:300]


class Think:
    """Splits a stream's content into thinking and answer where the model writes <think>...</think> inline first (a server with no
    reasoning parser): feed() gives [("reasoning" | "content", text)]. Text that may still be the start of a tag is held back."""

    def __init__(self):
        self.mode, self.buf, self.lead = "start", "", False

    def feed(self, text, final=False):
        self.buf += text
        out = []
        while True:
            if self.mode == "start":
                s = self.buf.lstrip()
                if s.startswith("<think>"):
                    self.buf, self.mode = s[7:].lstrip("\n"), "inside"
                    continue
                if not final and "<think>".startswith(s):  # (the very start, which may still turn into the tag)
                    break
                self.mode = "done"
            elif self.mode == "inside":
                i = self.buf.find("</think>")
                if i >= 0:
                    if self.buf[:i]:
                        out.append(("reasoning", self.buf[:i]))
                    self.buf, self.mode, self.lead = self.buf[i + 8:], "done", True
                    continue
                hold = 0 if final else next((k for k in range(7, 0, -1) if self.buf.endswith("</think>"[:k])), 0)  # (a closing tag begun)
                if self.buf[: len(self.buf) - hold]:
                    out.append(("reasoning", self.buf[: len(self.buf) - hold]))
                self.buf = self.buf[len(self.buf) - hold:]
                break
            else:
                if self.lead:
                    self.buf = self.buf.lstrip("\n")
                    self.lead = not self.buf
                if self.buf:
                    out.append(("content", self.buf))
                    self.buf = ""
                break
        return out


class Chat:
    def __init__(self, api, model, window=0, think=True, out=None, err=None, log=""):
        self.api, self.model, self.window, self.think = api, model, window, think
        self.out, self.err, self.log = out or sys.stdout, err or sys.stderr, log
        self.messages, self.used, self.interactive = [], 0, True
        self.color = "NO_COLOR" not in os.environ

    def dim(self, text, err=False):
        """Text in the terminal's dim style (nothing added where the stream is not a terminal)."""
        return f"\x1b[2m{text}\x1b[0m" if self.color and (self.err if err else self.out).isatty() else text

    def say(self, text="", end="\n", err=False):
        f = self.err if err else self.out
        f.write(text + end)
        f.flush()

    def turn(self, text, to_err=False):
        """One exchange: the question appended, the answer streamed and appended. Returns the answer; "" where it failed or was
        stopped (the question is then left out of the conversation, as if unasked)."""
        self.messages.append({"role": "user", "content": text})
        answer, thinking, said_think, think, finish, used = [], False, False, Think(), None, self.used

        def show(kind, piece):
            nonlocal thinking, said_think
            if kind == "reasoning":
                if not said_think:  # (the newline after <think>, and the ones after </think> below, are the template's, not text)
                    piece = piece.lstrip("\n")
                    if not piece:
                        return
                    said_think = thinking = True
                    self.say(self.dim("Thinking...", to_err), err=to_err)
                self.say(self.dim(piece, to_err), end="", err=to_err)
            else:
                if not answer:
                    piece = piece.lstrip("\n")
                    if not piece:
                        return
                if thinking:
                    thinking = False
                    self.say("\n" + self.dim("...done thinking.", to_err) + "\n", err=to_err)
                self.say(piece, end="")
                answer.append(piece)

        try:
            for kind, value in self.api.stream(self.model, self.messages, self.think):
                if kind == "usage":
                    used = value
                elif kind == "finish":
                    finish = value
                elif kind == "reasoning":
                    show("reasoning", value)
                else:
                    for k, piece in think.feed(value):
                        show(k, piece)
            for k, piece in think.feed("", final=True):
                show(k, piece)
        except ApiError as e:
            self.messages.pop()
            self.say("\n" + (full_message(self.window, self.interactive) if e.context_full else f"The server refused the request: {e.message}"), err=True)
            return ""
        except KeyboardInterrupt:
            self.messages.pop()
            self.say("\n(stopped)", err=True)
            return ""
        except (OSError, http.client.HTTPException) as e:
            self.messages.pop()
            self.say(f"\nThe server stopped answering ({e})." + (f" See its log: {self.log}" if self.log else ""), err=True)
            return ""
        reply = "".join(answer)
        if thinking:
            self.say("\n" + self.dim("...done thinking.", to_err), err=to_err)
        if reply:
            self.say()
        self.messages.append({"role": "assistant", "content": reply})
        self.used = used
        if finish == "length":
            what = "The answer was cut off" if reply else "The model was still thinking"
            advice = ("Start a new chat with /clear" + ("" if reply else ", or turn thinking off with /think")) if self.interactive else "Ask for less, or start glyd with a larger --context"
            self.say(f"\n({what}: the conversation reached the model's window" + (f" ({self.window:,} tokens)" if self.window else "") + f". {advice}.)", err=True)
        elif self.window and self.used >= 0.8 * self.window:
            self.say(self.dim(f"({self.used:,} of {self.window:,} tokens of this conversation used; /clear starts a new chat)", True), err=True)
        return reply

    def once(self, prompt):
        """One answer for a prompt: the answer on stdout, the thinking and any notice on stderr. Returns the exit status."""
        self.interactive = False
        return 0 if self.turn(prompt, to_err=True) else 1

    def loop(self, read=input):
        """The terminal's chat: >>> prompts until /bye or Ctrl-D."""
        try:
            import readline  # noqa: F401  (line editing and history for input())
        except ImportError:
            pass
        self.say(f"Chatting with {self.model}. /bye to leave, /clear for a new chat, /? for more.")
        while True:
            try:
                text = read(">>> ").strip()
                if text.startswith('"""'):
                    lines = [text[3:]]
                    while not lines[-1].rstrip().endswith('"""'):
                        lines.append(read("... "))
                    text = "\n".join(lines).rstrip()[:-3].strip()
            except EOFError:
                self.say()
                return 0
            except KeyboardInterrupt:
                self.say("\n(/bye or Ctrl-D to leave)")
                continue
            if not text:
                continue
            if text in ("/bye", "/exit", "/quit"):
                return 0
            if text == "/clear":
                self.messages, self.used = [], 0
                self.say("Started a new chat.")
            elif text == "/think":
                self.think = not self.think
                self.say(f"Thinking {'on' if self.think else 'off'}.")
            elif text in ("/?", "/help", "/h"):
                self.say(HELP)
            elif text.startswith("/") and " " not in text:
                self.say(f"Unknown command {text}. /? lists them.")
            else:
                self.turn(text)
                self.say()
