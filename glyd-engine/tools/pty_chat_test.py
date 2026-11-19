"""The terminal chat in a pseudo-terminal: python pty_chat_test.py ENGINE MODEL [extra args...]

Starts `ENGINE run MODEL ...` on a terminal, waits for its prompt, asks for a long story, presses Ctrl-C in the middle of the answer (the answer stops, the prompt comes back), asks a short question,
leaves with /bye (exit status 0), and starts it again to leave with Ctrl-D and with Ctrl-C at the prompt (exit status 130)."""
import os
import pty
import select
import signal
import sys
import termios
import time

ENGINE, MODEL, EXTRA = sys.argv[1], sys.argv[2], sys.argv[3:]
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (" -- " + str(detail)[-200:].replace("\n", "|") if detail != "" else ""))
    if not ok:
        fails.append(name)


class Session:
    def __init__(self, args=None):
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execv(ENGINE, [ENGINE, *(args or ["run", MODEL, "--no-think", *EXTRA])])
        self.buf = ""

    def echo(self):
        """Whether the terminal echoes what is typed (None where it cannot be read)."""
        try:
            return bool(termios.tcgetattr(self.fd)[3] & termios.ECHO)
        except Exception:
            return None

    def read_until(self, text, timeout):
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], 0.2)
            if r:
                try:
                    data = os.read(self.fd, 65536)
                except OSError:
                    return False
                if not data:
                    return False
                self.buf += data.decode(errors="replace")
                if text in self.buf:
                    return True
        return text in self.buf

    def send(self, s):
        os.write(self.fd, s.encode())

    def mark(self):
        self.buf = ""

    def wait_exit(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            p, status = os.waitpid(self.pid, os.WNOHANG)
            if p:
                return os.WEXITSTATUS(status) if os.WIFEXITED(status) else 128 + os.WTERMSIG(status)
            time.sleep(0.1)
        os.kill(self.pid, signal.SIGKILL)
        return None


s = Session()
check("the prompt comes up", s.read_until("> ", 600), s.buf)
s.mark()
s.send("Write a long story about a fox who learns to read.\n")
time.sleep(4)
s.send("\x03")  # Ctrl-C in the middle of the answer
check("Ctrl-C stops the answer and the prompt comes back", s.read_until("stopped", 30) and s.read_until("> ", 10), s.buf)
s.mark()
s.send("What is 2+2? Answer in one word.\n")
check("a question after it is answered", s.read_until("tokens/s", 60), s.buf)
s.mark()
s.send("/bye\n")
check("/bye leaves with status 0", s.wait_exit(10) == 0)
t = Session()
check("the prompt comes up again", t.read_until("> ", 600))
t.send("\x04")  # Ctrl-D
check("Ctrl-D leaves with status 0", t.wait_exit(10) == 0)
u = Session()
check("the prompt comes up a third time", u.read_until("> ", 600))
u.send("\x03")  # Ctrl-C at the prompt
check("Ctrl-C at the prompt leaves (130)", u.wait_exit(10) == 130)
if os.environ.get("PTY_SKIP_LOGIN"):
    print("login not tried (PTY_SKIP_LOGIN)")
    print("FAILED: " + ", ".join(fails) if fails else "all ok")
    sys.exit(1 if fails else 0)
# glyd login: the token is not echoed while it is typed, and Ctrl-C puts the terminal back as it was
v = Session(["login"])
check("login asks for a token", v.read_until("not shown", 30), v.buf)
time.sleep(0.5)
check("the terminal does not echo while the token is typed", v.echo() is False, v.echo())
v.send("\x03")
check("Ctrl-C leaves login (130)", v.wait_exit(10) == 130)
check("and the terminal echoes again", v.echo() in (True, None), v.echo())
w = Session(["login"])
w.read_until("not shown", 30)
time.sleep(0.5)
w.send("hf_not_a_token\n")
check("a token the Hub does not accept is refused in words", w.read_until("did not accept", 60), w.buf)
check("login leaves with status 1", w.wait_exit(10) == 1)
print("FAILED: " + ", ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
