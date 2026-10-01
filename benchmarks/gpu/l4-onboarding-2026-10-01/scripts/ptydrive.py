"""ptydrive.py OUT [PROMPT=REPLY ...] -- COMMAND...: COMMAND in a pty (100 columns), everything it writes kept in OUT; each time its output
(since the last reply) ends with PROMPT, REPLY and a newline are typed, after a person's pause. Exits with the command's status."""
import fcntl
import os
import pty
import select
import struct
import sys
import termios
import time


def main():
    out, i = sys.argv[1], sys.argv.index("--")
    expects = [a.split("=", 1) for a in sys.argv[2:i]]
    cmd = sys.argv[i + 1:]
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.execvp(cmd[0], cmd)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    log, since, step = open(out, "wb"), b"", 0
    while True:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if ready:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            log.write(data)
            log.flush()
            since += data
        if step < len(expects) and since.endswith(expects[step][0].encode()):
            time.sleep(1.5)
            os.write(fd, (expects[step][1] + "\n").encode())
            since, step = b"", step + 1
    _, status = os.waitpid(pid, 0)
    sys.exit(os.waitstatus_to_exitcode(status))


main()
