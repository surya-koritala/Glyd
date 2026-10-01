"""render.py TYPESCRIPT: a terminal's bytes as the text its screen ended with: \r returns to the column 0 (the next text overwrites), ESC[K clears
to the end of the line, other escape sequences are dropped, a line that is only a redrawn status is shown in its last state."""
import re
import sys

raw = open(sys.argv[1], "rb").read().decode("utf-8", "replace").replace("\r\n", "\n")
raw = re.sub(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", "", raw)  # (window titles)
lines = []
for line in raw.split("\n"):
    cur = []  # (the screen's line as a list of characters)
    col = 0
    for m in re.finditer(r"\x1b\[([0-9;?]*)([A-Za-z])|\r|([^\x1b\r]+)|\x1b.", line):
        if m.group(2):
            if m.group(2) == "K":
                del cur[col:]
            elif m.group(2) == "G":
                col = max(0, int(m.group(1) or 1) - 1)
            elif m.group(2) == "C":
                col += int(m.group(1) or 1)
            elif m.group(2) == "D":
                col = max(0, col - int(m.group(1) or 1))
        elif m.group(0) == "\r":
            col = 0
        elif m.group(3):
            for ch in m.group(3):
                if col < len(cur):
                    cur[col] = ch
                else:
                    cur.extend(" " * (col - len(cur)))
                    cur.append(ch)
                col += 1
    lines.append("".join(cur).rstrip())
print("\n".join(lines).rstrip())
