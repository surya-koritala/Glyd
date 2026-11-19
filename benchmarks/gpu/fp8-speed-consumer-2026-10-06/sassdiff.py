import re, sys
def funcs(path):
    out, cur, buf = {}, None, []
    for line in open(path):
        m = re.match(r'\s*Function : (\S+)', line)
        if m:
            if cur: out[cur] = buf
            cur, buf = m.group(1), []
            continue
        if cur is None: continue
        t = re.sub(r'/\*[0-9a-f]{4,}\*/', '', line)          # addresses
        t = re.sub(r'/\* 0x[0-9a-f]+ \*/', '', t).strip()    # encodings
        if t: buf.append(t)
    if cur: out[cur] = buf
    return out
a, b = funcs(sys.argv[1]), funcs(sys.argv[2])
f8 = lambda n: 'f8c' in n
same = [n for n in a if n in b and a[n] == b[n] and not f8(n)]
diff = [n for n in a if n in b and a[n] != b[n] and not f8(n)]
only = [n for n in set(a) ^ set(b) if not f8(n)]
print(f"{sys.argv[1]} vs {sys.argv[2]}: non-f8 functions {len([n for n in a if not f8(n)])} / {len([n for n in b if not f8(n)])}; identical SASS {len(same)}; different {len(diff)}; in one only {len(only)}")
for n in diff[:20]: print("  differs:", n, len(a[n]), len(b[n]))
for n in only[:20]: print("  one only:", n)
