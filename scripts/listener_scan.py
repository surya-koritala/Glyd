#!/usr/bin/env python3
"""Scan a running Glyd for listeners that are not on this computer's own address (Linux: /proc).

    python3 scripts/listener_scan.py PID [PID ...]

Lists every TCP socket in LISTEN state that the processes, and the processes they started, hold. Exit 0 if every one is on a loopback address (127.0.0.0/8, ::1);
exit 1 and the offenders where one is on 0.0.0.0, :: or a network address. Run by the `gpu` job against a default `glyd serve` (tests/ops_check.py), and the engine's tests
do the same for the default `glyd service` (glyd-engine/tests/listeners.rs). Standard library only.
"""

import ipaddress
import os
import sys


def children(pid, seen):
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            for c in open(f"/proc/{pid}/task/{t}/children").read().split():
                if int(c) not in seen:
                    seen.add(int(c))
                    children(int(c), seen)
        except OSError:
            pass


def inodes(pid):
    out = set()
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            link = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if link.startswith("socket:["):
            out.add(int(link[8:-1]))
    return out


def decode(hexaddr, v6):
    if not v6:
        return str(ipaddress.IPv4Address(bytes.fromhex(hexaddr)[::-1]))
    raw = bytes.fromhex(hexaddr)
    return str(ipaddress.IPv6Address(b"".join(raw[i : i + 4][::-1] for i in range(0, 16, 4))))


def listening():
    """{inode: (ip, port)} of every TCP socket in LISTEN state on this computer."""
    out = {}
    for name, v6 in (("tcp", False), ("tcp6", True)):
        try:
            lines = open(f"/proc/net/{name}").read().splitlines()[1:]
        except OSError:
            continue
        for l in lines:
            f = l.split()
            if f[3] == "0A":  # LISTEN
                addr, port = f[1].split(":")
                out[int(f[9])] = (decode(addr, v6), int(port, 16))
    return out


def scan(pids):
    """[(pid, ip, port)] of the listeners the processes and their children hold."""
    every = set(pids)
    for p in pids:
        children(p, every)
    table = listening()
    found = []
    for p in sorted(every):
        try:
            for i in inodes(p):
                if i in table:
                    found.append((p, *table[i]))
        except OSError:
            pass
    return found


def loopback(ip):
    a = ipaddress.ip_address(ip)
    return a.is_loopback or (a.version == 6 and a.ipv4_mapped is not None and a.ipv4_mapped.is_loopback)


if __name__ == "__main__":
    pids = [int(x) for x in sys.argv[1:]]
    found = scan(pids)
    for p, ip, port in found:
        print(f"pid {p} listens on {ip}:{port}" + ("" if loopback(ip) else "   <-- not this computer's own address"))
    bad = [f for f in found if not loopback(f[1])]
    if not found:
        print("no listener found (is the process up?)")
        sys.exit(2)
    sys.exit(1 if bad else 0)
