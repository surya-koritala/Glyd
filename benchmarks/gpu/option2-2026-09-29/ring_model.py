"""A discrete-event model of the route SPLIT's ring (gpu/glyd_gpu.cu: ring_pump, ring_restart, ring_queue and
mma12_ring_linear, as written) on CUDA's stream and event rules: a stream runs its operations in order, a wait takes the
event's last record at the time of the wait (none: no wait), a record is done when its stream reaches it. Random
kernel times, 3-16 slots, chunk sizes, lengths that change the split, a call off the queue now and then, the order
queued whole or a few ahead: every slot written only after its last reader's product ended, and read only after its
decode ended and before the next decode there began. Taking out either wait (the decode's on its slot's last product,
the product's on its decode) fails it.

    python ring_model.py [SEED] [--without free|ready]      (300 schedules from SEED; --without: that wait taken out,
                                                             the schedules that then fail counted)
"""
import random, sys

WITHOUT = sys.argv[sys.argv.index("--without") + 1] if "--without" in sys.argv else ""

class Rec:  # one record of an event: done when its stream reaches it
    def __init__(self): self.done = None  # time

class Stream:
    def __init__(self, name): self.name, self.ops = name, []
    def wait(self, ev): self.ops.append(("wait", ev.last))
    def record(self, ev): r = Rec(); ev.last = r; self.ops.append(("rec", r))
    def kernel(self, what, dur): self.ops.append(("k", what, dur))

class Event:
    def __init__(self): self.last = None

def run(streams):
    """Execute all streams' ops (as a GPU would): returns the kernels' intervals [(what, start, end)]."""
    t = {s.name: 0.0 for s in streams}; i = {s.name: 0 for s in streams}; out = []
    progress = True
    while progress:
        progress = False
        for s in streams:
            while i[s.name] < len(s.ops):
                op = s.ops[i[s.name]]
                if op[0] == "wait":
                    if op[1] is None: pass
                    elif op[1].done is None: break
                    else: t[s.name] = max(t[s.name], op[1].done)
                elif op[0] == "rec": op[1].done = t[s.name]
                else: out.append((op[1], t[s.name], t[s.name] + op[2])); t[s.name] += op[2]
                i[s.name] += 1; progress = True
    stuck = [s.name for s in streams if i[s.name] < len(s.ops)]
    assert not stuck, f"deadlock: {stuck}"
    return out

class Ring:  # the C ring, as written
    STARTS = 64
    def __init__(self, slots, slot_rows):
        self.slots, self.slot_rows = slots, slot_rows
        self.cur = None; self.q = []; self.issued = 0; self.next = 0; self.seq = 0; self.started = -1; self.last = {}
        self.busy = [False] * slots; self.read = [False] * slots
        self.ready = [Event() for _ in range(slots)]; self.free_ = [Event() for _ in range(slots)]
        self.start = [Event() for _ in range(self.STARTS)]; self.mark = Event()
        self.parts = {}
    def part(self, sms):
        if sms not in self.parts: self.parts[sms] = (Stream(f"sd{sms}"), Stream(f"sg{sms}"))
        return self.parts[sms]
    def pump(self, dur):
        sd, sg = self.cur
        while self.issued < len(self.q) and not self.busy[self.next] and self.q[self.issued]["gate"] <= self.started:
            c = self.q[self.issued]; s = self.next
            if self.read[s] and WITHOUT != "free": sd.wait(self.free_[s])
            if c["gate"] >= 0 and self.started - c["gate"] < self.STARTS: sd.wait(self.start[c["gate"] % self.STARTS])
            sd.kernel(("dec", s, c["id"]), dur(c))
            sd.record(self.ready[s])
            c["slot"] = s; self.busy[s] = True; self.next = (s + 1) % self.slots; self.issued += 1
    def join(self, part, cs):
        for st in part: st.record(self.mark); cs.wait(self.mark)
    def restart(self, part, cs):
        if self.cur:
            self.join(self.cur, cs)
            if self.cur is not part:
                for st in part: self.join(self.cur, st)
        self.q = []; self.last = {}; self.issued = 0; self.busy = [False] * self.slots; self.cur = part
    def queue(self, W, dur):
        O = W["O"]; per = min(O, self.slot_rows[W["K"]]); n = -(-O // per); rows = (-(-O // n) + 63) // 64 * 64
        for r0 in range(0, O, rows):
            k = (O, W["K"], r0, min(rows, O - r0)); gate = self.last.get(k, -1)
            self.q.append(dict(W=W, row0=r0, rows=min(rows, O - r0), slot=-1, seq=self.seq, gate=gate, id=(W["name"], r0, self.seq)))
            self.last[k] = self.seq; self.seq += 1
        self.pump(dur)
    def api_queue(self, sms, W, dur, cs):
        part = self.part(sms)
        if part is not self.cur:
            assert not self.q
            self.restart(part, self.cur[0] if self.cur else part[0])
        self.queue(W, dur)
    def reset(self, cs):
        self.restart(self.cur, cs)
    def linear(self, sms, W, cs, dur, gemm_dur):
        part = self.part(sms); sd, sg = part
        nxt = self.cur is part and self.q and self.q[0]["W"] is W and self.q[0]["row0"] == 0
        if not nxt:
            self.restart(part, cs); self.queue(W, dur)
        cs.record(self.mark); sg.wait(self.mark)
        first = True
        while self.q and self.q[0]["W"] is W and (first or self.q[0]["row0"]):
            first = False
            self.pump(dur)
            c = dict(self.q[0])
            assert c["slot"] >= 0, "front not issued"
            if WITHOUT != "ready": sg.wait(self.ready[c["slot"]])
            sg.record(self.start[c["seq"] % self.STARTS]); self.started = c["seq"]
            self.pump(dur)
            sg.kernel(("gemm", c["slot"], c["id"]), gemm_dur(c))
            sg.record(self.free_[c["slot"]])
            self.read[c["slot"]] = True; self.busy[c["slot"]] = False
            self.q.pop(0); self.issued -= 1
            self.pump(dur)
        sg.record(self.mark); cs.wait(self.mark)

def check(kernels):
    """Per slot: the kernels on it in time; a decode's interval must not overlap any gemm of the slot, and each gemm of
    chunk X must follow decode X, with no other decode of the slot between them."""
    by = {}
    for (kind, slot, cid), a, b in kernels:
        if slot is None: continue
        by.setdefault(slot, []).append((a, b, kind, cid))
    for slot, ks in by.items():
        decs = [k for k in ks if k[2] == "dec"]; gems = [k for k in ks if k[2] == "gemm"]
        for a, b, _, cid in gems:
            d = [x for x in decs if x[3] == cid]
            assert d, ("gemm without its decode in the slot", slot, cid)
            da, db = d[-1][0], d[-1][1]
            assert db <= a + 1e-9, ("gemm read before its decode ended", slot, cid, db, a)
            for x in decs:
                if x[3] != cid and x[1] > db and x[0] < b - 1e-9 and x[1] > a:
                    assert not (x[0] < b - 1e-9 and x[1] > da), ("another decode wrote the slot between the decode and the gemm's end", slot, cid, x)
        for x in decs:
            for a, b, _, cid in gems:
                if x[0] < b - 1e-9 and x[1] > a + 1e-9 and x[3] != cid:
                    raise AssertionError(("a decode wrote the slot while a gemm read it", slot, x, (a, b, cid)))
    return True

if __name__ == "__main__":
    seed0 = int(next((a for a in sys.argv[1:] if a.lstrip("-").isdigit()), 0))
    failed = []
    for seed in range(seed0, seed0 + 300):
        rnd = random.Random(seed)
        layer = [dict(name="qkv", O=7168, K=5120), dict(name="o", O=5120, K=5120), dict(name="gu", O=34816, K=5120), dict(name="down", O=5120, K=17408)]
        copies = rnd.choice([2, 4, 8])
        order = [dict(w, name=f"{w['name']}{c}") for c in range(copies) for w in layer]
        for w in order: w["base"] = w["name"][:-1]
        slot_elems = rnd.choice([17408 * 5120, 12800 * 4096, 8704 * 5120, 5120 * 17408 // 2])
        slots = rnd.choice([3, 4, 5, 6, 7, 8, 12, 16])
        slot_rows = {K: max(64, min(1 << 30, slot_elems // K // 64 * 64)) for K in (5120, 17408)}
        ring = Ring(slots, slot_rows)
        cs = Stream("cs")
        dscale, gscale = rnd.uniform(0.3, 3), rnd.uniform(0.3, 3)
        dur = lambda c: c["rows"] * c["W"]["K"] * dscale * rnd.uniform(0.5, 1.5) / 1e8
        gdur = lambda c: c["rows"] * c["W"]["K"] * gscale * rnd.uniform(0.5, 1.5) / 1e8
        ahead = rnd.choice([1, 2, 6, 100])
        for p in range(rnd.choice([2, 3, 5])):
            sms = rnd.choice([12, 12, 8])
            ring.reset(cs) if ring.cur else None
            queued = 0; pos = 0
            for i, W in enumerate(order):
                while queued < min(len(order), pos + ahead):
                    ring.api_queue(sms, order[queued], dur, cs); queued += 1
                pos = i + 1
                if rnd.random() < 0.05:  # an off-queue call now and then
                    W2 = dict(W, name=W["name"] + "x")
                    ring.linear(sms, W2, cs, dur, gdur); queued = len(order); pos = len(order)
                    break
                cs.kernel(("rest", None, None), rnd.uniform(0, 1))
                ring.linear(sms, W, cs, dur, gdur)
        streams = [cs] + [s for part in ring.parts.values() for s in part]
        try:
            check(run(streams))
        except AssertionError as e:
            if not WITHOUT:
                raise
            failed.append((seed, str(e)[:100]))
    if WITHOUT:
        print(f"without the {WITHOUT} wait: {len(failed)} of 300 schedules from {seed0} fail" + (f" (the first, seed {failed[0][0]}: {failed[0][1]})" if failed else ""))
    else:
        print("ok", seed0, seed0 + 300)
