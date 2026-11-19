// A CUDA driver on the host, for the development build's check of the expert cache without a GPU (src/offload_fake.rs; Linear GLY-130). It is `libcuda.so.1` as the engine loads it (src/ffi.rs `Driver`): the
// engine's `Dev::new`, `Buf`, `Pinned`, `Ev` and `Side` run on it, and a kernel does not (cuLaunchKernel is refused: the test plays the kernels' part, reading the buffers they would read).
//
//   c++ -std=c++17 -O1 -shared -fPIC -o $DIR/libcuda.so.1 tools/cuda_fake.cpp       (macOS: -dynamiclib)
//   LD_LIBRARY_PATH=$DIR cargo test --release -p glyd-engine --lib -- fake_gpu       (macOS: DYLD_LIBRARY_PATH; the test does nothing where the driver found is not this one)
//
// "Device" memory is host memory (an address is a device address), and the streams are queues of work that runs when the host makes it run: at a synchronize, at an event synchronize, or at as many operations as
// FAKE_CUDA_PROGRESS says (default 0: none) at each event query. That is the worst case of a real GPU's lateness: a copy from pinned memory reads its source when it runs, not when it is queued, so a host that writes
// the source between the two corrupts the copy, and a test that looks at the device's bytes sees it. Work waits for the events it waited for as CUDA has it (`cuStreamWaitEvent` waits for the event's latest record
// at the call; an event never recorded is no wait), and a wait that nothing can end (two streams waiting on each other) is an error: the call that needed it fails, and the test with it.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>

namespace {

struct Ev {
    uint64_t recorded = 0;  // records queued so far
    uint64_t done = 0;      // the latest record that has run
};

struct Op {
    enum Kind { COPY, SET, RECORD, WAIT } kind;
    void* dst = nullptr;
    const void* src = nullptr;
    size_t n = 0;
    int value = 0;
    Ev* ev = nullptr;
    uint64_t seq = 0;  // RECORD: this record's number; WAIT: the record waited for
};

struct Strm {
    std::deque<Op> q;
};

std::vector<Strm*> all;

void execute(const Op& op) {
    switch (op.kind) {
        case Op::COPY: memmove(op.dst, op.src, op.n); break;
        case Op::SET: memset(op.dst, op.value, op.n); break;
        case Op::RECORD: if (op.seq > op.ev->done) op.ev->done = op.seq; break;
        case Op::WAIT: break;
    }
}

bool runnable(const Op& op) { return op.kind != Op::WAIT || op.ev->done >= op.seq; }

// one operation run, from the first stream that has one it can run; false if there is none
bool step() {
    for (Strm* s : all) {
        if (!s->q.empty() && runnable(s->q.front())) {
            execute(s->q.front());
            s->q.pop_front();
            return true;
        }
    }
    return false;
}

constexpr int DEADLOCK = 719;  // (CUDA_ERROR_LAUNCH_FAILED: the closest thing a driver says to "this will never end")

template <class F> int until(F done) {
    while (!done()) {
        if (!step()) {
            fprintf(stderr, "cuda_fake: work waits on work that waits on it\n");
            return DEADLOCK;
        }
    }
    return 0;
}

int enqueue(void* stream, Op op) {
    if (!stream) {  // the legacy stream: nothing is queued behind it here
        if (op.kind == Op::COPY || op.kind == Op::SET) execute(op);
        return 0;
    }
    static_cast<Strm*>(stream)->q.push_back(op);
    return 0;
}

}  // namespace

extern "C" {

typedef void* H;

int cuInit(unsigned) { return 0; }
int cuDriverGetVersion(int* v) { *v = 13000; return 0; }
int cuDeviceGet(int* d, int i) { *d = i; return 0; }
int cuDeviceGetCount(int* n) { *n = 1; return 0; }
int cuDeviceGetName(char* name, int len, int) { snprintf(name, len, "Fake GPU"); return 0; }
int cuDeviceGetAttribute(int* v, int attr, int) { *v = attr == 75 ? 8 : attr == 76 ? 9 : attr == 16 ? 128 : 0; return 0; }
int cuDeviceTotalMem_v2(size_t* b, int) { *b = size_t(24) << 30; return 0; }
int cuDevicePrimaryCtxRetain(H* c, int) { static int ctx; *c = &ctx; return 0; }
int cuDevicePrimaryCtxRelease_v2(int) { return 0; }
int cuCtxSetCurrent(H) { return 0; }
int cuCtxSynchronize() { return until([] { for (Strm* s : all) if (!s->q.empty()) return false; return true; }); }
int cuMemGetInfo_v2(size_t* f, size_t* t) { *f = size_t(16) << 30; *t = size_t(24) << 30; return 0; }

int cuMemAlloc_v2(uint64_t* p, size_t n) {
    void* m = calloc(1, n ? n : 1);
    if (!m) return 2;
    *p = reinterpret_cast<uint64_t>(m);
    return 0;
}
int cuMemFree_v2(uint64_t p) { free(reinterpret_cast<void*>(p)); return 0; }
int cuMemHostAlloc(void** p, size_t n, unsigned) {
    size_t sz = ((n ? n : 1) + 4095) & ~size_t(4095);
    void* m = aligned_alloc(4096, sz);
    if (!m) return 2;
    memset(m, 0xCD, sz);  // (what a copy reads before it is written shows as a pattern, not as zeros that might be right)
    *p = m;
    return 0;
}
int cuMemFreeHost(void* p) { free(p); return 0; }

int cuMemcpyHtoD_v2(uint64_t d, const void* s, size_t n) { memmove(reinterpret_cast<void*>(d), s, n); return 0; }
int cuMemcpyDtoH_v2(void* d, uint64_t s, size_t n) { memmove(d, reinterpret_cast<void*>(s), n); return 0; }
int cuMemsetD8_v2(uint64_t d, unsigned char v, size_t n) { memset(reinterpret_cast<void*>(d), v, n); return 0; }
int cuMemcpyHtoDAsync_v2(uint64_t d, const void* s, size_t n, H st) { Op o{Op::COPY}; o.dst = reinterpret_cast<void*>(d); o.src = s; o.n = n; return enqueue(st, o); }
int cuMemcpyDtoHAsync_v2(void* d, uint64_t s, size_t n, H st) { Op o{Op::COPY}; o.dst = d; o.src = reinterpret_cast<void*>(s); o.n = n; return enqueue(st, o); }
int cuMemcpyDtoDAsync_v2(uint64_t d, uint64_t s, size_t n, H st) { Op o{Op::COPY}; o.dst = reinterpret_cast<void*>(d); o.src = reinterpret_cast<void*>(s); o.n = n; return enqueue(st, o); }
int cuMemsetD8Async(uint64_t d, unsigned char v, size_t n, H st) { Op o{Op::SET}; o.dst = reinterpret_cast<void*>(d); o.value = v; o.n = n; return enqueue(st, o); }

int cuStreamCreate(H* s, unsigned) {
    Strm* st = new Strm();
    all.push_back(st);
    *s = st;
    return 0;
}
int cuStreamDestroy_v2(H s) {
    for (size_t i = 0; i < all.size(); i++) if (all[i] == s) { all.erase(all.begin() + i); break; }
    delete static_cast<Strm*>(s);
    return 0;
}
int cuStreamSynchronize(H s) {
    if (!s) return 0;
    Strm* st = static_cast<Strm*>(s);
    return until([st] { return st->q.empty(); });
}

int cuEventCreate(H* e, unsigned) { *e = new Ev(); return 0; }
int cuEventDestroy_v2(H e) { delete static_cast<Ev*>(e); return 0; }
int cuEventRecord(H e, H s) {
    Ev* ev = static_cast<Ev*>(e);
    Op o{Op::RECORD};
    o.ev = ev;
    o.seq = ++ev->recorded;
    if (!s) { execute(o); return 0; }
    return enqueue(s, o);
}
int cuStreamWaitEvent(H s, H e, unsigned) {
    Ev* ev = static_cast<Ev*>(e);
    if (ev->recorded == 0 || !s) return 0;  // (never recorded: no wait)
    Op o{Op::WAIT};
    o.ev = ev;
    o.seq = ev->recorded;
    return enqueue(s, o);
}
int cuEventQuery(H e) {
    Ev* ev = static_cast<Ev*>(e);
    // (a GPU runs on its own: FAKE_CUDA_PROGRESS operations of whatever is queued run at a query)
    const char* p = getenv("FAKE_CUDA_PROGRESS");
    for (long n = p ? atol(p) : 0; n > 0 && ev->done < ev->recorded; n--) if (!step()) break;
    return ev->done >= ev->recorded ? 0 : 600;
}
int cuEventSynchronize(H e) {
    Ev* ev = static_cast<Ev*>(e);
    return until([ev] { return ev->done >= ev->recorded; });
}
int cuEventElapsedTime(float* ms, H, H) { *ms = 0.2f; return 0; }

int cuStreamBeginCapture_v2(H, int) { return 0; }
int cuStreamEndCapture(H, H* g) { *g = nullptr; return 0; }
int cuGraphInstantiateWithFlags(H* x, H, uint64_t) { *x = nullptr; return 0; }
int cuGraphLaunch(H, H) { return 0; }
int cuGraphExecDestroy(H) { return 0; }
int cuGraphDestroy(H) { return 0; }
int cuModuleLoadData(H* m, const void*) { *m = nullptr; return 0; }
int cuModuleGetFunction(H* f, H, const char*) { *f = nullptr; return 0; }
int cuFuncSetAttribute(H, int, int) { return 0; }
int cuLaunchKernel(H, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, H, void**, void**) { return 1; }
int cuGetErrorString(int e, const char** s) {
    *s = e == DEADLOCK ? "work waits on work that waits on it (a deadlock of streams)" : e == 600 ? "not ready" : "an error of the fake driver";
    return 0;
}

}  // extern "C"
