//! GLM-5.3-Flash's routed experts for the tokens of a prompt chunk (kernels/cb2_prefill.cu): `cb2_prefill_experts` is the whole layer's routed sum for a chunk of n tokens, the experts of the `glyd-hbr-codes/2` pack
//! (codebook rows: ternary, two, three or four bits, src/cpu_cb.rs) multiplied by all their tokens at once on the tensor cores, in the basis the pack's rows are in.
//!
//! The chunk's pairs (token, expert) are read back once (ids and weights, one wait for the stream), sorted by expert on the host, and the layer's experts that have a pair go through a buffer of slots on the GPU eight at
//! a time: the host copies a micro-batch's codes and books from the layer's pack (`ExpertDesc` for each expert: the file mapped in place, or any host memory) to a slot, the kernels for the micro-batch run on the caller's
//! stream when the copy has, and the slot is free again when they have. A copy from memory that is not pinned (a mapped file) is made by the driver through its own pinned buffers and returns when the bytes are staged,
//! so the call to the host's copy blocks the caller's thread for about the time of the transfer, with the kernels of the micro-batch before it running meanwhile (or `GLYD_CB2_COPIERS` threads make the copies).
//! The slot is as big as the layer's biggest expert (in the form the kernels read: a ternary row is made two-bit rows by a kernel after its copy, in a buffer of its own), and an expert copies only its own bytes.
//!
//! The result `yrot` (f32 [n, 4096]) is, for each token, the sum over its k pairs of weight * (activation . downT) in the rotated basis. It is summed with atomic adds in fp32 (a token's pairs come from several experts,
//! which run at once): the same to fp32 rounding, not to the bit, from run to run.
//!
//! `glyd-engine cb2prefill [--n 1024] [--k 8] [--experts 216] [--fmts cb2:0.5,tern:0.3,cb3:0.2] [--resident 0.15] [--busy 40] [--check 16] [--reps 3] [--skew 0] [--pack DIR --layer L]` (a development build's): random experts (of the formats
//! asked for, or a layer of a pack, mapped) and tokens, the time of a call and its sums for a few tokens against src/cpu_cb.rs's.

use crate::cpu_cb::Fmt;
use crate::dev::{copy_d2h_async, copy_h2d_async, memset_async, Buf, Dev, Ev, EvRef, Kernel, Pinned, Side};
use crate::ffi::{err, Error, Result, Stream};
use crate::launch;
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

const HID: usize = 4096;
const INTER: usize = 2048;
/// A two-bit expert's bytes in the pack: its codes (downT 2048 x 1024, then up 4096 x 1024) and its books (downT 2048 x 4 bf16, then up 4096 x 4).
pub const CODES: usize = 6 * 1024 * 1024;
pub const BOOKS: usize = 48 * 1024;
/// Experts a micro-batch, and the bytes the ring of micro-batches is made to hold (GLYD_CB2_DEPTH sets the number of micro-batches instead): the copies run this far ahead of the products, and, started before the
/// router's answer is back, hide behind the layer's attention. Chunks of fewer tokens than SPEC_MIN copy only the experts they use, after the router.
const MICRO: usize = 16;
const RING_BYTES: usize = 320 << 20;
const SPEC_MIN: usize = 256;
/// Pairs a tile, and at most a launch's (the activations' buffer holds a launch's).
const TILE: usize = 64;
const CAP_ROWS: usize = 16384;
/// A stage's shared memory of the first products with 128 pairs a block (`c2qx_up128`), and the pairs a launch's segments have on average at least for it (fewer leave too much of its tiles empty).
const UP128_STAGE: usize = (128 * 64 + 128 * 64) * 2;
const UP128_ROWS: i32 = 192;
/// A stage's shared memory of the second products of every width (`c2qx_down*`: 64 pairs by `out` outputs, steps of 64 neurons): 256 outputs a block, or 128
/// (`c2qx_down4w`) where a block takes 64 KiB at most (`dev::smem64`).
const fn downx_stage(out: usize) -> usize {
    (64 * 64 + 64 * out) * 2
}
/// Neurons a block of the first product takes, and outputs a block of the second.
const NEURONS: usize = 64;
const OUTPUTS: usize = 128;

/// Bytes of a row of 4,096 codes, and entries of its table, in the pack.
#[cfg_attr(not(feature = "dev"), allow(dead_code))] // development tools only
fn row_bytes(f: Fmt) -> usize {
    f.row_bytes(HID)
}

/// The form of a format the kernels read: a ternary row is made two-bit rows (its three entries and a zero).
fn kform(f: Fmt) -> Fmt {
    if f == Fmt::Tern {
        Fmt::Cb2
    } else {
        f
    }
}

/// The bits a code takes in the kernels' form of `f` (5, 6, 7: a trellis row of 2, 3, 4 bits a weight, the kernels' B = 5, 6, 7).
fn bits_of(f: Fmt) -> usize {
    match kform(f) {
        Fmt::Cb2 => 2,
        Fmt::Cb3 => 3,
        Fmt::Tq => 5,
        Fmt::Tq3 => 6,
        Fmt::Tq4 => 7,
        _ => 4,
    }
}

/// One expert of a layer on the host: where its codes ([downT rows | up rows], the pack's bytes) and its books ([downT | up]) are, and the format of each matrix (the pack's 'X+Y' is X for the up, Y for the down).
#[derive(Clone, Copy, Debug)]
pub struct ExpertDesc {
    pub codes: *const u8,
    pub books: *const u8,
    pub up: Fmt,
    pub down: Fmt,
    /// Where the expert already is on the GPU, if it is (the addresses of its codes and of its books, the bytes as the pack holds them: a place of the expert cache): the products read it there and nothing is copied.
    /// Not for an expert with a ternary matrix (the kernels need those made two-bit rows first).
    pub resident: Option<(u64, u64)>,
    /// The expert's width (a row's codes) and neurons: GLM's 4,096 and 2,048, Qwen3.8-Flash-Next's 2,560 and 640.
    pub hid: usize,
    pub inter: usize,
}

impl ExpertDesc {
    /// The bytes of the expert's codes and of its books as the host holds them.
    pub fn bytes(&self) -> (usize, usize) {
        (
            self.inter * self.down.row_bytes(self.hid) + 2 * self.inter * self.up.row_bytes(self.hid),
            (self.inter * self.down.entries() + 2 * self.inter * self.up.entries()) * 2,
        )
    }

    /// The same in the kernels' form (ternary rows as two-bit ones), each rounded up to 256.
    fn kbytes(&self) -> (usize, usize) {
        let up = |x: usize| x.div_ceil(256) * 256;
        (
            up(self.inter * kform(self.down).row_bytes(self.hid) + 2 * self.inter * kform(self.up).row_bytes(self.hid)),
            up((self.inter * kform(self.down).entries() + 2 * self.inter * kform(self.up).entries()) * 2),
        )
    }
}

/// Where the layer's experts are on the host when they are all two-bit and end to end: expert i's codes at `codes + i * cstride`, its books at `books + i * bstride`.
#[derive(Clone, Copy, Debug)]
pub struct Cb2Host {
    pub codes: *const u8,
    pub books: *const u8,
    pub experts: usize,
    pub cstride: usize,
    pub bstride: usize,
}

impl Cb2Host {
    pub fn descs(&self) -> Vec<ExpertDesc> {
        // SAFETY: pointer arithmetic inside the caller's arrays (the caller's word: `experts` strides of each).
        (0..self.experts)
            .map(|i| unsafe {
                ExpertDesc {
                    codes: self.codes.add(i * self.cstride),
                    books: self.books.add(i * self.bstride),
                    up: Fmt::Cb2,
                    down: Fmt::Cb2,
                    resident: None,
                    hid: HID,
                    inter: INTER,
                }
            })
            .collect()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Segs {
    n: i32,
    begin: [i32; 16],
    rows: [i32; 16],
    hoff: [i32; 16],
    tiles: [i32; 17],
    ucode: [i64; 16],
    dcode: [i64; 16],
    ubook: [i64; 16],
    dbook: [i64; 16],
    ubits: [i32; 16],
    dbits: [i32; 16],
}

/// Where an expert of a micro-batch is in the ring (the offsets of its matrices' codes and books in the kernels' form), and their bits.
#[derive(Clone, Copy, Default, Debug)]
struct Place {
    ucode: i64,
    dcode: i64,
    ubook: i64,
    dbook: i64,
    ubits: u8,
    dbits: u8,
}

struct Ring {
    // micro-batches the buffers hold
    n: usize,
    codes: Buf,
    books: Buf,
    // the ternary matrices' bytes as the host holds them, a set for each slot, until `c2_tern_cb2` has made them two-bit rows
    raw_codes: Option<Buf>,
    raw_books: Option<Buf>,
    slot_codes: usize,
    slot_books: usize,
    raw_slot_codes: usize,
    raw_slot_books: usize,
    // (untimed events, which the copier threads wait on by handle too)
    ready: Vec<Ev>,
    free: Vec<Ev>,
    next: usize,
}

struct Scratch {
    pairs: usize,
    ids_h: Pinned,
    wts_h: Pinned,
    tok_h: Pinned,
    wt_h: Pinned,
    tok_d: Buf,
    wt_d: Buf,
}

struct Inner {
    ring: Option<Ring>,
    scratch: Option<Scratch>,
    act: Option<Buf>,
    copy: Side,
    sync: Ev,
    /// the host's share: the call's rows of x' (bf16), the host's sums (fp32), both pinned, and the sums on the device
    host_x: Option<Pinned>,
    host_y: Option<Pinned>,
    host_yd: Option<Buf>,
}

#[cfg(unix)]
extern "C" {
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

/// Asks the system to bring the pages of [p, p + len) in (MADV_WILLNEED): a mapped file's pages that are not in memory are read beside whatever else runs; an advice only, whose failure does not matter.
fn will_need(p: *const u8, len: usize) {
    let start = p as usize / 4096 * 4096;
    // SAFETY: an advice about a range of the caller's host memory; it changes no byte.
    #[cfg(unix)]
    unsafe {
        madvise(start as *mut c_void, p as usize + len - start, 3)
    };
    #[cfg(windows)]
    unsafe {
        crate::os::win::prefetch(start as *mut c_void, p as usize + len - start)
    };
}

/// A micro-batch's copies to a ring slot made by threads of their own (each with a stream): a copy from memory that is not pinned holds its thread for the time of the transfer.
struct Done {
    left: Mutex<usize>,
    cv: Condvar,
    failed: Mutex<Option<String>>,
}

impl Done {
    fn new(n: usize) -> Arc<Done> {
        Arc::new(Done { left: Mutex::new(n), cv: Condvar::new(), failed: Mutex::new(None) })
    }

    fn finish(&self, e: Option<String>) {
        if let Some(e) = e {
            self.failed.lock().unwrap().get_or_insert(e);
        }
        *self.left.lock().unwrap() -= 1;
        self.cv.notify_all();
    }

    fn wait(&self) -> Result<()> {
        let mut left = self.left.lock().unwrap();
        while *left > 0 {
            left = self.cv.wait(left).unwrap();
        }
        match self.failed.lock().unwrap().take() {
            Some(e) => err(e),
            None => Ok(()),
        }
    }
}

struct Job {
    copies: Vec<(u64, usize, usize)>, // (device address, host address, bytes)
    // events the thread waits for before the first copy: the kernels that last read the slot
    after: Vec<EvRef>,
    done: Arc<Done>,
    // the host's bytes are not pinned: copied through the thread's own pinned buffers
    stage: bool,
}

/// A copier thread's pinned buffers for host bytes that are not pinned (a mapped file's) and the bytes of each: the thread copies a piece into one while the DMA of the one before runs. The driver's own path for
/// such a copy fills its buffer and copies it in turn, about 12 GB/s on the 4080's computer where pinned memory gives 25, and threads of its own do not run its copies side by side.
const STAGES: usize = 3;
const STAGE_BYTES: usize = 2 << 20;

struct Copiers {
    tx: Vec<Sender<Job>>,
    handles: Vec<JoinHandle<()>>,
}

fn copier(ordinal: i32, rx: Receiver<Job>, up: Sender<std::result::Result<(), String>>) {
    let bound = crate::dev::bind_thread(ordinal).and_then(|_| Side::new());
    let (side, stages) = match bound.and_then(|s| Ok((s, (0..STAGES).map(|_| Ok((Pinned::new(STAGE_BYTES)?, Ev::new()?))).collect::<Result<Vec<_>>>()?))) {
        Ok(s) => {
            let _ = up.send(Ok(()));
            s
        }
        Err(e) => {
            let _ = up.send(Err(e.msg));
            return;
        }
    };
    let mut next = 0usize;
    while let Ok(job) = rx.recv() {
        let r = (|| -> Result<()> {
            for ev in &job.after {
                ev.wait_host()?;
            }
            for &(dst, src, len) in &job.copies {
                if !job.stage {
                    // SAFETY: the host range is the caller's to keep until the call returns, the device range a slot of the ring, free until the kernels that read it have run.
                    unsafe { copy_h2d_async(dst, src as *const c_void, len, side.stream)? };
                    continue;
                }
                for off in (0..len).step_by(STAGE_BYTES) {
                    let n = STAGE_BYTES.min(len - off);
                    let (buf, ev) = &stages[next % STAGES];
                    next += 1;
                    // (the buffer's last DMA done before it is written again)
                    ev.wait_host()?;
                    // SAFETY: as above; the buffer holds STAGE_BYTES and its last copy has run.
                    unsafe {
                        std::ptr::copy_nonoverlapping((src + off) as *const u8, buf.ptr as *mut u8, n);
                        copy_h2d_async(dst + off as u64, buf.ptr, n, side.stream)?;
                    }
                    ev.record(side.stream)?;
                }
            }
            side.sync()
        })();
        job.done.finish(r.err().map(|e| e.msg));
    }
}

impl Copiers {
    fn new(ordinal: i32, n: usize) -> Result<Copiers> {
        let (mut tx, mut handles) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let (jt, jr) = channel::<Job>();
            let (ut, ur) = channel();
            handles.push(std::thread::spawn(move || copier(ordinal, jr, ut)));
            tx.push(jt);
            ur.recv().map_err(|_| Error("a copier thread ended before it started".into()))?.map_err(Error)?;
        }
        Ok(Copiers { tx, handles })
    }
}

impl Drop for Copiers {
    fn drop(&mut self) {
        self.tx.clear();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// The host's share of a call (`Cb2Kern::host`): the experts with at most `max_rows` pairs that are not on the GPU (nor among the micro-batches already on their way) computed by `threads` threads of the CPU from their
/// bytes in host memory (src/cpu_cb.rs), while the GPU streams and multiplies the rest; their sums added to the GPU's at the end of the call.
/// With `auto`, `max_rows` moves after each call toward the host's time being ~85% of the GPU loop's (the next layer's share like this one's: their pairs are spread alike).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostShare {
    pub max_rows: usize,
    pub threads: usize,
    pub auto: bool,
}

/// What the host's share of the last call took: its experts and pairs, and the host's time and the GPU loop's (the caller's thread, from the plan to the last launch), in milliseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostTook {
    pub experts: usize,
    pub pairs: usize,
    pub host_ms: f64,
    pub gpu_ms: f64,
}

pub struct Cb2Kern {
    /// the host's share of the next calls (None: the GPU takes every expert), and what the last call's took
    pub host: std::cell::Cell<Option<HostShare>>,
    pub took: std::cell::Cell<HostTook>,
    add: Kernel,
    // the products of two, three and four bits
    up: [Kernel; 3],
    down: [Kernel; 3],
    tern: Kernel,
    /// the ternary rows' conversion at any width (Qwen's), and the width and neurons the products are made for
    tern_rows: Kernel,
    /// the products of every width in one launch (Qwen3.8-Flash-Next's experts: `c2qx_*`), the outputs a block of their second, and the first products with 128 pairs a block
    /// (two stages only)
    mixed: Option<(Kernel, Kernel)>,
    dout: usize,
    up128: Option<Kernel>,
    hid: usize,
    inter: usize,
    stages: usize,
    limit: f32,
    copiers: Option<Copiers>,
    /// whether the experts' bytes on the host are pinned (copied as they are; else through the copier threads' own pinned buffers)
    pub pinned: std::cell::Cell<bool>,
    // (a development build's GLYD_CB2_NOCOPY=1: the experts are not copied, the products run on whatever the slots hold, to time them alone)
    nocopy: bool,
    inner: RefCell<Inner>,
}

impl Cb2Kern {
    /// The kernels, two or three stages of shared memory a block (GLYD_CB2_STAGES, 2 by default; two where a block takes 64 KiB at most, `dev::smem64`, which takes the
    /// first products of 64 pairs a block too, and the second of 128 outputs), the clamp of the SwiGLU (10), and GLYD_CB2_COPIERS threads (3 by default, each copying its share
    /// of a micro-batch through pinned buffers of its own; 0: the copies are made on the caller's thread, which they hold for the time of the transfers) that copy a micro-batch's experts to its slot beside the caller.
    pub fn load(dev: &Dev) -> Result<Cb2Kern> {
        Cb2Kern::load_dims(dev, HID, INTER, 10.0)
    }

    /// The kernels for experts of `inter` neurons over `hid` (GLM's 4,096 x 2,048 or Qwen3.8-Flash-Next's 2,560 x 640) and the SwiGLU's clamp (GLM's 10; none: the largest float).
    pub fn load_dims(dev: &Dev, hid: usize, inter: usize, limit: f32) -> Result<Cb2Kern> {
        let small = crate::dev::smem64();
        let stages = match crate::tune!("GLYD_CB2_STAGES").as_deref() {
            Some("3") if !small => 3,
            _ => 2,
        };
        let qwen = match (hid, inter) {
            (HID, INTER) => false,
            (2560, 640) => true,
            _ => return err(format!("the streaming prefill's kernels are made for experts of 2,048 x 4,096 and 640 x 2,560, not {inter} x {hid}")),
        };
        let pick = |glm: [&'static str; 2], q: [&'static str; 2]| dev.kernel(if qwen { q[stages - 2] } else { glm[stages - 2] });
        let up = [
            pick([kname!("c2_up2_2"), kname!("c2_up2_3")], [kname!("c2q_up2_2"), kname!("c2q_up2_3")])?,
            pick([kname!("c2_up3_2"), kname!("c2_up3_3")], [kname!("c2q_up3_2"), kname!("c2q_up3_3")])?,
            pick([kname!("c2_up4_2"), kname!("c2_up4_3")], [kname!("c2q_up4_2"), kname!("c2q_up4_3")])?,
        ];
        let down = [
            pick([kname!("c2_down2_2"), kname!("c2_down2_3")], [kname!("c2q_down2_2"), kname!("c2q_down2_3")])?,
            pick([kname!("c2_down3_2"), kname!("c2_down3_3")], [kname!("c2q_down3_2"), kname!("c2q_down3_3")])?,
            pick([kname!("c2_down4_2"), kname!("c2_down4_3")], [kname!("c2q_down4_2"), kname!("c2q_down4_3")])?,
        ];
        for k in up.iter().chain(&down) {
            k.allow_smem((stages * 24576) as u32)?;
        }
        // (c2qx_up128_2 takes 70.1 KiB, c2qx_down_2 86.5: not where a block takes 64 KiB at most)
        let up128 = if qwen && stages == 2 && !small {
            let k = dev.kernel(kname!("c2qx_up128_2"))?;
            k.allow_smem((2 * UP128_STAGE) as u32)?;
            Some(k)
        } else {
            None
        };
        let dout = if small { 128 } else { 256 };
        let mixed = if qwen {
            let (u, d) = match (stages, small) {
                (3, _) => (dev.kernel(kname!("c2qx_up_3"))?, dev.kernel(kname!("c2qx_down_3"))?),
                (_, true) => (dev.kernel(kname!("c2qx_up_2"))?, dev.kernel(kname!("c2qx_down4w_2"))?),
                _ => (dev.kernel(kname!("c2qx_up_2"))?, dev.kernel(kname!("c2qx_down_2"))?),
            };
            u.allow_smem((stages * 24576) as u32)?;
            d.allow_smem((stages * downx_stage(dout)) as u32)?;
            Some((u, d))
        } else {
            None
        };
        let workers = crate::tune!("GLYD_CB2_COPIERS").and_then(|v| v.parse().ok()).unwrap_or(3usize).min(8);
        let copiers = if workers > 0 { Some(Copiers::new(dev.ordinal, workers)?) } else { None };
        let nocopy = crate::tune!("GLYD_CB2_NOCOPY").is_some_and(|v| v == "1");
        Ok(Cb2Kern {
            host: std::cell::Cell::new(None),
            took: std::cell::Cell::new(HostTook::default()),
            add: dev.kernel(kname!("c2_add_f32"))?,
            up,
            down,
            tern: dev.kernel(kname!("c2_tern_cb2"))?,
            tern_rows: dev.kernel(kname!("c2_tern_rows"))?,
            mixed,
            dout,
            up128,
            hid,
            inter,
            stages,
            limit,
            copiers,
            pinned: std::cell::Cell::new(false),
            nocopy,
            inner: RefCell::new(Inner {
                ring: None,
                scratch: None,
                act: None,
                copy: Side::new()?,
                sync: Ev::new()?,
                host_x: None,
                host_y: None,
                host_yd: None,
            }),
        })
    }

    /// The device and pinned memory the calls hold given back (the ring, the plan's, the activations, the host's share's); the next call makes them again. The caller has synchronized the stream: no kernel or copy
    /// of a call is left to run.
    pub fn release_buffers(&self) {
        let mut g = self.inner.borrow_mut();
        g.ring = None;
        g.scratch = None;
        g.act = None;
        g.host_x = None;
        g.host_y = None;
        g.host_yd = None;
    }

    /// Bytes of device memory the call holds for a chunk of `pairs` pairs over experts of at most `biggest` bytes in the kernels' form (the ring, the plan, the activations of a launch), without ternary experts' buffers.
    pub fn device_bytes(pairs: usize, biggest: usize) -> usize {
        let slot = MICRO * (biggest + 98304);
        (RING_BYTES / (MICRO * biggest).max(1)).clamp(3, 16) * slot + pairs * 8 + pairs.min(CAP_ROWS) * INTER * 2
    }
}

/// The routed experts' sum for a chunk of `n` tokens with `k` pairs each, the layer's experts being two-bit and end to end in memory (`Cb2Host`): see [`cb2_prefill_experts`].
///
/// # Safety
/// As `cb2_prefill_experts`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cb2_prefill(dev: &Dev, kern: &Cb2Kern, host: &Cb2Host, n: usize, k: usize, xrb: u64, ids: u64, wts: u64, yrot: u64, s: Stream) -> Result<()> {
    if host.cstride != CODES || host.bstride != BOOKS {
        return err(format!("experts of {} and {} bytes: this format's experts are {CODES} and {BOOKS}", host.cstride, host.bstride));
    }
    cb2_prefill_experts(dev, kern, &host.descs(), n, k, xrb, ids, wts, yrot, s)
}

/// The routed experts' sum for a chunk of `n` tokens with `k` pairs each: `yrot` [n, 4096] (f32, written) = sum over a token's pairs of weights[pair] * (activation . downT) for the pair's expert, in the rotated basis.
/// `experts` are the layer's experts on the host (a pair's expert is `ids[pair]`, a number into this slice), `xrb` the tokens' rows of x' = R x (bf16 [n, 4096]), `ids` the experts (int64 [n * k]; pair p is token
/// p / k's), `wts` their weights (f32 [n * k]). Everything is queued on `s` after what `s` has run, and the call returns when the pairs have been read back and every micro-batch's kernels are queued (not when they
/// have run).
///
/// For a chunk of 256 tokens or more the copies of the first micro-batches (the experts in number order, the ones already on the GPU left out) are started before the router's ids are waited for, so that they run
/// while the GPU works through the layer's attention; the ring is as big as `RING_BYTES` holds. The pipeline is Strata's (MIT, github.com/Niko1221/Strata, prefill/prefill.cpp): the copy stream runs ahead of the
/// products, a ring of slots between them, each freed by an event after its kernels.
///
/// # Safety
/// Live device buffers of the sizes above on this device; the experts hold the bytes they say until the stream has run the call; nothing else uses the call's buffers (`Cb2Kern` is one thread's).
#[allow(clippy::too_many_arguments)]
pub unsafe fn cb2_prefill_experts(
    _dev: &Dev,
    kern: &Cb2Kern,
    experts: &[ExpertDesc],
    n: usize,
    k: usize,
    xrb: u64,
    ids: u64,
    wts: u64,
    yrot: u64,
    s: Stream,
) -> Result<()> {
    let pairs = n * k;
    if n == 0 || !(1..=16).contains(&k) || experts.is_empty() || xrb % 16 != 0 || yrot % 16 != 0 {
        return err(format!(
            "a prompt chunk of {n} tokens with {k} pairs each over {} experts: tokens, 1 to 16 pairs, experts, 16-byte aligned rows",
            experts.len()
        ));
    }
    let mut guard = kern.inner.borrow_mut();
    let g = &mut *guard;
    // the buffers: slots as big as the biggest expert, a set of ternary rows' bytes if the layer has any
    let (slot_codes, slot_books) =
        (MICRO * experts.iter().map(|d| d.kbytes().0).max().unwrap_or(0), MICRO * experts.iter().map(|d| d.kbytes().1).max().unwrap_or(0));
    let ring_n = crate::tune!("GLYD_CB2_DEPTH").and_then(|v| v.parse::<usize>().ok()).unwrap_or(RING_BYTES / slot_codes.max(1)).clamp(3, 16);
    if experts.iter().any(|d| (d.hid, d.inter) != (kern.hid, kern.inter)) {
        return err(format!("experts of other sizes than the kernels' {} x {}", kern.inter, kern.hid));
    }
    let (hid, inter) = (kern.hid, kern.inter);
    let tern_bytes = |d: &ExpertDesc| -> (usize, usize) {
        let m = |f: Fmt, rows: usize| if f == Fmt::Tern { (rows * Fmt::Tern.row_bytes(hid), rows * 6) } else { (0, 0) };
        let (u, d) = (m(d.up, 2 * inter), m(d.down, inter));
        (u.0 + d.0, u.1 + d.1)
    };
    let (raw_codes, raw_books) =
        (MICRO * experts.iter().map(|d| tern_bytes(d).0).max().unwrap_or(0), MICRO * experts.iter().map(|d| tern_bytes(d).1).max().unwrap_or(0));
    let grown =
        |r: &Ring| r.n != ring_n || r.slot_codes < slot_codes || r.slot_books < slot_books || r.raw_slot_codes < raw_codes || r.raw_slot_books < raw_books;
    if g.ring.as_ref().map_or(true, grown) {
        // (the stream has run the last call's kernels: the wait for its pairs came after them)
        g.ring = None;
        g.ring = Some(Ring {
            n: ring_n,
            codes: Buf::new(ring_n * slot_codes)?,
            books: Buf::new(ring_n * slot_books)?,
            raw_codes: if raw_codes > 0 { Some(Buf::new(ring_n * raw_codes)?) } else { None },
            raw_books: if raw_books > 0 { Some(Buf::new(ring_n * raw_books)?) } else { None },
            slot_codes,
            slot_books,
            raw_slot_codes: raw_codes,
            raw_slot_books: raw_books,
            ready: (0..ring_n).map(|_| Ev::untimed()).collect::<Result<_>>()?,
            free: (0..ring_n).map(|_| Ev::untimed()).collect::<Result<_>>()?,
            next: 0,
        });
    }
    if g.scratch.as_ref().map_or(true, |x| x.pairs < pairs) {
        g.scratch = None;
        g.scratch = Some(Scratch {
            pairs,
            ids_h: Pinned::new(pairs * 8)?,
            wts_h: Pinned::new(pairs * 4)?,
            tok_h: Pinned::new(pairs * 4)?,
            wt_h: Pinned::new(pairs * 4)?,
            tok_d: Buf::new(pairs * 4)?,
            wt_d: Buf::new(pairs * 4)?,
        });
    }
    let act_rows = pairs.min(CAP_ROWS);
    if g.act.as_ref().map_or(true, |b| b.bytes < act_rows * inter * 2) {
        g.act = None;
        g.act = Some(Buf::new(act_rows * inter * 2)?);
    }
    let (ring, sc, act) = (g.ring.as_mut().expect("the copy buffers"), g.scratch.as_ref().expect("scratch"), g.act.as_ref().expect("act"));
    let cs = g.copy.stream;
    // Which experts are copied and which are read where they are, in what order; a chunk of enough tokens uses every expert, so the order is known now and the copies can start before the router's answer
    let in_place = |e: usize| experts[e].resident.is_some() && experts[e].up != Fmt::Tern && experts[e].down != Fmt::Tern;
    let full = n >= SPEC_MIN;
    let (mut res_order, mut copy_order): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    if full {
        (0..experts.len()).for_each(|e| if in_place(e) { res_order.push(e) } else { copy_order.push(e) });
    }
    let first = ring.next;
    let mut issued = InFlight(Vec::new());
    if full {
        let chunks: Vec<&[usize]> = copy_order.chunks(MICRO).collect();
        (0..ring.n + 1).for_each(|c| advise(experts, chunks.get(c).copied()));
        for (c, chunk) in chunks.iter().enumerate() {
            issued.push(if c < ring.n { Some(issue(kern, ring, experts, cs, first, c, chunk)?) } else { None });
        }
    }
    // the pairs to the host (and x' where the host takes a share), and a wait for them (and for everything before them on the stream)
    crate::tmark!(s, "experts: start");
    let share = kern.host.get().filter(|h| h.max_rows > 0 && h.threads > 0);
    if share.is_some() {
        if g.host_x.as_ref().map_or(true, |b| b.bytes < n * hid * 2) {
            g.host_x = None;
            g.host_x = Some(Pinned::new(n * hid * 2)?);
            g.host_y = None;
            g.host_y = Some(Pinned::new(n * hid * 4)?);
            g.host_yd = None;
            g.host_yd = Some(Buf::new(n * hid * 4)?);
        }
        copy_d2h_async(g.host_x.as_ref().map_or(std::ptr::null_mut(), |b| b.ptr), xrb, n * hid * 2, s)?;
    }
    copy_d2h_async(sc.ids_h.ptr, ids, pairs * 8, s)?;
    copy_d2h_async(sc.wts_h.ptr, wts, pairs * 4, s)?;
    g.sync.record(s)?;
    g.sync.wait_host()?;
    // SAFETY: the pinned buffers hold `pairs` values each and the copies above have run.
    let (idv, wv) = (std::slice::from_raw_parts(sc.ids_h.ptr as *const i64, pairs), std::slice::from_raw_parts(sc.wts_h.ptr as *const f32, pairs));
    let mut count = vec![0usize; experts.len()];
    for &e in idv {
        if e < 0 || e as usize >= experts.len() {
            return err(format!("an expert number {e} among {} (the chunk's ids are the layer's local numbers)", experts.len()));
        }
        count[e as usize] += 1;
    }
    let mut start = vec![0usize; experts.len() + 1];
    for e in 0..experts.len() {
        start[e + 1] = start[e] + count[e];
    }
    // SAFETY: as above; the plan is `pairs` long.
    let (tok, wt) = (std::slice::from_raw_parts_mut(sc.tok_h.ptr as *mut i32, pairs), std::slice::from_raw_parts_mut(sc.wt_h.ptr as *mut f32, pairs));
    let mut fill = start.clone();
    for (p, &e) in idv.iter().enumerate() {
        let pos = fill[e as usize];
        fill[e as usize] += 1;
        tok[pos] = (p / k) as i32;
        wt[pos] = wv[p];
    }
    copy_h2d_async(sc.tok_d.ptr, sc.tok_h.ptr, pairs * 4, s)?;
    copy_h2d_async(sc.wt_d.ptr, sc.wt_h.ptr, pairs * 4, s)?;
    memset_async(yrot, 0, n * hid * 4, s)?;
    crate::tmark!(s, "experts: plan");
    // the host's share: experts of few pairs that are not on the GPU and whose copies have not started (a call of every expert keeps its first micro-batches, already on their way)
    let mut on_host = vec![false; experts.len()];
    if let Some(h) = share {
        let started = if full { (issued.iter().take_while(|i| i.is_some()).count() * MICRO).min(copy_order.len()) } else { 0 };
        let kept: std::collections::HashSet<usize> = copy_order[..started].iter().copied().collect();
        for e in 0..experts.len() {
            on_host[e] =
                count[e] > 0 && count[e] <= h.max_rows && !in_place(e) && !kept.contains(&e) && experts[e].up != Fmt::Tern && experts[e].down != Fmt::Tern;
        }
        if full {
            let rest: Vec<usize> = copy_order[started..].iter().copied().filter(|&e| !on_host[e]).collect();
            copy_order.truncate(started);
            copy_order.extend(rest);
            issued.truncate(copy_order.len().div_ceil(MICRO));
            issued.resize_with(copy_order.len().div_ceil(MICRO), || None);
        }
    }
    if !full {
        // few tokens: only the experts that have pairs, after the router
        (0..experts.len()).filter(|&e| count[e] > 0 && !on_host[e]).for_each(|e| if in_place(e) { res_order.push(e) } else { copy_order.push(e) });
    }
    let copy_chunks: Vec<&[usize]> = copy_order.chunks(MICRO).collect();
    if !full {
        *issued = (0..copy_chunks.len()).map(|_| None).collect();
        if kern.copiers.is_some() {
            for (c, chunk) in copy_chunks.iter().enumerate().take(ring.n) {
                issued[c] = Some(issue(kern, ring, experts, cs, first, c, chunk)?);
            }
        }
    }
    ring.next = first + copy_chunks.len();
    let smem = (kern.stages * 24576) as u32;
    let (cp, bp) = (ring.codes.ptr, ring.books.ptr);
    // the products of a set of experts that have pairs (`members`, with their places): the launches of at most CAP_ROWS pairs and 16 segments each, one for each width of codes present
    let run = |members: &[usize], places: &[Place]| -> Result<()> {
        for (sg, widths) in launches(members, &count, &start, places, CAP_ROWS) {
            if let Some((ux, dx)) = kern.mixed.as_ref() {
                let mut sg = sg;
                for (i, w) in widths.iter().enumerate() {
                    sg.ubits[i] = w.0 as i32;
                    sg.dbits[i] = w.1 as i32;
                }
                let tiles = sg.tiles[sg.n as usize] as u32;
                crate::tmark!(s, "experts: before up");
                // (segments of many pairs: 128 a block, the tiles counted again for it)
                let rows: i32 = sg.rows[..sg.n as usize].iter().sum();
                let least = crate::tune!("GLYD_CB2_UP128_ROWS").and_then(|v| v.parse::<i32>().ok()).unwrap_or(UP128_ROWS);
                match kern.up128.as_ref().filter(|_| rows >= least * sg.n) {
                    Some(k) => {
                        let mut s128 = sg;
                        for i in 0..sg.n as usize {
                            s128.tiles[i + 1] = s128.tiles[i] + (sg.rows[i] as usize).div_ceil(128) as i32;
                        }
                        launch!(k, [s128.tiles[sg.n as usize] as u32, (inter / NEURONS) as u32, 1], [256, 1, 1], smem (2 * UP128_STAGE) as u32, s; xrb, sc.tok_d.ptr, cp, bp, act.ptr, s128, kern.limit)?;
                    }
                    None => launch!(ux, [tiles, (inter / NEURONS) as u32, 1], [128, 1, 1], smem smem, s; xrb, sc.tok_d.ptr, cp, bp, act.ptr, sg, kern.limit)?,
                }
                crate::tmark!(s, "experts: up");
                launch!(dx, [tiles, (hid / kern.dout) as u32, 1], [kern.dout as u32, 1, 1], smem (kern.stages * downx_stage(kern.dout)) as u32, s; act.ptr, sc.tok_d.ptr, sc.wt_d.ptr, cp, bp, yrot, sg)?;
                crate::tmark!(s, "experts: down");
                continue;
            }
            for bits in 2..=4usize {
                if let Some(part) = pick(&sg, |i| widths[i].0 == bits) {
                    let tiles = part.tiles[part.n as usize] as u32;
                    launch!(kern.up[bits - 2], [tiles, (inter / NEURONS) as u32, 1], [128, 1, 1], smem smem, s; xrb, sc.tok_d.ptr, cp, bp, act.ptr, part, kern.limit)?;
                }
            }
            for bits in 2..=4usize {
                if let Some(part) = pick(&sg, |i| widths[i].1 == bits) {
                    let tiles = part.tiles[part.n as usize] as u32;
                    launch!(kern.down[bits - 2], [tiles, (hid / OUTPUTS) as u32, 1], [128, 1, 1], smem smem, s; act.ptr, sc.tok_d.ptr, sc.wt_d.ptr, cp, bp, yrot, part)?;
                }
            }
        }
        Ok(())
    };
    // the host's share on a thread of its own (its pairs from the plan, x' from the readback), beside the GPU's loop; its sums go up and are added after the GPU's products
    let (hx, hy, hyd) =
        (g.host_x.as_ref().map_or(0, |b| b.ptr as usize), g.host_y.as_ref().map_or(0, |b| b.ptr as usize), g.host_yd.as_ref().map_or(0, |b| b.ptr));
    let (limit, ex) = (kern.limit, SendPtr((experts.as_ptr(), experts.len())));
    let t_loop = std::time::Instant::now();
    let (host_res, gpu_ms) = std::thread::scope(|scope| -> Result<(Option<(usize, usize, f64)>, f64)> {
        let job = share.map(|h| {
            let (on_host, start, tok, wt) = (&on_host, &start, &*tok, &*wt);
            // SAFETY: the pinned buffers hold n rows (made above), x' read back with the pairs before the plan; nothing else touches them until this call's end.
            let (x, y) = unsafe { (std::slice::from_raw_parts(hx as *const u16, n * hid), std::slice::from_raw_parts_mut(hy as *mut f32, n * hid)) };
            scope.spawn(move || {
                let t0 = std::time::Instant::now();
                // SAFETY: the caller's experts, alive for the call (the scope ends before it returns), their bytes host memory.
                let (ep, en) = ex.get();
                let experts = unsafe { std::slice::from_raw_parts(ep, en) };
                unsafe { host_part(experts, on_host, start, tok, wt, x, y, hid, limit, h.threads) }.map(|(e, p)| (e, p, t0.elapsed().as_secs_f64() * 1e3))
            })
        });
        // the experts already on the GPU first: their products run while the first copies are under way
        for chunk in res_order.chunks(MICRO) {
            let members: Vec<usize> = chunk.iter().copied().filter(|&e| count[e] > 0).collect();
            if members.is_empty() {
                continue;
            }
            let places: Vec<Place> = members
                .iter()
                .map(|&e| {
                    let d = &experts[e];
                    let (rc_dev, rb_dev) = d.resident.expect("a resident expert");
                    let (dr, dbk) = (d.inter * d.down.row_bytes(d.hid), d.inter * d.down.entries() * 2);
                    // (in the expert cache's place: offsets from the ring's start, which the kernels add to its address)
                    Place {
                        ucode: (rc_dev + dr as u64) as i64 - cp as i64,
                        dcode: rc_dev as i64 - cp as i64,
                        ubook: (rb_dev + dbk as u64) as i64 - bp as i64,
                        dbook: rb_dev as i64 - bp as i64,
                        ubits: bits_of(d.up) as u8,
                        dbits: bits_of(d.down) as u8,
                    }
                })
                .collect();
            run(&members, &places)?;
        }
        crate::tmark!(s, "experts: resident");
        for c in 0..copy_chunks.len() {
            let chunk = copy_chunks[c];
            advise(experts, copy_chunks.get(c + ring.n + 1).copied());
            let iss = match issued[c].take() {
                Some(i) => i,
                None => issue(kern, ring, experts, cs, first, c, chunk)?,
            };
            match &iss.done {
                Some(d) => d.wait()?,
                None => ring.ready[iss.slot].wait(s)?,
            }
            for t in &iss.terns {
                if hid == HID {
                    launch!(kern.tern, [((t[4] * 205).div_ceil(256)) as u32, 1, 1], [256, 1, 1], s; t[0], t[1], t[2], t[3], t[4] as i32)?;
                } else {
                    launch!(kern.tern_rows, [((t[4] as usize * (hid / 20)).div_ceil(256)) as u32, 1, 1], [256, 1, 1], s; t[0], t[1], t[2], t[3], t[4] as i32, hid as i32)?;
                }
            }
            let have: Vec<usize> = (0..chunk.len()).filter(|&i| count[chunk[i]] > 0).collect();
            if !have.is_empty() {
                let members: Vec<usize> = have.iter().map(|&i| chunk[i]).collect();
                let places: Vec<Place> = have.iter().map(|&i| iss.places[i]).collect();
                run(&members, &places)?;
            }
            ring.free[iss.slot].record(s)?;
            // the slot is the next copy's once its kernels have run: the copier thread waits for that, and this thread goes on
            if kern.copiers.is_some() && c + ring.n < copy_chunks.len() && issued[c + ring.n].is_none() {
                issued[c + ring.n] = Some(issue(kern, ring, experts, cs, first, c + ring.n, copy_chunks[c + ring.n])?);
            }
        }
        let gpu_ms = t_loop.elapsed().as_secs_f64() * 1e3;
        let r = match job {
            Some(j) => Some(j.join().map_err(|_| Error("the host's share of the experts panicked".into()))??),
            None => None,
        };
        Ok((r, gpu_ms))
    })?;
    if let Some((experts_h, pairs_h, host_ms)) = host_res {
        copy_h2d_async(hyd, hy as *const c_void, n * hid * 4, s)?;
        launch!(kern.add, [((n * hid) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; yrot, hyd, (n * hid) as i64)?;
        kern.took.set(HostTook { experts: experts_h, pairs: pairs_h, host_ms, gpu_ms });
        if let Some(h) = share.filter(|h| h.auto) {
            // (the host's time against the GPU loop's: more rows where the host waited, fewer where the GPU did)
            let (ratio, r) = (host_ms / gpu_ms.max(1e-3), h.max_rows as f64);
            let next = if ratio < 0.7 {
                r * 1.2 + 1.0
            } else if ratio > 1.0 {
                r * 0.8
            } else {
                r
            };
            kern.host.set(Some(HostShare { max_rows: (next as usize).clamp(1, 64), ..h }));
        }
    }
    crate::tmark!(s, "experts: streamed");
    Ok(())
}

/// A value sent to another thread whose owner vouches for it (a slice of the caller's expert descriptors: raw pointers to host memory that outlives the thread's scope).
#[derive(Clone, Copy)]
struct SendPtr<T>(T);
// SAFETY: see above; the pointers are only read.
unsafe impl<T> Send for SendPtr<T> {}

impl<T: Copy> SendPtr<T> {
    /// The value (a method, so that a closure takes the wrapper whole and not its field).
    fn get(self) -> T {
        self.0
    }
}

/// The expert as the host's kernels read it (src/cpu_cb.rs): its bytes where the descriptor says (codes of its downT then its up, then their books), its formats, the SwiGLU's clamp.
///
/// # Safety
/// The descriptor's bytes are host memory that outlives the view.
unsafe fn host_view(d: &ExpertDesc, limit: f32) -> Result<crate::cpu_cb::GlmExpert<'static>> {
    use crate::cpu_cb::{GlmExpert, GlmMat};
    let (cb, bb) = d.bytes();
    let (codes, books) = (std::slice::from_raw_parts(d.codes, cb), std::slice::from_raw_parts(d.books, bb));
    let (dc, uc) = codes.split_at(d.inter * d.down.row_bytes(d.hid));
    let (db, ub) = books.split_at(d.inter * d.down.entries() * 2);
    GlmExpert::mixed(
        GlmMat { fmt: d.up, rows: 2 * d.inter, cols: d.hid, codes: uc, book: ub },
        GlmMat { fmt: d.down, rows: d.inter, cols: d.hid, codes: dc, book: db },
        limit,
    )
}

/// The host's share of a call: the pairs of the experts `on_host` (the plan's tokens and weights, an expert's from `start`) computed from their bytes, the weighted rows added up by token into y [n, hid] (zeroed first).
/// The experts and the pairs it took.
///
/// # Safety
/// As `host_view` for every expert taken.
#[allow(clippy::too_many_arguments)]
unsafe fn host_part(
    experts: &[ExpertDesc],
    on_host: &[bool],
    start: &[usize],
    tok: &[i32],
    wt: &[f32],
    x: &[u16],
    y: &mut [f32],
    hid: usize,
    limit: f32,
    threads: usize,
) -> Result<(usize, usize)> {
    use crate::cpu_cb::GlmPair;
    let which: Vec<usize> = (0..experts.len()).filter(|&e| on_host[e]).collect();
    let views = which.iter().map(|&e| host_view(&experts[e], limit)).collect::<Result<Vec<_>>>()?;
    let mut pairs = Vec::new();
    for (v, &e) in views.iter().zip(&which) {
        for p in start[e]..start[e + 1] {
            pairs.push(GlmPair { expert: v, token: tok[p] as usize, weight: wt[p] });
        }
    }
    y.fill(0.0);
    if pairs.is_empty() {
        return Ok((0, 0));
    }
    let mut out = vec![0f32; pairs.len() * hid];
    if crate::tune!("GLYD_CB2_HOST_PAIRWISE").is_some() {
        crate::cpu_cb::experts_pinned(crate::cpu_hb::checked_kernel(), crate::cpu_experts::Z::Bf16(x), &pairs, &mut out, threads, false)?;
    } else {
        crate::cpu_cb::experts_grouped(crate::cpu_hb::checked_kernel(), crate::cpu_experts::Z::Bf16(x), &pairs, &mut out, threads)?;
    }
    for (pair, row) in pairs.iter().zip(out.chunks(hid)) {
        y[pair.token * hid..(pair.token + 1) * hid].iter_mut().zip(row).for_each(|(a, b)| *a += b);
    }
    Ok((which.len(), pairs.len()))
}

/// The micro-batches whose copies are under way, waited for when the call lets go of them: a call that ends on an error leaves no copier thread waiting on an event of a ring that a later call or
/// `release_buffers` destroys (a destroyed event in cuEventSynchronize: the process's end).
struct InFlight(Vec<Option<Issued>>);

impl std::ops::Deref for InFlight {
    type Target = Vec<Option<Issued>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for InFlight {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        for iss in self.0.iter().flatten() {
            if let Some(d) = &iss.done {
                let _ = d.wait();
            }
        }
    }
}

/// A micro-batch whose copies have been started: its ring slot, where each of its experts is, the ternary matrices to make two-bit rows, and (copier threads) the handle to wait for the copies by.
struct Issued {
    slot: usize,
    places: Vec<Place>,
    terns: Vec<[u64; 5]>,
    done: Option<Arc<Done>>,
}

/// Asks the system for the pages of the experts of a micro-batch.
fn advise(experts: &[ExpertDesc], batch: Option<&[usize]>) {
    for &e in batch.unwrap_or(&[]).iter().filter(|&&e| experts[e].resident.is_none()) {
        let (c, bk) = experts[e].bytes();
        will_need(experts[e].codes, c);
        will_need(experts[e].books, bk);
    }
}

/// Where the experts `members` of micro-batch slot `slot` are in the ring (the offsets of their matrices' codes and books in the kernels' form), the copies that put them there (device address, host address, bytes),
/// and the ternary matrices to be made two-bit rows (raw codes, raw books, codes, books, rows).
fn plan_batch(ring: &Ring, experts: &[ExpertDesc], slot: usize, members: &[usize]) -> (Vec<Place>, Vec<(u64, usize, usize)>, Vec<[u64; 5]>) {
    let (cbase, bbase) = (ring.codes.ptr + (slot * ring.slot_codes) as u64, ring.books.ptr + (slot * ring.slot_books) as u64);
    let (mut rc, mut rb) = (
        ring.raw_codes.as_ref().map_or(0, |x| x.ptr) + (slot * ring.raw_slot_codes) as u64,
        ring.raw_books.as_ref().map_or(0, |x| x.ptr) + (slot * ring.raw_slot_books) as u64,
    );
    let (mut co, mut bo) = (0usize, 0usize);
    let (mut places, mut copies, mut terns) = (Vec::new(), Vec::new(), Vec::new());
    for &e in members {
        let d = &experts[e];
        let (kc, kb) = d.kbytes();
        let (dcode, ucode) = (co, co + d.inter * kform(d.down).row_bytes(d.hid));
        let (dbook, ubook) = (bo, bo + d.inter * kform(d.down).entries() * 2);
        places.push(Place {
            ucode: (slot * ring.slot_codes + ucode) as i64,
            dcode: (slot * ring.slot_codes + dcode) as i64,
            ubook: (slot * ring.slot_books + ubook) as i64,
            dbook: (slot * ring.slot_books + dbook) as i64,
            ubits: bits_of(d.up) as u8,
            dbits: bits_of(d.down) as u8,
        });
        if d.up != Fmt::Tern && d.down != Fmt::Tern {
            let (c, bk) = d.bytes();
            copies.push((cbase + co as u64, d.codes as usize, c));
            copies.push((bbase + bo as u64, d.books as usize, bk));
        } else {
            // per matrix: the downT rows come first in the host's bytes, the up rows after them
            let (dr, dbk) = (d.inter * d.down.row_bytes(d.hid), d.inter * d.down.entries() * 2);
            for (fmt, rows, code_src, book_src, code_dst, book_dst) in [
                (d.down, d.inter, d.codes as usize, d.books as usize, dcode, dbook),
                (d.up, 2 * d.inter, d.codes as usize + dr, d.books as usize + dbk, ucode, ubook),
            ] {
                let (cl, bl) = (rows * fmt.row_bytes(d.hid), rows * fmt.entries() * 2);
                if fmt == Fmt::Tern {
                    copies.push((rc, code_src, cl));
                    copies.push((rb, book_src, bl));
                    terns.push([rc, rb, cbase + code_dst as u64, bbase + book_dst as u64, rows as u64]);
                    rc += cl as u64;
                    rb += bl as u64;
                } else {
                    copies.push((cbase + code_dst as u64, code_src, cl));
                    copies.push((bbase + book_dst as u64, book_src, bl));
                }
            }
        }
        co += kc;
        bo += kb;
    }
    (places, copies, terns)
}

/// Starts the copies of micro-batch `c` (the experts `members`) to its slot, which is `(first + c) % ring.n`: on the copier threads, which wait for the kernels that last read the slot, or on the copy stream, which does.
///
/// # Safety
/// As `cb2_prefill_experts`.
unsafe fn issue(kern: &Cb2Kern, ring: &Ring, experts: &[ExpertDesc], cs: Stream, first: usize, c: usize, members: &[usize]) -> Result<Issued> {
    let slot = (first + c) % ring.n;
    let (places, copies, terns) = plan_batch(ring, experts, slot, members);
    let copies = merged(copies, MERGED_PIECE);
    let done = if let Some(cp) = kern.copiers.as_ref() {
        let w = cp.tx.len().min(copies.len().max(1));
        let done = Done::new(w);
        for t in 0..w {
            let mine: Vec<(u64, usize, usize)> = copies.iter().copied().skip(t).step_by(w).collect();
            cp.tx[t].send(Job { copies: mine, after: vec![ring.free[slot].as_ref()], done: done.clone(), stage: !kern.pinned.get() }).expect("a copier thread");
        }
        Some(done)
    } else {
        // the copy stream waits for the kernels that last read the slot
        ring.free[slot].wait(cs)?;
        for &(dst, src, len) in copies.iter().filter(|_| !kern.nocopy) {
            // SAFETY: the host holds the experts' bytes (the caller's word) and the slot has room for each.
            copy_h2d_async(dst, src as *const c_void, len, cs)?;
        }
        ring.ready[slot].record(cs)?;
        None
    };
    Ok(Issued { slot, places, terns, done })
}

/// The copies (device address, host address, bytes) with the ones that continue each other on both sides made one (the pack's experts are end to end, codes and books apart, as a slot's are: a micro-batch of
/// consecutive experts is two copies, not two an expert; each a driver call with its own cost), then cut into pieces of at most `piece` bytes for the copier threads to share.
fn merged(mut copies: Vec<(u64, usize, usize)>, piece: usize) -> Vec<(u64, usize, usize)> {
    copies.sort_unstable_by_key(|c| c.0);
    let mut out: Vec<(u64, usize, usize)> = Vec::new();
    for (dst, src, len) in copies {
        match out.last_mut() {
            Some(l) if l.0 + l.2 as u64 == dst && l.1 + l.2 == src => l.2 += len,
            _ => out.push((dst, src, len)),
        }
    }
    out.into_iter().flat_map(|(dst, src, len)| (0..len).step_by(piece.max(1)).map(move |o| (dst + o as u64, src + o, piece.min(len - o)))).collect()
}

/// The pieces a micro-batch's copies are cut into for the copier threads (`merged`; 1, 2, 4 and 16 MB measured alike).
const MERGED_PIECE: usize = 2 << 20;

/// The launches of a micro-batch: `batch` are its experts (numbers in the layer) and `places` where they are in the ring, `count` and `start` the pairs of every expert of the layer and where its first is in the plan; a
/// launch holds at most `cap` pairs (an expert with more is split across launches) and 16 segments, a segment being an expert's matrices, its first pair and its pairs, the row of the activations it starts at and
/// the prefix of its tiles. With each launch go the bits of the codes of each segment's up and down matrix.
fn launches(batch: &[usize], count: &[usize], start: &[usize], places: &[Place], cap: usize) -> Vec<(Segs, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let (mut at, mut done) = (0usize, 0usize); // the expert of the batch, and its pairs already in a launch
    while at < batch.len() {
        let mut sg = Segs::default();
        let mut widths = Vec::new();
        let mut rows_in = 0usize;
        while at < batch.len() && (sg.n as usize) < 16 && rows_in < cap {
            let e = batch[at];
            let take = (count[e] - done).min(cap - rows_in);
            let i = sg.n as usize;
            let pl = places[at];
            sg.ucode[i] = pl.ucode;
            sg.dcode[i] = pl.dcode;
            sg.ubook[i] = pl.ubook;
            sg.dbook[i] = pl.dbook;
            widths.push((pl.ubits as usize, pl.dbits as usize));
            sg.begin[i] = (start[e] + done) as i32;
            sg.rows[i] = take as i32;
            sg.hoff[i] = rows_in as i32;
            sg.tiles[i + 1] = sg.tiles[i] + take.div_ceil(TILE) as i32;
            sg.n += 1;
            rows_in += take;
            done += take;
            if done == count[e] {
                at += 1;
                done = 0;
            }
        }
        out.push((sg, widths));
    }
    out
}

/// The segments of `sg` that `keep` takes (by index), as launch arguments of their own: their rows in the activations stay where they are, the prefix of tiles is of the ones kept. None if none is.
fn pick(sg: &Segs, keep: impl Fn(usize) -> bool) -> Option<Segs> {
    let mut out = Segs::default();
    for i in (0..sg.n as usize).filter(|&i| keep(i)) {
        let j = out.n as usize;
        out.begin[j] = sg.begin[i];
        out.rows[j] = sg.rows[i];
        out.hoff[j] = sg.hoff[i];
        out.ucode[j] = sg.ucode[i];
        out.dcode[j] = sg.dcode[i];
        out.ubook[j] = sg.ubook[i];
        out.dbook[j] = sg.dbook[i];
        out.tiles[j + 1] = out.tiles[j] + (sg.tiles[i + 1] - sg.tiles[i]);
        out.n += 1;
    }
    (out.n > 0).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel's decode of two codes (kernels/cb2_prefill.cu `c2_w2`) with prmt as the PTX manual has it: the bytes of {y:x} picked by the selector's nibbles.
    fn byte_perm(x: u32, y: u32, s: u32) -> u32 {
        let b = ((y as u64) << 32) | x as u64;
        (0..4).fold(0u32, |o, i| {
            let nib = (s >> (4 * i)) & 15;
            let v = (b >> (8 * (nib & 7))) as u8;
            o | (v as u32) << (8 * i)
        })
    }

    #[test]
    fn two_codes_select_two_entries() {
        let mut r = 0x1234_5678_9ABC_DEF1u64;
        let mut next = || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r
        };
        for _ in 0..2000 {
            let (r01, r23, w) = (next() as u32, (next() >> 3) as u32, (next() >> 5) as u32);
            let e = [r01 & 0xFFFF, r01 >> 16, r23 & 0xFFFF, r23 >> 16];
            for p in 0..8 {
                let v = (w >> (4 * p)) & 15;
                let got = byte_perm(r01, r23, ((v * 65) & 0x303) * 34 + 0x1010);
                let (c0, c1) = ((w >> (4 * p)) & 3, (w >> (4 * p + 2)) & 3);
                assert_eq!(got, e[c0 as usize] | e[c1 as usize] << 16, "word {w:#x} pair {p}");
            }
        }
    }

    /// Copies that continue each other on both sides become one and are cut into pieces; the others stay apart; every byte is copied once to where it was going.
    #[test]
    fn copies_merge_where_they_continue() {
        let copies = vec![(1000u64, 50usize, 10usize), (0, 0, 10), (10, 10, 30), (40, 41, 5), (1010, 60, 6)];
        let m = merged(copies.clone(), 16);
        assert_eq!(m, vec![(0, 0, 16), (16, 16, 16), (32, 32, 8), (40, 41, 5), (1000, 50, 16)]);
        let bytes =
            |v: &[(u64, usize, usize)]| v.iter().flat_map(|&(d, s, l)| (0..l).map(move |i| (d + i as u64, s + i))).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(bytes(&m), bytes(&copies));
    }

    #[test]
    fn launches_cover_every_pair_once() {
        let count = [0usize, 70, 5, 0, 130, 64, 1000, 3];
        let mut start = vec![0usize; count.len() + 1];
        for e in 0..count.len() {
            start[e + 1] = start[e] + count[e];
        }
        let batch = [1usize, 2, 4, 5, 6, 7];
        // (a place for each expert of the batch: codes at 1,000 times its number, widths 2, 3 and 4 in turn)
        let places: Vec<Place> = (0..batch.len())
            .map(|i| Place {
                ucode: 1000 * i as i64 + 7,
                dcode: 1000 * i as i64,
                ubook: 10 * i as i64 + 1,
                dbook: 10 * i as i64,
                ubits: 2 + (i % 3) as u8,
                dbits: 2 + ((i + 1) % 3) as u8,
            })
            .collect();
        for cap in [64usize, 100, 300, 5000] {
            let ls = launches(&batch, &count, &start, &places, cap);
            let mut seen = vec![0u8; start[count.len()]];
            for (sg, widths) in &ls {
                assert!(sg.n >= 1 && sg.n <= 16 && widths.len() == sg.n as usize);
                let mut rows = 0;
                for i in 0..sg.n as usize {
                    // the segment's offsets are one of the batch's experts', its rows start where the previous ones end
                    let j = (sg.dcode[i] / 1000) as usize;
                    assert_eq!((sg.ucode[i], sg.ubook[i], sg.dbook[i]), (1000 * j as i64 + 7, 10 * j as i64 + 1, 10 * j as i64));
                    assert_eq!(widths[i], (places[j].ubits as usize, places[j].dbits as usize));
                    assert_eq!(sg.hoff[i] as usize, rows);
                    rows += sg.rows[i] as usize;
                    assert_eq!(sg.tiles[i + 1] - sg.tiles[i], (sg.rows[i] as usize).div_ceil(TILE) as i32);
                    let e = batch[j];
                    assert!(sg.begin[i] as usize >= start[e] && (sg.begin[i] + sg.rows[i]) as usize <= start[e + 1]);
                    for p in sg.begin[i]..sg.begin[i] + sg.rows[i] {
                        seen[p as usize] += 1;
                    }
                }
                assert!(rows <= cap);
                // the parts by the up's width: every segment in exactly one, the tiles of a part the sum of its segments'
                let mut parts = 0;
                for bits in 2..=4 {
                    if let Some(part) = pick(sg, |i| widths[i].0 == bits) {
                        parts += part.n;
                        assert_eq!(
                            part.tiles[part.n as usize],
                            (0..sg.n as usize).filter(|&i| widths[i].0 == bits).map(|i| sg.tiles[i + 1] - sg.tiles[i]).sum::<i32>()
                        );
                    }
                }
                assert_eq!(parts, sg.n);
            }
            let want: Vec<u8> = (0..seen.len()).map(|p| (batch.iter().any(|&e| p >= start[e] && p < start[e + 1])) as u8).collect();
            assert_eq!(seen, want, "cap {cap}: every pair of the batch in exactly one segment");
        }
    }
}

#[cfg(feature = "dev")]
pub use bench::run;

#[cfg(feature = "dev")]
mod bench {
    use super::*;
    use crate::args::Args;
    use crate::cpu_cb::{experts_pinned, GlmExpert, GlmPair};
    use crate::cpu_experts::Z;
    use std::time::Instant;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn f(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
        }
    }

    fn bf16(v: f32) -> u16 {
        let b = v.to_bits();
        ((b.wrapping_add(0x7FFF + ((b >> 16) & 1))) >> 16) as u16
    }

    /// A layer file of a pack mapped read-only (the codes' and the books' offsets as `GlmMoeCb` finds them: the first kept expert's tensors, the others end to end after them).
    struct Mapped {
        ptr: *mut u8,
        #[cfg_attr(windows, allow(dead_code))]
        len: usize,
    }

    #[cfg(unix)]
    extern "C" {
        fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut std::ffi::c_void;
        fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    }

    impl Drop for Mapped {
        fn drop(&mut self) {
            // SAFETY: mapped in `map_layer`, unmapped once.
            #[cfg(unix)]
            unsafe {
                munmap(self.ptr as *mut std::ffi::c_void, self.len)
            };
            #[cfg(windows)]
            unsafe {
                crate::os::win::unmap(self.ptr.cast())
            };
        }
    }

    fn map_layer(dir: &str, layer: usize) -> Result<(Mapped, Cb2Host)> {
        use glyd_gpu::safetensors::File;
        #[cfg(unix)]
        use std::os::unix::io::AsRawFd;
        let path = std::path::Path::new(dir).join(format!("layer-{layer:03}.safetensors"));
        let f = File::open(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let mut kept: Vec<usize> =
            f.tensors.iter().filter_map(|(n, _)| n.strip_suffix(".u.0").and_then(|e| e.strip_prefix('e')).and_then(|e| e.parse().ok())).collect();
        kept.sort();
        let first = *kept.first().ok_or_else(|| Error("a layer file with no expert".into()))?;
        let at = |e: usize, what: &str| -> Result<usize> {
            Ok(f.get(&format!("e{e:03}.{what}")).ok_or_else(|| Error(format!("no tensor e{e:03}.{what}")))?.offset as usize)
        };
        let (b0, c0) = (at(first, "d.1")?, at(first, "d.0")?);
        for (i, &e) in kept.iter().enumerate() {
            if at(e, "d.1")? != b0 + i * BOOKS
                || at(e, "u.1")? != b0 + i * BOOKS + 2048 * 8
                || at(e, "d.0")? != c0 + i * CODES
                || at(e, "u.0")? != c0 + i * CODES + 2048 * 1024
            {
                return err(format!("expert {e}'s tensors are not where the {i}th of a run end to end are"));
            }
        }
        let file = std::fs::File::open(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let len = file.metadata().map_err(|e| Error(format!("{}: {e}", path.display())))?.len() as usize;
        // SAFETY: a read-only private mapping of a file held open; it outlives the descriptor.
        #[cfg(unix)]
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) } as *mut u8;
        #[cfg(unix)]
        if ptr as usize == usize::MAX {
            return err(format!("{}: cannot be mapped", path.display()));
        }
        #[cfg(windows)]
        let ptr = crate::os::win::map_read(&file).map_err(|e| Error(format!("{}: cannot be mapped: {e}", path.display())))? as *mut u8;
        // SAFETY: inside the mapping (the offsets are from the file's header, checked by File::open to be within it).
        let host = unsafe { Cb2Host { codes: ptr.add(c0), books: ptr.add(b0), experts: kept.len(), cstride: CODES, bstride: BOOKS } };
        Ok((Mapped { ptr, len }, host))
    }

    /// `--copytest 1`: the ways of getting a layer's experts to the GPU, timed: a pageable copy of a micro-batch (eight experts) from the host's memory on the caller's thread and on 1 to 4 threads, from pinned memory, and from
    /// the layer's range registered in place (what registering costs, and the copies from it).
    fn copytest(dev: &Dev, host: &Cb2Host, file_backed: bool) -> Result<()> {
        use std::time::Instant;
        let block = MICRO * CODES;
        let blocks = host.experts / MICRO;
        let dst = Buf::new(block)?;
        let side = Side::new()?;
        let gbs = |bytes: usize, s: f64| bytes as f64 / s / 1e9;
        // the caller's thread
        let mut t = Vec::new();
        for pass in 0..2 {
            for b in 0..blocks {
                let t0 = Instant::now();
                // SAFETY: the host range is inside the layer's, the device buffer holds a block.
                unsafe { copy_h2d_async(dst.ptr, host.codes.add(b * block) as *const _, block, side.stream)? };
                side.sync()?;
                if pass == 1 {
                    t.push(t0.elapsed().as_secs_f64());
                }
            }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "pageable, the caller's thread: {:.1} GB/s at the median of {} blocks of {} MB ({:.1} ms a block)",
            gbs(block, t[t.len() / 2]),
            t.len(),
            block >> 20,
            t[t.len() / 2] * 1e3
        );
        // threads of their own
        for w in [1usize, 2, 3, 4] {
            let c = Copiers::new(dev.ordinal, w)?;
            let bufs: Vec<Buf> = (0..w).map(|_| Buf::new(block)).collect::<Result<_>>()?;
            for pass in 0..2 {
                let t0 = Instant::now();
                for b in 0..blocks {
                    // the block's eight experts dealt round the threads
                    let done = Done::new(w);
                    for (i, tx) in c.tx.iter().enumerate() {
                        let copies =
                            (i..MICRO).step_by(w).map(|j| (bufs[i].ptr + (j * CODES) as u64, host.codes as usize + (b * MICRO + j) * CODES, CODES)).collect();
                        tx.send(Job { copies, after: Vec::new(), done: done.clone(), stage: false }).expect("a copier");
                    }
                    done.wait()?;
                }
                if pass == 1 {
                    println!("pageable, {w} copier threads: {:.1} GB/s over {} blocks", gbs(blocks * block, t0.elapsed().as_secs_f64()), blocks);
                }
            }
        }
        // through pinned buffers of our own: threads copy a block from the host's memory into one, the DMA of the one before runs meanwhile
        for w in [2usize, 4, 8] {
            const BUFS: usize = 3;
            let pins: Vec<Pinned> = (0..BUFS).map(|_| Pinned::new(block)).collect::<Result<_>>()?;
            let evs: Vec<Ev> = (0..BUFS).map(|_| Ev::new()).collect::<Result<_>>()?;
            let t0 = Instant::now();
            for b in 0..blocks {
                let i = b % BUFS;
                evs[i].wait_host()?;
                let src = host.codes as usize + b * block;
                let dstp = pins[i].ptr as usize;
                std::thread::scope(|sc| {
                    for t in 0..w {
                        sc.spawn(move || {
                            let (lo, hi) = (block / w * t, if t + 1 == w { block } else { block / w * (t + 1) });
                            // SAFETY: disjoint ranges of the pinned block, and of the host's bytes (the caller's word).
                            unsafe { std::ptr::copy_nonoverlapping((src + lo) as *const u8, (dstp + lo) as *mut u8, hi - lo) };
                        });
                    }
                });
                // SAFETY: a pinned block and the device block of the same size.
                unsafe { copy_h2d_async(dst.ptr, pins[i].ptr, block, side.stream)? };
                evs[i].record(side.stream)?;
            }
            side.sync()?;
            println!("through pinned buffers, {w} threads copying: {:.1} GB/s", gbs(blocks * block, t0.elapsed().as_secs_f64()));
        }
        // pinned
        let pin = Pinned::new(block)?;
        let t0 = Instant::now();
        for _ in 0..blocks {
            // SAFETY: a pinned block and a device block of the same size.
            unsafe { copy_h2d_async(dst.ptr, pin.ptr, block, side.stream)? };
        }
        side.sync()?;
        println!("pinned: {:.1} GB/s", gbs(blocks * block, t0.elapsed().as_secs_f64()));
        // registered in place: the pages of the layer's codes range
        let page = 4096usize;
        let lo = (host.codes as usize + page - 1) / page * page;
        let hi = (host.codes as usize + host.experts * CODES) / page * page;
        for flags in [if file_backed { 8u32 } else { 0 }] {
            let t0 = Instant::now();
            // SAFETY: a page-aligned range inside the host's memory, registered once and unregistered below.
            let r = unsafe { crate::dev::register_host(lo as *mut c_void, hi - lo, flags) };
            let reg = t0.elapsed().as_secs_f64();
            match r {
                Err(e) => println!("registering {} MB in place (flags {flags}): {}", (hi - lo) >> 20, e.msg),
                Ok(()) => {
                    println!(
                        "registering {} MB in place (flags {flags}): {:.1} ms ({:.2} ms a GB)",
                        (hi - lo) >> 20,
                        reg * 1e3,
                        reg * 1e3 / ((hi - lo) as f64 / 1e9)
                    );
                    let t0 = Instant::now();
                    let mut moved = 0usize;
                    let mut off = lo;
                    while off + block <= hi {
                        // SAFETY: inside the registered range; the device block holds `block` bytes.
                        unsafe { copy_h2d_async(dst.ptr, off as *const c_void, block, side.stream)? };
                        off += block;
                        moved += block;
                    }
                    side.sync()?;
                    println!("copies from the registered range: {:.1} GB/s", gbs(moved, t0.elapsed().as_secs_f64()));
                    let t0 = Instant::now();
                    // SAFETY: registered above; the copies from it are done (synchronized).
                    unsafe { crate::dev::unregister_host(lo as *mut c_void)? };
                    println!("unregistering: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        Ok(())
    }

    pub fn run(a: &Args) -> Result<()> {
        let dev = Dev::new(0)?;
        let kern = Cb2Kern::load(&dev)?;
        let (n, k, check, reps, skew) = (a.num("n", 1024usize)?, a.num("k", 8usize)?, a.num("check", 16usize)?, a.num("reps", 3usize)?, a.num("skew", 0.0f64)?);
        let mut r = Rng(a.num("seed", 1u64)?.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        // the experts: a layer of a pack, or random bytes of the formats asked for (--fmts cb2:0.5,tern:0.3,cb3:0.2, or X+Y: up and down; the share of each)
        let mut bytes_v: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mapped;
        let (descs, host): (Vec<ExpertDesc>, Cb2Host) = if let Some(dir) = a.flag("pack") {
            let (m, h) = map_layer(dir, a.num("layer", 3usize)?)?;
            mapped = Some(m);
            (h.descs(), h)
        } else {
            mapped = None;
            let ne = a.num("experts", 216usize)?;
            let mut specs: Vec<(Fmt, Fmt, f64)> = Vec::new();
            for part in a.flag("fmts").unwrap_or("cb2:1").split(',') {
                let (name, share) = part.split_once(':').unwrap_or((part, "1"));
                let (u, d) = name.split_once('+').unwrap_or((name, name));
                let f = |x: &str| Fmt::parse(x).ok_or_else(|| Error(format!("--fmts {part}: tern, cb2, cb3 or cb4 (X+Y: the up's and the down's)")));
                specs.push((f(u)?, f(d)?, share.parse().map_err(|_| Error(format!("--fmts {part}: a number after the colon")))?));
            }
            let total: f64 = specs.iter().map(|x| x.2).sum();
            let mut descs = Vec::new();
            for _ in 0..ne {
                let mut pick = r.f(0.0, 1.0) as f64 * total;
                let sp = specs.iter().find(|x| {
                    pick -= x.2;
                    pick <= 0.0
                });
                let (up, down) = sp.map_or((specs[0].0, specs[0].1), |x| (x.0, x.1));
                let probe = ExpertDesc { codes: std::ptr::null(), books: std::ptr::null(), up, down, resident: None, hid: HID, inter: INTER };
                let (cb, bb) = probe.bytes();
                let mut codes = vec![0u8; cb];
                codes.chunks_mut(8).for_each(|c| c.copy_from_slice(&r.next().to_le_bytes()[..c.len()]));
                // (a ternary byte is five base-3 digits: 0 to 242)
                let dn = 2048 * row_bytes(down);
                if down == Fmt::Tern {
                    codes[..dn].iter_mut().for_each(|b| *b %= 243);
                }
                if up == Fmt::Tern {
                    codes[dn..].iter_mut().for_each(|b| *b %= 243);
                }
                let books: Vec<u8> = (0..bb / 2).flat_map(|_| bf16(r.f(-0.06, 0.06)).to_le_bytes()).collect();
                bytes_v.push((codes, books));
                descs.push(probe);
            }
            for (d, (c, b)) in descs.iter_mut().zip(&bytes_v) {
                d.codes = c.as_ptr();
                d.books = b.as_ptr();
            }
            let h = Cb2Host { codes: std::ptr::null(), books: std::ptr::null(), experts: ne, cstride: CODES, bstride: BOOKS };
            (descs, h)
        };
        let _ = &mapped;
        // --resident F: a share F of the experts (the ones without ternary matrices) is put on the GPU first, as the expert cache's places hold experts, and passed by address
        let mut descs = descs;
        let mut on_gpu: Vec<(Buf, Buf)> = Vec::new();
        let share = a.num("resident", 0.0f64)?;
        if share > 0.0 {
            for (i, d) in descs.iter_mut().enumerate() {
                if d.up == Fmt::Tern || d.down == Fmt::Tern || (i as f64 * 0.618_033_988_75).fract() >= share {
                    continue;
                }
                let (cb, bb) = d.bytes();
                let (c, b) = (Buf::new(cb)?, Buf::new(bb)?);
                // SAFETY: the host bytes of the descriptor, as long as the buffers are.
                unsafe {
                    c.write(0, std::slice::from_raw_parts(d.codes, cb))?;
                    b.write(0, std::slice::from_raw_parts(d.books, bb))?;
                }
                d.resident = Some((c.ptr, b.ptr));
                on_gpu.push((c, b));
            }
            println!("{} experts of {} are on the GPU already", on_gpu.len(), descs.len());
        }
        let mix = |f: Fmt| descs.iter().filter(|d| d.up == f || d.down == f).count();
        println!(
            "{} (sm_{}{}): {} experts{} (tern {}, cb2 {}, cb3 {}, cb4 {} matrices' experts), a chunk of {n} tokens with {k} pairs each, {} stages",
            dev.name,
            dev.cc.0,
            dev.cc.1,
            descs.len(),
            if mapped.is_some() { " of a mapped layer" } else { " of random bytes" },
            mix(Fmt::Tern),
            mix(Fmt::Cb2),
            mix(Fmt::Cb3),
            mix(Fmt::Cb4),
            kern.stages
        );
        if a.num("copytest", 0usize)? == 1 {
            let h = if mapped.is_some() { host } else { return err("--copytest needs --pack DIR") };
            return copytest(&dev, &h, true);
        }
        // the tokens: x' bf16, k distinct experts each (a skew > 0 makes the low numbers likelier), weights
        let x: Vec<u16> = (0..n * HID).map(|_| bf16(r.f(-1.0, 1.0))).collect();
        let mut ids = vec![0i64; n * k];
        let mut wts = vec![0f32; n * k];
        for t in 0..n {
            let mut chosen: Vec<usize> = Vec::new();
            while chosen.len() < k {
                let u = r.f(0.0, 1.0) as f64;
                let e = (((if skew > 0.0 { u.powf(1.0 + skew * 3.0) } else { u }) * descs.len() as f64) as usize).min(descs.len() - 1);
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            for (c, &e) in chosen.iter().enumerate() {
                ids[t * k + c] = e as i64;
                wts[t * k + c] = r.f(0.1, 0.5);
            }
        }
        let (xb, idb, wb, yb) = (Buf::new(x.len() * 2)?, Buf::new(ids.len() * 8)?, Buf::new(wts.len() * 4)?, Buf::zeroed(n * HID * 4)?);
        xb.write(0, &x)?;
        idb.write(0, &ids)?;
        wb.write(0, &wts)?;
        let used: Vec<usize> = (0..descs.len()).filter(|e| ids.contains(&(*e as i64))).collect();
        let touched = used.len();
        let moved: usize = used.iter().map(|&e| descs[e].bytes().0 + descs[e].bytes().1).sum();
        let mut times = Vec::new();
        // --busy MS: the stream is kept busy (memory sets, about 0.75 ms each) for MS milliseconds before the call, as a layer's attention would: the copies started at the call's entry run beside it
        let busy = a.num("busy", 0.0f64)?;
        let scrap = if busy > 0.0 { Some(Buf::new(512 << 20)?) } else { None };
        for rep in 0..reps + 1 {
            dev.sync()?;
            let t0 = Instant::now();
            if let Some(b) = &scrap {
                for _ in 0..(busy / 0.75).ceil() as usize {
                    b.fill_async(1, dev.stream)?;
                }
            }
            // SAFETY: live buffers of the sizes the call reads and writes, host bytes that outlive it.
            unsafe { cb2_prefill_experts(&dev, &kern, &descs, n, k, xb.ptr, idb.ptr, wb.ptr, yb.ptr, dev.stream)? };
            dev.sync()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if rep > 0 {
                times.push(ms);
            }
            println!("  call {rep}: {ms:.1} ms");
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[times.len() / 2];
        let flops = (n * k) as f64 * 2.0 * (4096.0 * 4096.0 + 2048.0 * 4096.0);
        println!(
            "median {med:.1} ms for {} pairs over {touched} experts: {:.1} TFLOPS of the pairs' products, {:.1} GB/s of the experts' bytes ({:.0} MB)",
            n * k,
            flops / med / 1e9,
            moved as f64 / med / 1e6,
            moved as f64 / 1e6
        );
        // a few tokens against the host kernels
        let mut y = vec![0f32; n * HID];
        yb.read(0, &mut y)?;
        let mut worst = 0f64;
        let step = (n / check.max(1)).max(1);
        for t in (0..n).step_by(step).take(check) {
            let z: Vec<f32> = x[t * HID..(t + 1) * HID].iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            // SAFETY: the host bytes live as long as the descriptors' memory (`bytes_v` or the mapping), which outlive this loop.
            let experts: Vec<GlmExpert> = (0..k)
                .map(|c| {
                    let d = &descs[ids[t * k + c] as usize];
                    let (cb, bb) = d.bytes();
                    let (cp, bp) = unsafe { (std::slice::from_raw_parts(d.codes, cb), std::slice::from_raw_parts(d.books, bb)) };
                    let (dc, uc) = cp.split_at(2048 * row_bytes(d.down));
                    let (db, ub) = bp.split_at(2048 * d.down.entries() * 2);
                    let mat = |fmt: Fmt, rows: usize, codes: &[u8], book: &[u8]| -> crate::cpu_cb::GlmMat<'static> {
                        // SAFETY: as above.
                        unsafe {
                            crate::cpu_cb::GlmMat {
                                fmt,
                                rows,
                                cols: HID,
                                codes: std::mem::transmute::<&[u8], &'static [u8]>(codes),
                                book: std::mem::transmute::<&[u8], &'static [u8]>(book),
                            }
                        }
                    };
                    GlmExpert::mixed(mat(d.up, 4096, uc, ub), mat(d.down, 2048, dc, db), 10.0)
                })
                .collect::<Result<_>>()?;
            let pr: Vec<GlmPair> = experts.iter().enumerate().map(|(c, e)| GlmPair { expert: e, token: 0, weight: wts[t * k + c] }).collect();
            let mut out = vec![0f32; k * HID];
            experts_pinned(crate::cpu_hb::checked_kernel(), Z::F32(&z), &pr, &mut out, 8, false)?;
            let (mut e2, mut n2) = (0f64, 0f64);
            for o in 0..HID {
                let want: f64 = (0..k).map(|c| out[c * HID + o] as f64).sum();
                let got = y[t * HID + o] as f64;
                e2 += (got - want) * (got - want);
                n2 += want * want;
            }
            let rel = (e2 / n2).sqrt();
            worst = worst.max(rel);
        }
        let ok = worst < 2e-3;
        println!("{} {} tokens against the host kernels: worst {worst:.2e} of the row's norm (bound 2e-3)", if ok { "ok  " } else { "FAIL" }, check.min(n));
        let _ = (&bytes_v, &on_gpu);
        if ok {
            Ok(())
        } else {
            err("the chunk's sums are not within their bound")
        }
    }
}
