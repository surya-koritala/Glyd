//! Piecewise CUDA graphs (f:cuda-graphs:serving:2: vLLM's PIECEWISE mode, the pieces of a pass between its attention layers captured, the attention left outside). A pass that no
//! whole graph takes (a prompt's chunk, a step of several prompts' chunks, a step of rows that decode and prompts' chunks) runs each run of operations that act row by row
//! (`role`: the products, norms, activations and experts between two attention or recurrent layers) as one graph, and the others (the attention layers, the recurrent layers'
//! operations, which read a sequence's cache or state by its rows) eager between them. The host then queues one launch a piece where it queued every kernel of it.
//!
//! A set of pieces is captured for a pass's exact rows, not for a padded size as vLLM's buckets are: the kernel a product takes turns on its rows (the library's route, cuBLAS's
//! algorithm), so a padded pass would not be the eager pass's arithmetic, and a padding row would be one more row in the step. A size is captured the second time it is met
//! (`Pieces::meet`; the first pass of a size is eager, which also makes every buffer its kernels take), and the `MOST` sets last used are kept. A replay is the eager pass
//! bit for bit: the same kernels with the same arguments in the same order, the operations between the pieces the eager pass's.
//!
//! A scheduler's mixed steps are mostly of one size (a step's prompt budget fills it to a chunk: `ModelStep::prompt_budget`), and so are the chunks of long prompts, so a few sets
//! take most passes. GLYD_ENGINE_PIECES=0 (a development build's) runs every such pass eager, to compare.

use super::Op;
use crate::dev::{Ev, Graph};

/// The sets of pieces kept at most (each a graph a piece, about two a layer).
pub(super) const MOST: usize = 16;

/// The sizes met once and not captured that are remembered (the oldest forgotten past it).
const MET: usize = 64;

/// What an operation is to a pass of pieces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Role {
    /// It runs inside a piece: it acts on each row alone (`rows::row_wise`) and its launches turn on nothing but the pass's rows and the engine's buffers.
    Piece,
    /// It runs eager between two pieces: an operation on a sequence's cache or state, the latent attention (its caches by sequence inside: `Dyn::seq_parts`), the early
    /// guess of experts (one row, read on the host), DeepSeek-V4's (rows laid out by the pass's groups).
    Cut,
    /// It does nothing (`Op::Nop`): neither.
    Nothing,
}

pub(super) fn role(op: &Op) -> Role {
    match op {
        Op::Nop => Role::Nothing,
        Op::Mla(_) | Op::MoeEarly { .. } | Op::Fp8 { .. } | Op::CompQ { .. } | Op::IdxQ { .. } | Op::OGroupLora { .. } | Op::OGroupLoraStack { .. } => {
            Role::Cut
        }
        op if super::rows::row_wise(op) => Role::Piece,
        _ => Role::Cut,
    }
}

/// The pieces a description's operations make (its first operation, then its layers' in order): the runs of `Role::Piece` operations between the others.
pub(super) fn count<'a>(ops: impl Iterator<Item = &'a Op>) -> usize {
    let (mut n, mut open) = (0, false);
    for op in ops {
        match role(op) {
            Role::Piece if !open => (n, open) = (n + 1, true),
            Role::Cut => open = false,
            _ => {}
        }
    }
    n
}

/// How `Dyn::forward` runs a pass: every operation eager; its pieces captured (nothing run: the operations between the pieces are not queued, nor the head); or each piece's
/// graph of set `i` launched in its place and the rest eager.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Mode {
    Eager,
    Capture,
    Replay(usize),
}

/// A pass of pieces being made: its mode, the next piece, and whether one is open (begun and not closed).
pub(super) struct Run {
    pub(super) mode: Mode,
    pub(super) k: usize,
    pub(super) open: bool,
}

/// A piece: its graph, and with GLYD_PROF=1 the events captured in it (`Dyn::mark`'s, which each replay records).
pub(super) struct Piece {
    pub(super) graph: Graph,
    pub(super) marks: Vec<(&'static str, Ev)>,
}

/// A pass's pieces for its rows, and when they were last used.
pub(super) struct PieceSet {
    pub(super) rows: usize,
    pub(super) pieces: Vec<Piece>,
    used: u64,
}

/// What a pass of some rows does (`Pieces::meet`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Meet {
    /// Run eager: the first time its size is met.
    Eager,
    /// Capture its size's pieces, then replay them.
    Capture,
    /// Replay set `i`.
    Replay(usize),
}

/// The sets of pieces captured, the sizes met once, and the switches a check sets.
#[derive(Default)]
pub(super) struct Pieces {
    pub(super) sets: Vec<PieceSet>,
    met: Vec<(usize, u64)>,
    tick: u64,
    /// Capture a size the first time it is met (usercheck's comparison of every pass), not the second.
    pub(super) first: bool,
    /// No pieces: a capture failed, or a check runs the passes eager.
    pub(super) off: bool,
}

impl Pieces {
    /// A pass of `rows` rows: its set's replay where there is one; else a capture the second time the size is met (the first where `first`), else eager.
    pub(super) fn meet(&mut self, rows: usize) -> Meet {
        self.tick += 1;
        if let Some(i) = self.sets.iter().position(|s| s.rows == rows) {
            self.sets[i].used = self.tick;
            return Meet::Replay(i);
        }
        if let Some(i) = self.met.iter().position(|m| m.0 == rows) {
            self.met.swap_remove(i);
            return Meet::Capture;
        }
        if self.first {
            return Meet::Capture;
        }
        if self.met.len() == MET {
            let oldest = (0..self.met.len()).min_by_key(|&i| self.met[i].1).expect("met is full");
            self.met.swap_remove(oldest);
        }
        self.met.push((rows, self.tick));
        Meet::Eager
    }

    /// The set captured for `rows` kept, the least recently used dropped where `MOST` are (a graph in flight is freed when it completes): its index.
    pub(super) fn add(&mut self, rows: usize, pieces: Vec<Piece>) -> usize {
        if self.sets.len() == MOST {
            let lru = (0..self.sets.len()).min_by_key(|&i| self.sets[i].used).expect("sets are full");
            self.sets.swap_remove(lru);
        }
        self.sets.push(PieceSet { rows, pieces, used: self.tick });
        self.sets.len() - 1
    }

    /// Every set dropped (the buffers they read are made again), and the sizes met.
    pub(super) fn clear(&mut self) {
        self.sets.clear();
        self.met.clear();
    }
}

impl super::Dyn {
    /// Whether passes run their pieces as graphs: the engine captures graphs (no block of experts that reads the host), on one GPU, no layer in the computer's memory, no
    /// corpus pass recording the routes, no probe, and not turned off (GLYD_ENGINE_PIECES=0, a check's, or a capture that failed).
    fn pieces_on(&self) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| crate::tune!("GLYD_ENGINE_PIECES").is_some_and(|v| v == "0") || crate::tune!("GLYD_OP_PROBE").is_some());
        !off && !self.pieces.off
            && self.rgraph_on
            && self.big.ram.is_none()
            && self.stage.first
            && self.stage.last
            && super::tp::comm().is_none()
            && !crate::hb_usage::recording()
    }

    /// usercheck's switches, the sets kept dropped: every pass eager (`off`), or in pieces captured the first time a size is met (`first`). Whether passes may run in pieces
    /// at all here (`pieces_on` with the switches set so).
    #[cfg(feature = "dev")]
    pub(crate) fn pieces_for_check(&mut self, first: bool, off: bool) -> bool {
        self.pieces.clear();
        (self.pieces.first, self.pieces.off) = (first, false);
        let on = self.pieces_on();
        self.pieces.off = off;
        on
    }

    /// A pass of the big model on its `n` rows (the embedding queued), which no whole graph takes: its pieces' replay (`Pieces::meet`), captured first where this is the
    /// size's second time; else eager.
    pub(super) unsafe fn pieces_pass(&mut self, n: usize, pos: (u64, u64), head: usize) -> crate::ffi::Result<()> {
        if !self.pieces_on() {
            return self.forward(&self.big, n, pos, head);
        }
        let i = match self.pieces.meet(n) {
            Meet::Eager => return self.forward(&self.big, n, pos, head),
            Meet::Replay(i) => i,
            Meet::Capture => match self.pieces_capture(n, pos, head) {
                Ok(set) => self.pieces.add(n, set),
                Err(e) => {
                    // (nothing ran: the pass eager, and every pass from here)
                    eprintln!("glyd: a pass of {n} rows could not be captured in pieces ({e}): such passes run eager");
                    self.pieces.off = true;
                    self.pieces.clear();
                    return self.forward(&self.big, n, pos, head);
                }
            },
        };
        self.pmode.set(Mode::Replay(i));
        let r = self.forward(&self.big, n, pos, head);
        self.pmode.set(Mode::Eager);
        self.pplayed.set(Some(i));
        self.piece_replays.set(self.piece_replays.get() + 1);
        r
    }

    /// The pieces of a pass of `n` rows captured (nothing run), as many as the description makes (`count`).
    unsafe fn pieces_capture(&mut self, n: usize, pos: (u64, u64), head: usize) -> crate::ffi::Result<Vec<Piece>> {
        // (--prof: the events of the pass so far kept aside, the capture's kept with its pieces)
        let before = self.prof.as_ref().map(|p| std::mem::take(&mut *p.borrow_mut()));
        self.pmode.set(Mode::Capture);
        let r = self.forward(&self.big, n, pos, head);
        self.pmode.set(Mode::Eager);
        if self.popen.replace(false) {
            // (an error inside a piece: its capture ended, the graph dropped)
            self.capturing.set(false);
            let _ = self.dev.end_capture();
        }
        if let (Some(p), Some(b)) = (&self.prof, before) {
            let left = std::mem::replace(&mut *p.borrow_mut(), b);
            self.spare.borrow_mut().extend(left.into_iter().map(|x| x.1));
        }
        let set = std::mem::take(&mut *self.pcap.borrow_mut());
        r?;
        let d = &self.big.d;
        let want = count(std::iter::once(&d.first).chain(d.layers.iter().flatten()));
        if set.len() != want {
            return crate::ffi::err(format!("{} pieces captured, where the description makes {want}", set.len()));
        }
        Ok(set)
    }

    /// Op `i` of layer `l` of `p` in a pass `pc` of pieces: an operation of a piece captured (a capture) or left to the piece's graph (a replay), the piece opened first where
    /// none is: true, it is done. An operation that cuts the pieces closes the one open: false, the caller runs it, except in a capture (true: nothing else is queued).
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn in_piece(
        &self,
        pc: &mut Run,
        p: &super::Part,
        l: usize,
        i: usize,
        op: &Op,
        n: usize,
        pos: (u64, u64),
        head: usize,
    ) -> crate::ffi::Result<bool> {
        match (pc.mode, role(op)) {
            (Mode::Eager, _) => Ok(false),
            (_, Role::Nothing) => Ok(true),
            (_, Role::Cut) => {
                self.piece_close(pc)?;
                Ok(pc.mode == Mode::Capture)
            }
            (Mode::Capture, Role::Piece) => {
                if !pc.open {
                    self.dev.begin_capture()?;
                    self.popen.set(true);
                    self.capturing.set(true);
                    pc.open = true;
                }
                self.op(p, l, i, op, n, pos, head)?;
                Ok(true)
            }
            (Mode::Replay(set), Role::Piece) => {
                if !pc.open {
                    let piece = self
                        .pieces
                        .sets
                        .get(set)
                        .and_then(|s| s.pieces.get(pc.k))
                        .ok_or_else(|| crate::ffi::Error(format!("piece {} of set {set}: none captured", pc.k)))?;
                    // (--prof: its events go among the pass's eager ones before the next)
                    if let Some(evs) = &self.prof {
                        self.psplice.borrow_mut().push((evs.borrow().len(), pc.k));
                    }
                    piece.graph.launch(self.dev.stream)?;
                    self.pcount.set(self.pcount.get() + 1);
                    pc.open = true;
                }
                Ok(true)
            }
        }
    }

    /// The piece open in `pc` closed: a capture's graph made and kept (with its events: `Dyn::mark`'s nodes); the next piece is the next one.
    pub(super) fn piece_close(&self, pc: &mut Run) -> crate::ffi::Result<()> {
        if !pc.open {
            return Ok(());
        }
        (pc.open, pc.k) = (false, pc.k + 1);
        if pc.mode == Mode::Capture {
            self.capturing.set(false);
            self.popen.set(false);
            let graph = self.dev.end_capture()?;
            let marks = self.prof.as_ref().map(|p| std::mem::take(&mut *p.borrow_mut())).unwrap_or_default();
            self.pcap.borrow_mut().push(Piece { graph, marks });
        }
        Ok(())
    }
}

/// A replayed pass's events in the order the stream recorded them: the eager ones (`evs`), and before the eager event at `at` of each `(at, k)` of `put`, piece k's (`piece`).
pub(super) fn splice<'a, T>(
    evs: &'a [(&'static str, T)],
    put: &[(usize, usize)],
    piece: impl Fn(usize) -> &'a [(&'static str, T)],
) -> Vec<(&'static str, &'a T)> {
    let mut out = Vec::with_capacity(evs.len() + 8 * put.len());
    let mut put = put.iter().peekable();
    for j in 0..=evs.len() {
        while let Some(&&(_, k)) = put.peek().filter(|x| x.0 == j) {
            out.extend(piece(k).iter().map(|(l, e)| (*l, e)));
            put.next();
        }
        if let Some((l, e)) = evs.get(j) {
            out.push((*l, e));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic::{DepthMixOp, Slot};

    /// A layer of products and norms around a recurrent layer's convolution and delta rule (Qwen3.5's linear attention), twice: a piece runs from the operation after
    /// a cut to the next cut, so the two layers make three pieces (before the first convolution, between the first delta rule and the second convolution, after the last).
    #[test]
    fn the_operations_between_cuts_are_the_pieces() {
        let lin = || Op::Linear { w: 0, x: Slot::H, y: Slot::O, name: "product" };
        let norm = || Op::NormAdd { y: Slot::O, w: 0, plus1: false };
        let layer = [
            norm(),
            lin(),
            Op::GdnConv { w: 0, state: 0, cd: 8, row: 8 },
            Op::GdnDelta { a_log: 0, dt_bias: 0, state: 1, cd: 8, nk: 1, nv: 1 },
            lin(),
            norm(),
            lin(),
            lin(),
        ];
        let first = Op::Nop;
        let ops: Vec<&Op> = std::iter::once(&first).chain(layer.iter()).chain(layer.iter()).collect();
        assert_eq!(count(ops.into_iter()), 3);
        assert_eq!((role(&layer[2]), role(&layer[3])), (Role::Cut, Role::Cut));
        // (a no-op neither opens a piece nor cuts one)
        assert_eq!(role(&Op::Nop), Role::Nothing);
        assert_eq!(count([Op::Nop, Op::Nop].iter()), 0);
        assert_eq!(count([lin(), Op::Nop, lin()].iter()), 1);
        // (an operation that acts row by row but reads the host: a cut)
        let mix = DepthMixOp { add: false, nb: 1, w_n: 0, w_p: 0, push: None, w: 0, eps: 1e-6 };
        assert_eq!(role(&Op::MoeEarly { block: 0, mix }), Role::Cut);
        assert_eq!(role(&lin()), Role::Piece);
    }

    /// A size is captured the second time it is met (the first with `first`), replayed from then on; the least recently used set goes when `MOST` are kept.
    #[test]
    fn a_size_is_captured_when_met_again_and_the_least_used_set_goes() {
        let mut p = Pieces::default();
        assert_eq!(p.meet(512), Meet::Eager);
        assert_eq!(p.meet(300), Meet::Eager);
        assert_eq!(p.meet(512), Meet::Capture);
        assert_eq!(p.add(512, Vec::new()), 0);
        assert_eq!(p.meet(512), Meet::Replay(0));
        assert_eq!(p.meet(300), Meet::Capture, "remembered as met once");
        p.add(300, Vec::new());
        p.first = true;
        for n in 2..MOST {
            assert_eq!(p.meet(1000 + n), Meet::Capture);
            p.add(1000 + n, Vec::new());
        }
        assert_eq!(p.sets.len(), MOST);
        // (512 used again: 300 is the least recently used)
        assert_eq!(p.meet(512), Meet::Replay(0));
        assert_eq!(p.meet(7), Meet::Capture);
        p.add(7, Vec::new());
        assert_eq!(p.sets.len(), MOST);
        assert!(p.sets.iter().all(|s| s.rows != 300) && p.sets.iter().any(|s| s.rows == 512), "the least recently used set went");
        p.clear();
        p.first = false;
        assert_eq!(p.meet(512), Meet::Eager);
    }

    /// Sizes met once are remembered up to `MET`, the oldest forgotten first.
    #[test]
    fn the_oldest_size_met_once_is_forgotten() {
        let mut p = Pieces::default();
        for n in 0..=MET {
            assert_eq!(p.meet(n + 1), Meet::Eager);
        }
        assert_eq!(p.meet(1), Meet::Eager, "the first size was forgotten");
        assert_eq!(p.meet(MET + 1), Meet::Capture);
    }

    /// The pieces' events go before the eager event recorded after their launch: eager e0, piece 0's (a, b), eager e1, e2, piece 1's (c), and a piece after the last eager one.
    #[test]
    fn a_replays_events_are_in_the_streams_order() {
        let evs = [("e0", 0u32), ("e1", 1), ("e2", 2)];
        let pieces = [vec![("a", 10u32), ("b", 11)], vec![("c", 12)], vec![("d", 13)]];
        let got: Vec<&str> = splice(&evs, &[(1, 0), (3, 1)], |k| &pieces[k]).into_iter().map(|x| x.0).collect();
        assert_eq!(got, ["e0", "a", "b", "e1", "e2", "c"]);
        let got: Vec<u32> = splice(&evs, &[(0, 2), (1, 0)], |k| &pieces[k]).into_iter().map(|x| *x.1).collect();
        assert_eq!(got, [13, 0, 10, 11, 1, 2]);
    }
}
