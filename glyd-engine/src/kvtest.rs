//! A known answer for the KV cache's write and read path, with no model and no scheduler: bf16 values made on the host, written into blocks of every layer the three ways the engine writes them (a run of tokens, a token at a
//! time, pieces), read back and compared bit for bit with what was written. The values are every kind a cache sees: in the window of their (layer, head, channel) and out of it, both zeros, subnormals, infinities, NaNs of several
//! payloads, the largest and smallest numbers, blocks with few exceptions, with as many as a block holds in its own room and its first extra memory, and with more (kept as they are in the second extra memory), and the 65,536
//! bit patterns once each. Then every block is written again (from its first token, and from its middle), compared again, and released: the extra memory must be empty at the end.

use crate::dev::Buf;
use crate::ffi::{err, Result};
use crate::model::Engine;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// `k` distinct indices of 0 .. n, in a random order.
    fn pick(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut all: Vec<usize> = (0..n).collect();
        let k = k.min(n);
        for i in 0..k {
            let j = i + self.below(n - i);
            all.swap(i, j);
        }
        all.truncate(k);
        all
    }
}

/// How a block's tokens are written: all of them in one call, one call a token (what a decode step does), or in pieces of 5, 7 and 4 tokens.
#[derive(Clone, Copy, Debug)]
enum Style {
    Run,
    OneByOne,
    Pieces,
}

/// What a block holds: values in their windows with `exceptions` of them out of it (and the kinds of value in `SPECIALS` among them where `specials`), or, for `Patterns(half)`, the half-th run of the 65,536 bit patterns (every
/// one once over the two blocks of a layer).
#[derive(Clone, Copy)]
enum Content {
    Window { exceptions: usize, specials: bool },
    Patterns(usize),
}

/// The blocks each layer is given: (what, how written).
const BLOCKS: [(Content, Style); 15] = [
    (Content::Window { exceptions: 0, specials: false }, Style::Run),
    (Content::Window { exceptions: 0, specials: true }, Style::Run),
    (Content::Window { exceptions: 0, specials: true }, Style::OneByOne),
    (Content::Window { exceptions: 40, specials: false }, Style::Pieces),
    (Content::Window { exceptions: 300, specials: true }, Style::Run),
    (Content::Window { exceptions: 300, specials: true }, Style::OneByOne),
    (Content::Window { exceptions: 1000, specials: true }, Style::Run),
    (Content::Window { exceptions: 1000, specials: false }, Style::Pieces),
    (Content::Window { exceptions: 1000, specials: false }, Style::OneByOne),
    (Content::Window { exceptions: 3000, specials: true }, Style::Run),
    (Content::Window { exceptions: 3000, specials: false }, Style::OneByOne),
    (Content::Window { exceptions: 3000, specials: false }, Style::Pieces),
    (Content::Window { exceptions: 32768, specials: true }, Style::Run),
    (Content::Patterns(0), Style::Run),
    (Content::Patterns(1), Style::OneByOne),
];

/// Values of every special kind: both zeros, subnormals (the largest and the smallest of each sign), the smallest normal numbers, the largest, both infinities, NaNs of several payloads, 1 and -1.
const SPECIALS: [u16; 20] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0xff81, 0x7fff, 0xffff, 0x3f80,
    0xbf80,
];

/// The first block of the test's blocks (they are 15 consecutive ones of the cache).
const FIRST_BLOCK: i32 = 40;

/// What the compared values were, by kind, and how many were out of their windows.
#[derive(Default, Clone, Copy)]
struct Kinds {
    values: u64,
    zeros: u64,
    neg_zeros: u64,
    subnormals: u64,
    infinities: u64,
    nans: u64,
    outside: u64,
}

impl Kinds {
    fn add(&mut self, v: u16, inside: bool) {
        self.values += 1;
        let (exp, man) = ((v >> 7) & 0xff, v & 0x7f);
        match (exp, man) {
            (0, 0) if v == 0 => self.zeros += 1,
            (0, 0) => self.neg_zeros += 1,
            (0, _) => self.subnormals += 1,
            (255, 0) => self.infinities += 1,
            (255, _) => self.nans += 1,
            _ => {}
        }
        self.outside += !inside as u64;
    }

    fn sum(&mut self, o: &Kinds) {
        self.values += o.values;
        self.zeros += o.zeros;
        self.neg_zeros += o.neg_zeros;
        self.subnormals += o.subnormals;
        self.infinities += o.infinities;
        self.nans += o.nans;
        self.outside += o.outside;
    }

    fn describe(&self) -> String {
        format!(
            "{} values ({} zeros, {} negative zeros, {} subnormals, {} infinities, {} NaNs, {} outside their windows)",
            self.values, self.zeros, self.neg_zeros, self.subnormals, self.infinities, self.nans, self.outside
        )
    }
}

/// Whether a value is inside its channel's window (its exponent >> 1 is in base .. base + 8).
fn inside(v: u16, base: u8) -> bool {
    (base as u16..base as u16 + 8).contains(&((v >> 8) & 0x7f))
}

/// A block's keys and values as the write takes them ([16 tokens, H heads, D channels] each) for the windows `kb` and `vb` ([H, D] bytes: the base of each channel's window of 8 values of exponent >> 1); `all` is the layer's
/// shuffled list of the 65,536 bit patterns.
fn make_block(rng: &mut Rng, what: Content, kb: &[u8], vb: &[u8], h: usize, d: usize, all: &[u16]) -> (Vec<u16>, Vec<u16>) {
    let n = 16 * h * d;
    let base_of = |i: usize| -> usize { (if i / n == 0 { kb } else { vb })[(i % n / d) % h * d + i % d] as usize };
    let mut vals: Vec<u16> = match what {
        Content::Patterns(half) => (0..2 * n).map(|i| all[(half * 2 * n + i) % all.len()]).collect(),
        Content::Window { exceptions, specials } => {
            // every value inside its window: a random sign, an exponent >> 1 of the window's 8, a random low byte
            let mut v: Vec<u16> =
                (0..2 * n).map(|i| ((rng.next() & 1) as u16) << 15 | ((base_of(i) + rng.below(8)) as u16) << 8 | (rng.next() & 0xff) as u16).collect();
            // `exceptions` of them moved out of their window (an exponent >> 1 of the other 120, any sign and low byte)
            for i in rng.pick(2 * n, exceptions) {
                let base = base_of(i);
                let mut e7 = rng.below(128);
                while (base..base + 8).contains(&e7) {
                    e7 = rng.below(128);
                }
                v[i] = ((rng.next() & 1) as u16) << 15 | (e7 as u16) << 8 | (rng.next() & 0xff) as u16;
            }
            if specials {
                for (i, x) in rng.pick(2 * n, SPECIALS.len()).into_iter().zip(SPECIALS) {
                    v[i] = x;
                }
            }
            v
        }
    };
    let values = vals.split_off(n);
    (vals, values)
}

/// The block's tokens `from` to 15 written as `style` says, from the device buffers `kd` and `vd` ([16, H, D] each); `slots` holds the block's 16 slots (int64).
///
/// # Safety
/// Live buffers of the engine, the stream in order.
unsafe fn write_block(e: &Engine, l: usize, kd: &Buf, vd: &Buf, slots: &Buf, from: usize, style: Style) -> Result<()> {
    let hd = e.kv.geom.heads as usize * 128;
    let pieces: Vec<(usize, usize)> = match style {
        Style::Run => vec![(from, 16 - from)],
        Style::OneByOne => (from..16).map(|t| (t, 1)).collect(),
        Style::Pieces => {
            let (mut v, mut t) = (Vec::new(), from);
            for len in [5usize, 7, 4].iter().cycle() {
                if t >= 16 {
                    break;
                }
                let m = (*len).min(16 - t);
                v.push((t, m));
                t += m;
            }
            v
        }
    };
    for (t, m) in pieces {
        e.lib.kv_write(
            &e.kv.geom,
            e.kv.pools[l].ptr,
            e.kv.kbase[l].ptr,
            e.kv.vbase[l].ptr,
            l as i64,
            kd.ptr + (t * hd * 2) as u64,
            hd as i64,
            vd.ptr + (t * hd * 2) as u64,
            hd as i64,
            slots.ptr + (t * 8) as u64,
            m as i64,
            e.kv.status.ptr,
            e.kv.lost.ptr,
            e.kv.nlost,
            e.kv.ring,
            e.dev.stream,
        )?;
    }
    Ok(())
}

/// What the compared values were by kind, how many differ, and how many differing ones have been printed.
#[derive(Default)]
struct Tally {
    kinds: Kinds,
    differ: u64,
    shown: usize,
}

/// What the test reads the cache back through, and its buffers.
struct Probe {
    blocks: Buf,
    iota: Buf,
    out: Buf,
    slots: Vec<Buf>,
    h: usize,
    d: usize,
}

impl Probe {
    /// The blocks of layer `l` decoded, each value compared with `want` ([block](keys, values)) in the windows `bases` (keys', values'): the values that differ are added to `t` (the first few printed), the compared values counted by kind.
    fn compare(&self, e: &Engine, l: usize, want: &[(Vec<u16>, Vec<u16>)], bases: (&[u8], &[u8]), t: &mut Tally, label: &str) -> Result<()> {
        let (h, d, nb) = (self.h, self.d, want.len());
        // SAFETY: live buffers of the engine; the stream is synchronized after.
        unsafe {
            e.lib.kv_gather(
                &e.kv.geom,
                e.kv.pools[l].ptr,
                e.kv.kbase[l].ptr,
                e.kv.vbase[l].ptr,
                self.blocks.ptr,
                self.iota.ptr,
                nb as i64,
                self.out.ptr,
                e.dev.stream,
            )?
        };
        e.dev.sync()?;
        let mut got = vec![0u16; nb * 16 * h * 2 * d];
        self.out.read(0, &mut got)?;
        let (kb, vb) = bases;
        for (i, (k, v)) in want.iter().enumerate() {
            for tok in 0..16 {
                for hh in 0..h {
                    for c in 0..d {
                        let at = ((i * 16 + tok) * h + hh) * 2 * d + c;
                        let src = (tok * h + hh) * d + c;
                        for (x, w, base) in [(got[at], k[src], kb), (got[at + d], v[src], vb)] {
                            t.kinds.add(w, inside(w, base[hh * d + c]));
                            if x != w {
                                t.differ += 1;
                                if t.shown < 12 {
                                    t.shown += 1;
                                    println!("DIFFERENT ({label}) layer {l} block {i} token {tok} head {hh} channel {c}: written {w:#06x}, read back {x:#06x}");
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Every block in `data` (keys, values) on the device.
fn upload(data: &[(Vec<u16>, Vec<u16>)]) -> Result<Vec<(Buf, Buf)>> {
    data.iter()
        .map(|(k, v)| -> Result<(Buf, Buf)> {
            let (a, b) = (Buf::new(2 * k.len())?, Buf::new(2 * v.len())?);
            a.write(0, k)?;
            b.write(0, v)?;
            Ok((a, b))
        })
        .collect()
}

/// The known-answer test over every layer of the engine's cache; false where any value differs, a block was lost, or the extra memory is not empty after every block was released.
pub fn run(e: &mut Engine) -> Result<bool> {
    let (h, d, layers) = (e.kv.geom.heads as usize, 128usize, e.kv.layers);
    let nb = BLOCKS.len();
    let hd = h * d;
    if (FIRST_BLOCK as i64) + nb as i64 > e.kv.nblocks {
        return err(format!("the cache has {} blocks, the test needs {}", e.kv.nblocks, FIRST_BLOCK as usize + nb));
    }
    let blocks: Vec<i32> = (0..nb as i32).map(|i| FIRST_BLOCK + i).collect();
    let probe = Probe {
        blocks: Buf::new(4 * nb)?,
        iota: Buf::new(4 * nb)?,
        out: Buf::new(nb * 16 * h * 2 * d * 2)?,
        slots: blocks
            .iter()
            .map(|&b| -> Result<Buf> {
                let s = Buf::new(16 * 8)?;
                s.write(0, &(0..16).map(|t| b as i64 * 16 + t).collect::<Vec<i64>>())?;
                Ok(s)
            })
            .collect::<Result<_>>()?,
        h,
        d,
    };
    probe.blocks.write(0, &blocks)?;
    probe.iota.write(0, &(0..nb as i32).collect::<Vec<_>>())?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut first, mut again, mut mid) = (Tally::default(), Tally::default(), Tally::default());
    let mut shown = 0usize;
    for l in 0..layers {
        let (mut kb, mut vb) = (vec![0u8; hd], vec![0u8; hd]);
        e.kv.kbase[l].read(0, &mut kb)?;
        e.kv.vbase[l].read(0, &mut vb)?;
        let mut all: Vec<u16> = (0..=u16::MAX).collect();
        for i in (1..all.len()).rev() {
            let j = rng.below(i + 1);
            all.swap(i, j);
        }
        let data: Vec<(Vec<u16>, Vec<u16>)> = BLOCKS.iter().map(|b| make_block(&mut rng, b.0, &kb, &vb, h, d, &all)).collect();
        // 1. every block written as its style says
        let dev = upload(&data)?;
        for i in 0..nb {
            // SAFETY: live buffers of the engine; the writes are in stream order before the read.
            unsafe { write_block(e, l, &dev[i].0, &dev[i].1, &probe.slots[i], 0, BLOCKS[i].1)? };
        }
        first.shown = shown;
        probe.compare(e, l, &data, (&kb, &vb), &mut first, "first write")?;
        shown = first.shown;
        // 2. every block written again from its first token, with the values of the block five places on, in the style of the block three places on
        let again_data: Vec<(Vec<u16>, Vec<u16>)> = (0..nb).map(|i| data[(i + 5) % nb].clone()).collect();
        let dev2 = upload(&again_data)?;
        for i in 0..nb {
            // SAFETY: as above.
            unsafe { write_block(e, l, &dev2[i].0, &dev2[i].1, &probe.slots[i], 0, BLOCKS[(i + 3) % nb].1)? };
        }
        again.shown = shown;
        probe.compare(e, l, &again_data, (&kb, &vb), &mut again, "written again")?;
        shown = again.shown;
        // 3. tokens 8 to 15 of every block written again with new values (a sequence rewritten from its middle): tokens 0 to 7 stay what the second write left
        let fresh: Vec<(Vec<u16>, Vec<u16>)> = BLOCKS.iter().map(|b| make_block(&mut rng, b.0, &kb, &vb, h, d, &all)).collect();
        let mut mid_data = again_data.clone();
        for i in 0..nb {
            mid_data[i].0[8 * hd..].copy_from_slice(&fresh[i].0[8 * hd..]);
            mid_data[i].1[8 * hd..].copy_from_slice(&fresh[i].1[8 * hd..]);
        }
        let dev3 = upload(&mid_data)?;
        for i in 0..nb {
            // SAFETY: as above; only tokens 8 to 15 are written.
            unsafe { write_block(e, l, &dev3[i].0, &dev3[i].1, &probe.slots[i], 8, BLOCKS[i].1)? };
        }
        mid.shown = shown;
        probe.compare(e, l, &mid_data, (&kb, &vb), &mut mid, "tokens 8 to 15 written again")?;
        shown = mid.shown;
        // 4. released: the extra memory the blocks held goes back
        e.release_pages(&blocks)?;
    }
    e.dev.sync()?;
    let w = e.kv.status_words()?;
    let mut all = first.kinds;
    all.sum(&again.kinds);
    all.sum(&mid.kinds);
    let (differ, differ_again, differ_mid) = (first.differ, again.differ, mid.differ);
    let total = differ + differ_again + differ_mid;
    println!("known-answer test of the KV cache: {layers} layers, {nb} blocks of 16 tokens of {h} heads in each (keys and values), written as a run of tokens, token by token and in pieces");
    println!("  first writes:       {}, differing {differ}", first.kinds.describe());
    println!("  written again:      {}, differing {differ_again}", again.kinds.describe());
    println!("  tokens 8-15 again:  {}, differing {differ_mid}", mid.kinds.describe());
    println!("  in all: values checked {}, differing {total}", all.values);
    println!(
        "  the cache's status: entries without a place {}, blocks kept in the second extra memory {}, blocks lost {}; extra memory in use after releasing every block: first {} of {}, second {} of {}",
        w[0], w[10], w[11], w[4], e.kv.chunks, w[8], e.kv.slabs
    );
    let ok = total == 0 && w[11] == 0 && w[4] == 0 && w[8] == 0;
    if w[10] == 0 {
        println!("  (no block went to the second extra memory: that way of storing was not exercised)");
    }
    println!("{}", if ok { "known-answer test: every value read back bit for bit" } else { "known-answer test: FAILED" });
    Ok(ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test's blocks are what they say: values in their windows with exactly the exceptions asked for out of them (and the special kinds among them), and every bit pattern once over the two pattern blocks.
    #[test]
    fn blocks_are_made_as_asked() {
        let (h, d) = (2usize, 128usize);
        let n = 16 * h * d;
        let (kb, vb): (Vec<u8>, Vec<u8>) = ((0..h * d).map(|i| (i % 100) as u8).collect(), (0..h * d).map(|i| ((i * 7) % 110) as u8).collect());
        let mut rng = Rng(12345);
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let outside = |k: &[u16], v: &[u16]| -> usize {
            let at = |i: usize| (i % n / d) % h * d + i % d;
            (0..2 * n).filter(|&i| !inside(if i < n { k[i] } else { v[i - n] }, if i < n { kb[at(i)] } else { vb[at(i)] })).count()
        };
        let (k, v) = make_block(&mut rng, Content::Window { exceptions: 0, specials: false }, &kb, &vb, h, d, &all);
        assert_eq!((k.len(), v.len(), outside(&k, &v)), (n, n, 0));
        let (k, v) = make_block(&mut rng, Content::Window { exceptions: 300, specials: false }, &kb, &vb, h, d, &all);
        assert_eq!(outside(&k, &v), 300);
        // the special kinds are all there where asked for
        let (k, v) = make_block(&mut rng, Content::Window { exceptions: 0, specials: true }, &kb, &vb, h, d, &all);
        let has = |x: u16| k.contains(&x) || v.contains(&x);
        assert!(SPECIALS.iter().all(|&x| has(x)));
        // every value an exception
        let (k, v) = make_block(&mut rng, Content::Window { exceptions: 2 * n, specials: false }, &kb, &vb, h, d, &all);
        assert_eq!(outside(&k, &v), 2 * n);
        // two pattern blocks of a bigger model's size hold every pattern once; here (a block is 2 n values) the halves are the list's runs
        let n8 = 16 * 8 * 128usize;
        let (mut seen, mut dup) = (vec![false; 65536], 0);
        let mut all: Vec<u16> = (0..=u16::MAX).collect();
        for i in (1..all.len()).rev() {
            let j = rng.below(i + 1);
            all.swap(i, j);
        }
        for half in 0..2 {
            let (k, v) = make_block(&mut rng, Content::Patterns(half), &vec![0u8; 8 * 128], &vec![0u8; 8 * 128], 8, 128, &all);
            assert_eq!(k.len(), n8);
            for x in k.into_iter().chain(v) {
                dup += seen[x as usize] as usize;
                seen[x as usize] = true;
            }
        }
        assert_eq!((dup, seen.iter().filter(|&&s| s).count()), (0, 65536));
    }
}
