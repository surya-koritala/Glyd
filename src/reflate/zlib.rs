//! zlib's `deflate_slow` (levels 4–9), run over the plain text to
//! predict the stream's tokens, and the corrections where the stream
//! differs.
//!
//! The emulation follows deflate.c: a rolling 15-bit hash of three
//! bytes, `head` and `prev` chains indexed by the low 15 bits of the
//! position, position 0 unmatchable (it is zlib's NIL), matches no
//! farther than `MAX_DIST`, `longest_match` with the level's chain,
//! good, nice and lazy limits, lazy evaluation one position ahead, the
//! TOO_FAR rule for 3-byte matches, and a block every 16,383 tokens.
//! Bytes past the end of the input compare as zero.

use super::{coder::{Bit, Decoder, Encoder}, Token};

const MIN_MATCH: u32 = 3;
const MAX_MATCH: u32 = 258;
const W_SIZE: u32 = 32768;
const MIN_LOOKAHEAD: u32 = MAX_MATCH + MIN_MATCH + 1;
const MAX_DIST: u32 = W_SIZE - MIN_LOOKAHEAD;
const TOO_FAR: u32 = 4096;
const HASH_MASK: u32 = 0x7fff;
/// Tokens per block: zlib flushes when its buffer of 16,384 holds one less.
pub const BLOCK_TOKENS: u32 = 16383;

/// `(good, lazy, nice, chain)` per level, deflate.c's table.
const CONFIG: [(u32, u32, u32, u32); 10] = [
    (0, 0, 0, 0),
    (4, 4, 8, 4),
    (4, 5, 16, 8),
    (4, 6, 32, 32),
    (4, 4, 16, 16),
    (8, 16, 32, 32),
    (8, 16, 128, 128),
    (8, 32, 128, 256),
    (32, 128, 258, 1024),
    (32, 258, 258, 4096),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub level: u8,
    pub filtered: bool,
}

/// zlib's matcher state over a plain text.
pub struct Zlib<'a> {
    plain: &'a [u8],
    p: Params,
    head: Vec<u32>,
    prev: Vec<u32>,
    ins_h: u32,
    /// every position below this has been inserted
    inserted: u32,
    /// the match found one position ahead by a lazy search: (length, start)
    carry: Option<(u32, u32)>,
}

impl<'a> Zlib<'a> {
    pub fn new(plain: &'a [u8], p: Params) -> Self {
        let mut z = Zlib { plain, p, head: vec![0; 1 << 15], prev: vec![0; 1 << 15], ins_h: 0, inserted: 0, carry: None };
        if plain.len() >= 2 {
            z.ins_h = ((plain[0] as u32) << 5) ^ plain[1] as u32;
            z.ins_h &= HASH_MASK;
        }
        z
    }

    #[inline]
    fn at(&self, i: u32) -> u8 {
        self.plain.get(i as usize).copied().unwrap_or(0)
    }

    #[inline]
    fn lookahead(&self, pos: u32) -> u32 {
        self.plain.len() as u32 - pos
    }

    /// Positions up to `to` (exclusive) into the chains, as zlib inserts
    /// each `strstart` with at least MIN_MATCH bytes left.
    fn insert_to(&mut self, to: u32) {
        while self.inserted < to {
            let pos = self.inserted;
            if self.lookahead(pos) >= MIN_MATCH {
                self.ins_h = ((self.ins_h << 5) ^ self.at(pos + 2) as u32) & HASH_MASK;
                self.prev[(pos & HASH_MASK) as usize] = self.head[self.ins_h as usize];
                self.head[self.ins_h as usize] = pos;
            }
            self.inserted += 1;
        }
    }

    /// The chain's head for `pos` after its insertion: the previous
    /// position with the same hash, or 0.
    fn hash_head(&mut self, pos: u32) -> u32 {
        self.insert_to(pos + 1);
        if self.lookahead(pos) < MIN_MATCH {
            return 0;
        }
        self.prev[(pos & HASH_MASK) as usize]
    }

    /// deflate.c's longest_match: the longest match at `pos` longer than
    /// `prev_len`, starting from `cur` and following the chain.
    fn longest_match(&self, pos: u32, mut cur: u32, prev_len: u32) -> (u32, u32) {
        let (good, _, nice, chain) = CONFIG[self.p.level as usize];
        let mut chain_length = chain;
        if prev_len >= good {
            chain_length >>= 2;
        }
        let lookahead = self.lookahead(pos);
        let nice = nice.min(lookahead);
        let limit = if pos > MAX_DIST { pos - MAX_DIST } else { 0 };
        let mut best_len = prev_len;
        let mut best_start = 0u32;
        loop {
            if self.at(cur + best_len) == self.at(pos + best_len) && self.at(cur + best_len - 1) == self.at(pos + best_len - 1) && self.at(cur) == self.at(pos) && self.at(cur + 1) == self.at(pos + 1) {
                let mut len = 2u32;
                while len < MAX_MATCH && self.at(cur + len) == self.at(pos + len) {
                    len += 1;
                }
                if len > best_len {
                    best_start = cur;
                    best_len = len;
                    if len >= nice {
                        break;
                    }
                }
            }
            cur = self.prev[(cur & HASH_MASK) as usize];
            chain_length -= 1;
            if cur <= limit || chain_length == 0 {
                break;
            }
        }
        (best_len.min(lookahead), best_start)
    }

    /// A match at `pos` as deflate_slow looks for one: `prev_len` the
    /// length to beat (2 at a fresh position); 0 when there is none
    /// worth taking.
    fn search(&mut self, pos: u32, prev_len: u32) -> (u32, u32) {
        let (_, lazy, _, _) = CONFIG[self.p.level as usize];
        let head = self.hash_head(pos);
        if head == 0 || prev_len >= lazy || pos - head > MAX_DIST {
            return (0, 0);
        }
        let (len, start) = self.longest_match(pos, head, prev_len);
        if len <= prev_len {
            return (0, 0);
        }
        if len <= 5 && (self.p.filtered || (len == MIN_MATCH && pos - start > TOO_FAR)) {
            return (0, 0);
        }
        (len, start)
    }

    /// zlib's token at `pos`, from its lazy evaluation.
    pub fn predict(&mut self, pos: u32) -> Token {
        let (len, start) = match self.carry.take() {
            Some(m) => m,
            None => self.search(pos, MIN_MATCH - 1),
        };
        if len < MIN_MATCH {
            return Token::Lit(self.at(pos));
        }
        // Lazy: a longer match one position on makes this a literal.
        let (len2, start2) = self.search(pos + 1, len);
        if len2 > len {
            self.carry = Some((len2, start2));
            return Token::Lit(self.at(pos));
        }
        Token::Ref { len: len as u16, dist: (pos - start) as u16 }
    }

    /// The state moved past the token that was actually at `pos`.
    pub fn commit(&mut self, pos: u32, token: Token) -> u32 {
        match token {
            Token::Lit(_) => {
                self.insert_to(pos + 1);
                pos + 1
            }
            Token::Ref { len, .. } => {
                let len = if len == 259 { 258 } else { len as u32 };
                self.carry = None;
                // zlib inserts up to max_insert = strstart + lookahead - MIN_MATCH.
                let max_insert = self.plain.len() as u32 - MIN_MATCH;
                self.insert_to((pos + len).min(max_insert + 1));
                self.inserted = self.inserted.max(pos + len);
                pos + len
            }
        }
    }

    /// The candidates at `pos` in chain order, for a distance to be
    /// named as the n-th of them.
    fn chain(&mut self, pos: u32) -> impl Iterator<Item = u32> + '_ {
        let mut cur = self.hash_head(pos);
        let limit = if pos > MAX_DIST { pos - MAX_DIST } else { 0 };
        let prev = &self.prev;
        std::iter::from_fn(move || {
            if cur == 0 || cur <= limit {
                return None;
            }
            let d = pos - cur;
            cur = prev[(cur & HASH_MASK) as usize];
            Some(d)
        })
        .take(4096)
    }
}

/// The contexts of the corrections coder.
struct Models {
    same: [Bit; 2],
    is_ref: [Bit; 2],
    len_same: Bit,
    len: Vec<Bit>,
    dist_same: Bit,
    hop_found: Bit,
    hop: [Bit; 64],
    dist: Vec<Bit>,
    block_same: Bit,
    block_count: [Bit; 64],
}

impl Models {
    fn new() -> Self {
        Models { same: [Bit::default(); 2], is_ref: [Bit::default(); 2], len_same: Bit::default(), len: vec![Bit::default(); 512], dist_same: Bit::default(), hop_found: Bit::default(), hop: [Bit::default(); 64], dist: vec![Bit::default(); 1 << 16], block_same: Bit::default(), block_count: [Bit::default(); 64] }
    }
}

fn kind(t: &Token) -> usize {
    matches!(t, Token::Ref { .. }) as usize
}

/// A block as the coder sees it: its tokens, or a stored run of bytes
/// (whose tokens zlib made and threw away; the emulation runs over the
/// bytes with its own predictions, so the state comes out the same).
pub enum Plan<'a> {
    Tokens(&'a [Token]),
    Stored(u32),
}

/// A stored run: the emulation's own tokens committed up to `end`,
/// literals where a match would cross it.
fn run_stored(z: &mut Zlib, mut pos: u32, end: u32) -> u32 {
    while pos < end {
        let t = match z.predict(pos) {
            Token::Ref { len, .. } if pos + len as u32 <= end => Token::Ref { len, dist: 1 },
            _ => Token::Lit(0),
        };
        pos = z.commit(pos, t);
    }
    pos
}

/// The tokens of `blocks` (each a token count and its tokens' slice
/// into one list) against the emulation: the corrections, from which
/// `recreate` gets the tokens back with the plain text alone.
pub fn predict(plain: &[u8], p: Params, blocks: &[Plan]) -> Vec<u8> {
    let mut z = Zlib::new(plain, p);
    let mut m = Models::new();
    let mut e = Encoder::new();
    let mut pos = 0u32;
    for plan in blocks {
        let tokens = match plan {
            Plan::Tokens(t) => *t,
            Plan::Stored(n) => {
                pos = run_stored(&mut z, pos, pos + n);
                continue;
            }
        };
        let predicted = if plain.len() as u32 - pos == 0 { 0 } else { BLOCK_TOKENS };
        let n = tokens.len() as u32;
        // A block's token count: as predicted (16,383, or whatever runs
        // to the end), or given.
        let to_end = tokens.iter().map(|t| match t { Token::Lit(_) => 1, Token::Ref { len, .. } => if *len == 259 { 258 } else { *len as u32 } }).sum::<u32>() == plain.len() as u32 - pos;
        let as_predicted = n == predicted || (n < BLOCK_TOKENS && to_end);
        e.bit(&mut m.block_same, as_predicted as u32);
        if !as_predicted {
            e.count(&mut m.block_count, n);
        }
        for &actual in tokens.iter() {
            let guess = z.predict(pos);
            let same = guess == actual;
            e.bit(&mut m.same[kind(&guess)], same as u32);
            if !same {
                e.bit(&mut m.is_ref[kind(&guess)], kind(&actual) as u32);
                if let Token::Ref { len, dist } = actual {
                    let (glen, gdist) = match guess {
                        Token::Ref { len, dist } => (len, dist),
                        _ => (0, 0),
                    };
                    if glen != 0 {
                        e.bit(&mut m.len_same, (len == glen) as u32);
                    }
                    if len != glen || glen == 0 {
                        e.tree(&mut m.len, 9, len as u32 - 3);
                    }
                    if glen != 0 {
                        e.bit(&mut m.dist_same, (dist == gdist) as u32);
                    }
                    if dist != gdist || glen == 0 {
                        let hop = z.chain(pos).position(|d| d == dist as u32);
                        e.bit(&mut m.hop_found, hop.is_some() as u32);
                        match hop {
                            Some(h) => e.count(&mut m.hop, h as u32),
                            None => e.tree(&mut m.dist, 16, dist as u32 - 1),
                        }
                    }
                }
            }
            pos = z.commit(pos, actual);
        }
    }
    e.finish()
}

/// The tokens back: the emulation with the corrections applied. One
/// token list per plan (`None` a coded block, `Some(n)` a stored run of
/// `n` bytes, whose list is empty).
pub fn recreate(plain: &[u8], p: Params, corrections: &[u8], plans: &[Option<u32>]) -> Option<Vec<Vec<Token>>> {
    let mut z = Zlib::new(plain, p);
    let mut m = Models::new();
    let mut d = Decoder::new(corrections);
    let mut pos = 0u32;
    let mut out = Vec::with_capacity(plans.len());
    for plan in plans {
        if let Some(n) = plan {
            pos = run_stored(&mut z, pos, pos + n);
            out.push(Vec::new());
            continue;
        }
        let as_predicted = d.bit(&mut m.block_same) == 1;
        let n = if as_predicted { BLOCK_TOKENS } else { d.count(&mut m.block_count) };
        let mut tokens = Vec::with_capacity(n.min(BLOCK_TOKENS) as usize);
        for _ in 0..n {
            if as_predicted && pos == plain.len() as u32 {
                break;
            }
            let guess = z.predict(pos);
            let actual = if d.bit(&mut m.same[kind(&guess)]) == 1 {
                guess
            } else if d.bit(&mut m.is_ref[kind(&guess)]) == 0 {
                Token::Lit(*plain.get(pos as usize)?)
            } else {
                let (glen, gdist) = match guess {
                    Token::Ref { len, dist } => (len, dist),
                    _ => (0, 0),
                };
                let len = if glen != 0 && d.bit(&mut m.len_same) == 1 { glen } else { d.tree(&mut m.len, 9) as u16 + 3 };
                let dist = if glen != 0 && d.bit(&mut m.dist_same) == 1 {
                    gdist
                } else if d.bit(&mut m.hop_found) == 1 {
                    let h = d.count(&mut m.hop) as usize;
                    z.chain(pos).nth(h)? as u16
                } else {
                    d.tree(&mut m.dist, 16) as u16 + 1
                };
                Token::Ref { len, dist }
            };
            tokens.push(actual);
            pos = z.commit(pos, actual);
        }
        out.push(tokens);
    }
    (pos == plain.len() as u32).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::super::{parse, tests::{text, zlib_deflate}, Kind};
    use super::*;

    fn tokens_of(kind: &Kind) -> &[Token] {
        match kind {
            Kind::Fixed(t) | Kind::Dynamic(_, t) => t,
            Kind::Stored(_) => &[],
        }
    }

    /// zlib's own streams predicted with few corrections, and every
    /// token back from them; a stream from another matcher (level 1,
    /// emulated as 6) still comes back, with more.
    #[test]
    fn zlib_levels_predicted_and_recreated() {
        let plain = text(1 << 20);
        for (level, filtered) in [(6, false), (4, false), (5, false), (7, false), (8, false), (9, false), (6, true)] {
            let stream = zlib_deflate(&plain, level, if filtered { "Z_FILTERED" } else { "Z_DEFAULT_STRATEGY" });
            let s = parse(&stream).unwrap();
            let blocks: Vec<&[Token]> = s.blocks.iter().map(|b| tokens_of(&b.kind)).collect();
            let plans: Vec<Plan> = blocks.iter().map(|b| Plan::Tokens(b)).collect();
            let p = Params { level: level as u8, filtered };
            let c = predict(&plain, p, &plans);
            let n_tokens: usize = blocks.iter().map(|b| b.len()).sum();
            #[cfg(feature = "deflate")]
            {
                let (r, _) = preflate_rs::preflate_whole_deflate_stream(&stream, &preflate_rs::PreflateConfig::default()).unwrap();
                eprintln!("level {level} filtered {filtered}: stream {} B, {} tokens in {} blocks; corrections ours {} B, preflate {} B", stream.len(), n_tokens, blocks.len(), c.len(), r.corrections.len());
            }
            assert!(c.len() * 200 < stream.len(), "level {level} filtered {filtered}: {} bytes of corrections for {} tokens, {} bytes of stream", c.len(), n_tokens, stream.len());
            let back = recreate(&plain, p, &c, &vec![None; blocks.len()]).unwrap();
            assert!(back.iter().map(|b| &b[..]).eq(blocks.iter().copied()), "level {level}: tokens back");
        }
        let stream = zlib_deflate(&plain, 1, "Z_DEFAULT_STRATEGY");
        let s = parse(&stream).unwrap();
        let blocks: Vec<&[Token]> = s.blocks.iter().map(|b| tokens_of(&b.kind)).collect();
        let plans: Vec<Plan> = blocks.iter().map(|b| Plan::Tokens(b)).collect();
        let p = Params { level: 6, filtered: false };
        let c = predict(&plain, p, &plans);
        let back = recreate(&plain, p, &c, &vec![None; blocks.len()]).unwrap();
        assert!(back.iter().map(|b| &b[..]).eq(blocks.iter().copied()));
    }
}
