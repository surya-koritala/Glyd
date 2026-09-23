//! zlib's `deflate_fast` (levels 1–3) and `deflate_slow` (4–9), run
//! over the plain text to predict the stream's tokens, and the
//! corrections where the stream differs.
//!
//! The emulation follows deflate.c: a rolling 15-bit hash of three
//! bytes, `head` and `prev` chains indexed by the low 15 bits of the
//! position, position 0 unmatchable (it is zlib's NIL), matches no
//! farther than `MAX_DIST`, `longest_match` with the level's chain,
//! good, nice and lazy limits, lazy evaluation one position ahead, the
//! TOO_FAR rule for 3-byte matches, and a block every 16,383 tokens.
//! Bytes past the end of the input compare as zero.

use super::{coder::{Bit, Decoder, Encoder}, trees, Block, Kind, Token};

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

/// Level 0 here means no matching at all (Z_HUFFMAN_ONLY, or a stream
/// of stored blocks); 1–3 `deflate_fast`, 4–9 `deflate_slow`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub level: u8,
    pub filtered: bool,
    /// Z_FIXED: every block under the fixed code
    pub fixed: bool,
}

impl Params {
    pub fn all() -> impl Iterator<Item = Params> {
        (0..=9u8).flat_map(|level| [false, true].into_iter().filter(move |f| !*f || level >= 4).map(move |filtered| Params { level, filtered, fixed: false }))
    }
}

/// zlib's matcher state over a plain text. Positions in the tables
/// are relative to `base`, 16 bits as in zlib's window, the tables
/// slid by 32 KB when the relative position reaches zlib's limit (the
/// entries that go to 0 were out of reach already); 0 is NIL.
pub struct Zlib<'a> {
    plain: &'a [u8],
    /// the plain text with MAX_MATCH + 8 zero bytes after it, so a
    /// comparison may run past the end without a check
    padded: Vec<u8>,
    p: Params,
    head: Vec<u16>,
    prev: Vec<u16>,
    base: u32,
    /// the rolling hash after the last insert
    ins_h: u32,
    /// every position below this has been inserted
    inserted: u32,
    /// the match found one position ahead by a lazy search: (length, start)
    carry: Option<(u32, u32)>,
}

const SLIDE_AT: u32 = W_SIZE + MAX_DIST;

impl<'a> Zlib<'a> {
    pub fn new(plain: &'a [u8], p: Params) -> Self {
        let mut padded = Vec::with_capacity(plain.len() + MAX_MATCH as usize + 8);
        padded.extend_from_slice(plain);
        padded.resize(plain.len() + MAX_MATCH as usize + 8, 0);
        Zlib { plain, padded, p, head: vec![0; 1 << 15], prev: vec![0; 1 << 15], base: 0, ins_h: 0, inserted: 0, carry: None }
    }

    // The reads below stay inside `padded`: every caller's index is at
    // most the plain length plus MAX_MATCH + 1, and the padding is
    // MAX_MATCH + 8.
    #[inline]
    fn at(&self, i: u32) -> u8 {
        debug_assert!((i as usize) < self.padded.len());
        unsafe { *self.padded.get_unchecked(i as usize) }
    }

    #[inline]
    fn pair(&self, i: u32) -> u16 {
        debug_assert!(i as usize + 2 <= self.padded.len());
        unsafe { self.padded.as_ptr().add(i as usize).cast::<u16>().read_unaligned() }
    }

    #[inline]
    fn word(&self, i: u32) -> u64 {
        debug_assert!(i as usize + 8 <= self.padded.len());
        unsafe { self.padded.as_ptr().add(i as usize).cast::<u64>().read_unaligned() }
    }

    #[inline]
    fn lookahead(&self, pos: u32) -> u32 {
        self.plain.len() as u32 - pos
    }

    /// zlib's slide_hash: the window moved by 32 KB.
    fn slide(&mut self) {
        for v in self.head.iter_mut().chain(self.prev.iter_mut()) {
            *v = if *v >= W_SIZE as u16 { *v - W_SIZE as u16 } else { 0 };
        }
        self.base += W_SIZE;
    }

    /// Positions up to `to` (exclusive) into the chains, as zlib inserts
    /// each `strstart` with at least MIN_MATCH bytes left.
    fn insert_to(&mut self, to: u32) {
        let end = to.min(self.plain.len() as u32 - MIN_MATCH + 1);
        if self.inserted >= end {
            self.inserted = self.inserted.max(to);
            return;
        }
        // zlib rolls its hash across consecutive inserts; after a run of
        // skipped positions it starts again from the bytes, which gives
        // the same value as rolling would have.
        let mut pos = self.inserted;
        let mut h = (((self.at(pos) as u32) << 5) ^ self.at(pos + 1) as u32) & HASH_MASK;
        let mut slide_at = self.base + SLIDE_AT;
        while pos < end {
            if pos >= slide_at {
                self.slide();
                slide_at = self.base + SLIDE_AT;
            }
            h = ((h << 5) ^ self.at(pos + 2) as u32) & HASH_MASK;
            let slot = (pos & HASH_MASK) as usize;
            unsafe {
                *self.prev.get_unchecked_mut(slot) = *self.head.get_unchecked(h as usize);
                *self.head.get_unchecked_mut(h as usize) = (pos - self.base) as u16;
            }
            pos += 1;
        }
        self.ins_h = h;
        self.inserted = to.max(end);
    }

    /// The chain's head for `pos` after its insertion: the previous
    /// position with the same hash, relative to `base`, or 0.
    fn hash_head(&mut self, pos: u32) -> u32 {
        self.insert_to(pos + 1);
        if self.lookahead(pos) < MIN_MATCH {
            return 0;
        }
        self.prev[(pos & HASH_MASK) as usize] as u32
    }

    /// deflate.c's longest_match: the longest match at `pos` longer than
    /// `prev_len`, starting from the relative position `cur` and
    /// following the chain; the length and the match's absolute start.
    fn longest_match(&self, pos: u32, mut cur: u32, prev_len: u32) -> (u32, u32) {
        let (good, _, nice, chain) = CONFIG[self.p.level as usize];
        let mut chain_length = chain;
        if prev_len >= good {
            chain_length >>= 2;
        }
        let lookahead = self.lookahead(pos);
        let nice = nice.min(lookahead);
        let rel = pos - self.base;
        let limit = if rel > MAX_DIST { rel - MAX_DIST } else { 0 };
        let mut best_len = prev_len;
        let mut best_start = 0u32;
        let scan_end = self.pair(pos + best_len - 1);
        let scan_start = self.pair(pos);
        let mut scan_end = scan_end;
        loop {
            let m = self.base + cur;
            if self.pair(m + best_len - 1) == scan_end && self.pair(m) == scan_start {
                let mut len = 2u32;
                while len + 8 <= MAX_MATCH {
                    let x = self.word(m + len) ^ self.word(pos + len);
                    if x != 0 {
                        len += x.trailing_zeros() / 8;
                        break;
                    }
                    len += 8;
                }
                if len + 8 > MAX_MATCH {
                    while len < MAX_MATCH && self.at(m + len) == self.at(pos + len) {
                        len += 1;
                    }
                }
                if len > best_len {
                    best_start = m;
                    best_len = len;
                    if len >= nice {
                        break;
                    }
                    scan_end = self.pair(pos + best_len - 1);
                }
            }
            cur = unsafe { *self.prev.get_unchecked((cur & HASH_MASK) as usize) } as u32;
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
        if head == 0 || prev_len >= lazy || (pos - self.base) - head > MAX_DIST {
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

    /// zlib's token at `pos`: none for level 0, deflate_fast's, or
    /// deflate_slow's from its lazy evaluation.
    pub fn predict(&mut self, pos: u32) -> Token {
        if self.p.level == 0 {
            return Token::Lit(self.at(pos));
        }
        if self.p.level <= 3 {
            let head = self.hash_head(pos);
            if head == 0 || (pos - self.base) - head > MAX_DIST {
                return Token::Lit(self.at(pos));
            }
            let (len, start) = self.longest_match(pos, head, MIN_MATCH - 1);
            return if len >= MIN_MATCH { Token::Ref { len: len as u16, dist: (pos - start) as u16 } } else { Token::Lit(self.at(pos)) };
        }
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
                let (_, lazy, _, _) = CONFIG[self.p.level as usize];
                if self.p.level <= 3 && (len > lazy || self.lookahead(pos + len) < MIN_MATCH) {
                    // deflate_fast skips the match's positions when it is
                    // longer than max_insert_length.
                    self.insert_to(pos + 1);
                    self.inserted = self.inserted.max(pos + len);
                } else {
                    // zlib inserts up to max_insert = strstart + lookahead - MIN_MATCH.
                    let max_insert = self.plain.len() as u32 - MIN_MATCH;
                    self.insert_to((pos + len).min(max_insert + 1));
                    self.inserted = self.inserted.max(pos + len);
                }
                pos + len
            }
        }
    }

    /// The candidates at `pos` in chain order, as distances, for a
    /// distance to be named as the n-th of them.
    fn chain(&mut self, pos: u32) -> impl Iterator<Item = u32> + '_ {
        let mut cur = self.hash_head(pos);
        let rel = pos - self.base;
        let limit = if rel > MAX_DIST { rel - MAX_DIST } else { 0 };
        let prev = &self.prev;
        std::iter::from_fn(move || {
            if cur == 0 || cur <= limit {
                return None;
            }
            let d = rel - cur;
            cur = prev[(cur & HASH_MASK) as usize] as u32;
            Some(d)
        })
        .take(4096)
    }
}

/// A stored run's tokens are the emulation's own, a literal where a
/// match would cross the run's end, so the state comes out as zlib's
/// did when it made and threw them away.
fn capped(guess: Token, pos: u32, end: u32, plain: &[u8]) -> Token {
    match guess {
        Token::Ref { len, .. } if pos + len as u32 > end => Token::Lit(plain[pos as usize]),
        t => t,
    }
}

/// How many of the first `limit` tokens `p` gets wrong.
fn mismatches(plain: &[u8], p: Params, blocks: &[Block], limit: usize) -> usize {
    let mut z = Zlib::new(plain, p);
    let (mut pos, mut seen, mut wrong) = (0u32, 0usize, 0usize);
    for b in blocks {
        match &b.kind {
            Kind::Stored(bytes) => {
                let end = pos + bytes.len() as u32;
                while pos < end {
                    let t = capped(z.predict(pos), pos, end, plain);
                    pos = z.commit(pos, t);
                }
            }
            Kind::Fixed(tokens) | Kind::Dynamic(_, tokens) => {
                for &actual in tokens {
                    if seen == limit {
                        return wrong;
                    }
                    wrong += (z.predict(pos) != actual) as usize;
                    seen += 1;
                    pos = z.commit(pos, actual);
                }
            }
        }
    }
    wrong
}

/// The parameters that predict the stream best, judged on its first
/// 4,096 tokens; the lower level on a tie. Z_FIXED when every coded
/// block is fixed.
pub fn detect(plain: &[u8], blocks: &[Block]) -> Params {
    let fixed = blocks.iter().all(|b| !matches!(b.kind, Kind::Dynamic(..))) && blocks.iter().any(|b| matches!(b.kind, Kind::Fixed(_)));
    let mut best = (usize::MAX, Params { level: 6, filtered: false, fixed });
    for p in Params::all() {
        let p = Params { fixed, ..p };
        let wrong = mismatches(plain, p, blocks, 4096);
        if wrong < best.0 {
            best = (wrong, p);
        }
        if wrong == 0 {
            break;
        }
    }
    best.1
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
    last: Bit,
    stored: Bit,
    stored_len: [Bit; 64],
    block_same: Bit,
    block_count: [Bit; 64],
    kind_same: Bit,
    header_same: Bit,
    header_len: [Bit; 64],
    byte: Vec<Bit>,
}

impl Models {
    fn new() -> Self {
        Models {
            same: [Bit::default(); 2],
            is_ref: [Bit::default(); 2],
            len_same: Bit::default(),
            len: vec![Bit::default(); 512],
            dist_same: Bit::default(),
            hop_found: Bit::default(),
            hop: [Bit::default(); 64],
            dist: vec![Bit::default(); 1 << 16],
            last: Bit::default(),
            stored: Bit::default(),
            stored_len: [Bit::default(); 64],
            block_same: Bit::default(),
            block_count: [Bit::default(); 64],
            kind_same: Bit::default(),
            header_same: Bit::default(),
            header_len: [Bit::default(); 64],
            byte: vec![Bit::default(); 256],
        }
    }
}

fn kind(t: &Token) -> usize {
    matches!(t, Token::Ref { .. }) as usize
}

fn span(tokens: &[Token]) -> u32 {
    tokens.iter().map(|t| match t { Token::Lit(_) => 1, Token::Ref { len, .. } => if *len == 259 { 258 } else { *len as u32 } }).sum()
}

/// One token coded against the guess; `pos` moved past it.
fn code_token(e: &mut Encoder, m: &mut Models, z: &mut Zlib, pos: u32, actual: Token) -> u32 {
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
    z.commit(pos, actual)
}

fn decode_token(d: &mut Decoder, m: &mut Models, z: &mut Zlib, pos: u32, plain: &[u8]) -> Option<(Token, u32)> {
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
    Some((actual, z.commit(pos, actual)))
}

/// The blocks against the emulation: the corrections, from which
/// `recreate` gets the blocks back with the plain text alone. Per
/// block: whether it is the last (predicted: it reaches the end),
/// whether it is stored (predicted: no; then its length), the token
/// count (predicted: 16,383 or to the end), the tokens, then whether
/// its kind and, for a dynamic block, its header are what zlib's trees
/// give for those tokens (else the header as it was).
pub fn predict(plain: &[u8], p: Params, blocks: &[Block]) -> Vec<u8> {
    let mut z = Zlib::new(plain, p);
    let mut m = Models::new();
    let mut e = Encoder::new();
    let mut pos = 0u32;
    let total = plain.len() as u32;
    for b in blocks {
        let (stored_len, tokens): (Option<u32>, &[Token]) = match &b.kind {
            Kind::Stored(bytes) => (Some(bytes.len() as u32), &[]),
            Kind::Fixed(t) | Kind::Dynamic(_, t) => (None, t),
        };
        let end = pos + stored_len.unwrap_or_else(|| span(tokens));
        e.bit(&mut m.last, (b.last == (end == total)) as u32);
        e.bit(&mut m.stored, stored_len.is_some() as u32);
        if let Some(n) = stored_len {
            e.count(&mut m.stored_len, n);
            // The run's tokens are the emulation's own on both sides:
            // nothing to code.
            while pos < end {
                let t = capped(z.predict(pos), pos, end, plain);
                pos = z.commit(pos, t);
            }
            continue;
        }
        let n = tokens.len() as u32;
        let as_predicted = n == BLOCK_TOKENS || (n < BLOCK_TOKENS && end == total);
        e.bit(&mut m.block_same, as_predicted as u32);
        if !as_predicted {
            e.count(&mut m.block_count, n);
        }
        for &actual in tokens {
            pos = code_token(&mut e, &mut m, &mut z, pos, actual);
        }
        let t = trees::build(tokens);
        let predicted_kind = trees::kind_of(&t, (end - (pos - span(tokens))) as usize, p.fixed);
        let actual_kind = if matches!(b.kind, Kind::Fixed(_)) { 1 } else { 2 };
        e.bit(&mut m.kind_same, (predicted_kind == actual_kind) as u32);
        if predicted_kind != actual_kind {
            e.bit(&mut m.kind_same, (actual_kind == 2) as u32);
        }
        if let Kind::Dynamic(h, _) = &b.kind {
            let same = *h == t.header;
            e.bit(&mut m.header_same, same as u32);
            if !same {
                let (bytes, bits) = super::header_bits(h);
                e.count(&mut m.header_len, bytes.len() as u32);
                e.tree(&mut m.byte, 3, bits);
                for &x in &bytes {
                    e.tree(&mut m.byte, 8, x as u32);
                }
            }
        }
    }
    e.finish()
}

/// The blocks back: the emulation with the corrections applied.
pub fn recreate(plain: &[u8], p: Params, corrections: &[u8], n_blocks: usize) -> Option<Vec<Block>> {
    let mut z = Zlib::new(plain, p);
    let mut m = Models::new();
    let mut d = Decoder::new(corrections);
    let mut pos = 0u32;
    let total = plain.len() as u32;
    let mut out = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        let last_as_predicted = d.bit(&mut m.last) == 1;
        let start = pos;
        if d.bit(&mut m.stored) == 1 {
            let n = d.count(&mut m.stored_len);
            let end = start.checked_add(n)?;
            if end > total {
                return None;
            }
            while pos < end {
                let t = capped(z.predict(pos), pos, end, plain);
                pos = z.commit(pos, t);
            }
            let last = last_as_predicted == (end == total);
            out.push(Block { last, bit_start: 0, kind: Kind::Stored(plain[start as usize..end as usize].to_vec()) });
            continue;
        }
        let as_predicted = d.bit(&mut m.block_same) == 1;
        let n = if as_predicted { BLOCK_TOKENS } else { d.count(&mut m.block_count) };
        let mut tokens = Vec::with_capacity(n.min(BLOCK_TOKENS) as usize);
        for _ in 0..n {
            if as_predicted && pos == total {
                break;
            }
            let (t, next) = decode_token(&mut d, &mut m, &mut z, pos, plain)?;
            tokens.push(t);
            pos = next;
        }
        let t = trees::build(&tokens);
        let predicted_kind = trees::kind_of(&t, (pos - start) as usize, p.fixed);
        let kind = if d.bit(&mut m.kind_same) == 1 { predicted_kind } else if d.bit(&mut m.kind_same) == 1 { 2 } else { 1 };
        let last = last_as_predicted == (pos == total);
        let kind = if kind == 1 {
            Kind::Fixed(tokens)
        } else {
            let header = if d.bit(&mut m.header_same) == 1 {
                t.header
            } else {
                let len = d.count(&mut m.header_len) as usize;
                let _bits = d.tree(&mut m.byte, 3);
                let bytes: Vec<u8> = (0..len).map(|_| d.tree(&mut m.byte, 8) as u8).collect();
                super::read_header(&mut super::BitReader::new(&bytes))?
            };
            Kind::Dynamic(header, tokens)
        };
        out.push(Block { last, bit_start: 0, kind });
    }
    (pos == total).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::super::{parse, tests::{text, zlib_deflate}};
    use super::*;

    /// zlib's own streams predicted with few corrections, and every
    /// block back from them; a stream from another level (1, emulated
    /// as 6) still comes back, with more.
    #[test]
    fn zlib_levels_predicted_and_recreated() {
        let plain = text(1 << 20);
        for (level, filtered) in [(6, false), (1, false), (2, false), (3, false), (4, false), (5, false), (7, false), (8, false), (9, false), (6, true)] {
            let stream = zlib_deflate(&plain, level, if filtered { "Z_FILTERED" } else { "Z_DEFAULT_STRATEGY" });
            let s = parse(&stream).unwrap();
            let p = Params { level: level as u8, filtered, fixed: false };
            // Levels 8 and 9 make the same stream on this text; the lower
            // wins the tie.
            let detected = detect(&plain, &s.blocks);
            assert!(detected == p || (level == 9 && detected.level == 8), "level {level} filtered {filtered}: detected {detected:?}");
            let c = predict(&plain, p, &s.blocks);
            let n_tokens: usize = s.blocks.iter().map(|b| match &b.kind { Kind::Fixed(t) | Kind::Dynamic(_, t) => t.len(), _ => 0 }).sum();
            #[cfg(feature = "deflate")]
            {
                let (r, _) = preflate_rs::preflate_whole_deflate_stream(&stream, &preflate_rs::PreflateConfig::default()).unwrap();
                eprintln!("level {level} filtered {filtered}: stream {} B, {} tokens in {} blocks; corrections ours {} B, preflate {} B", stream.len(), n_tokens, s.blocks.len(), c.len(), r.corrections.len());
            }
            assert!(c.len() * 400 < stream.len(), "level {level} filtered {filtered}: {} bytes of corrections for {} tokens, {} bytes of stream", c.len(), n_tokens, stream.len());
            let back = recreate(&plain, p, &c, s.blocks.len()).unwrap();
            assert!(back.iter().zip(&s.blocks).all(|(a, b)| a.last == b.last && a.kind == b.kind), "level {level}: blocks back");
        }
        let stream = zlib_deflate(&plain, 1, "Z_DEFAULT_STRATEGY");
        let s = parse(&stream).unwrap();
        let p = Params { level: 6, filtered: false, fixed: false };
        let c = predict(&plain, p, &s.blocks);
        assert!(recreate(&plain, p, &c, s.blocks.len()).unwrap().iter().zip(&s.blocks).all(|(a, b)| a.kind == b.kind));
    }
}
