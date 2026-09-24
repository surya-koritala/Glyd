//! zstd 1.5.7's block pre-splitter (`zstd_preSplit.c`): once a frame
//! has saved a few bytes, a full 128 KB block may be cut where its
//! content changes. The fast strategy compares the byte histograms of
//! the block's first and last 512 bytes and, when they differ enough,
//! cuts at 32, 64 or 96 KB by which border the middle resembles; the
//! double-fast strategy walks the block in 8 KB chunks with a sampled
//! two-byte fingerprint and cuts at the first chunk that deviates from
//! the accumulated past, the tolerance easing with each chunk kept.

const BLOCK: usize = 128 << 10;
const CHUNK: usize = 8 << 10;
const SEGMENT: usize = 512;
const THRESHOLD_PENALTY_RATE: u64 = 16;
const THRESHOLD_BASE: u64 = THRESHOLD_PENALTY_RATE - 2;
const THRESHOLD_PENALTY: u64 = 3;
const HASHLOG_MAX: u32 = 10;
const KNUTH: u32 = 0x9e3779b9;

/// `Fingerprint`: event counts over `1 << hash_log` buckets.
struct Fingerprint {
    events: [u32; 1 << HASHLOG_MAX],
    nb_events: u64,
}

impl Fingerprint {
    fn new() -> Fingerprint {
        Fingerprint { events: [0; 1 << HASHLOG_MAX], nb_events: 0 }
    }

    /// `hash2`: a byte at 8 bits, else two bytes multiplied down.
    fn hash(s: &[u8], at: usize, hash_log: u32) -> usize {
        if hash_log == 8 {
            return s[at] as usize;
        }
        ((u16::from_le_bytes([s[at], s[at + 1]]) as u32).wrapping_mul(KNUTH) >> (32 - hash_log)) as usize
    }

    /// `recordFingerprint_generic`: every `rate`-th position of `s`
    /// (the count of events noted as the integer quotient).
    fn record(&mut self, s: &[u8], rate: usize, hash_log: u32) {
        self.events = [0; 1 << HASHLOG_MAX];
        self.nb_events = 0;
        let limit = s.len() - 2 + 1;
        let mut n = 0;
        while n < limit {
            self.events[Self::hash(s, n, hash_log)] += 1;
            n += rate;
        }
        self.nb_events += (limit / rate) as u64;
    }

    /// `HIST_add` with `nbEvents` set to the segment's length.
    fn histogram(&mut self, s: &[u8]) {
        for &b in s {
            self.events[b as usize] += 1;
        }
        self.nb_events = s.len() as u64;
    }

    /// `fpDistance`.
    fn distance(&self, other: &Fingerprint, hash_log: u32) -> u64 {
        (0..1usize << hash_log).map(|n| (self.events[n] as i64 * other.nb_events as i64 - other.events[n] as i64 * self.nb_events as i64).unsigned_abs()).sum()
    }

    /// `compareFingerprints`: whether `new` deviates from `self` beyond
    /// the threshold eased by `penalty`.
    fn differs(&self, new: &Fingerprint, penalty: u64, hash_log: u32) -> bool {
        let p50 = self.nb_events * new.nb_events;
        let deviation = self.distance(new, hash_log);
        let threshold = p50 * (THRESHOLD_BASE + penalty) / THRESHOLD_PENALTY_RATE;
        deviation >= threshold
    }

    /// `mergeEvents`.
    fn merge(&mut self, other: &Fingerprint) {
        for (a, b) in self.events.iter_mut().zip(&other.events) {
            *a += b;
        }
        self.nb_events += other.nb_events;
    }
}

/// `ZSTD_splitBlock_fromBorders` (the fast strategy's level).
fn from_borders(s: &[u8]) -> usize {
    let (mut first, mut last, mut middle) = (Fingerprint::new(), Fingerprint::new(), Fingerprint::new());
    first.histogram(&s[..SEGMENT]);
    last.histogram(&s[BLOCK - SEGMENT..BLOCK]);
    if !first.differs(&last, 0, 8) {
        return BLOCK;
    }
    middle.histogram(&s[BLOCK / 2 - SEGMENT / 2..BLOCK / 2 + SEGMENT / 2]);
    let from_begin = first.distance(&middle, 8);
    let from_end = last.distance(&middle, 8);
    let min_distance = (SEGMENT * SEGMENT / 3) as u64;
    if (from_begin as i64 - from_end as i64).unsigned_abs() < min_distance {
        return 64 << 10;
    }
    if from_begin > from_end {
        32 << 10
    } else {
        96 << 10
    }
}

/// `ZSTD_splitBlock_byChunks` at `level` 0 to 3 (the double-fast
/// strategy uses 0: one position in 43, bucketed by first byte).
fn by_chunks(s: &[u8], level: usize) -> usize {
    const RATES: [usize; 4] = [43, 11, 5, 1];
    const HASH_LOGS: [u32; 4] = [8, 9, 10, 10];
    let (rate, hash_log) = (RATES[level], HASH_LOGS[level]);
    let mut past = Fingerprint::new();
    let mut new = Fingerprint::new();
    let mut penalty = THRESHOLD_PENALTY;
    past.record(&s[..CHUNK], rate, hash_log);
    let mut pos = CHUNK;
    while pos <= BLOCK - CHUNK {
        new.record(&s[pos..pos + CHUNK], rate, hash_log);
        if past.differs(&new, penalty, hash_log) {
            return pos;
        }
        past.merge(&new);
        penalty = penalty.saturating_sub(1);
        pos += CHUNK;
    }
    BLOCK
}

/// `ZSTD_splitBlock`: where the 128 KB block at the start of `s` is
/// cut, `level` being 0 for the fast strategy and 1 for double-fast
/// (`ZSTD_optimalBlockSize`'s table by strategy).
pub fn split_block(s: &[u8], level: usize) -> usize {
    debug_assert!(s.len() >= BLOCK);
    if level == 0 {
        from_borders(s)
    } else {
        by_chunks(s, level - 1)
    }
}
