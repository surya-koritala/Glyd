//! Context mixing for the cold tier (`--cold`): every bit is predicted
//! from what followed the same contexts before, the predictions mixed
//! by a small online-trained network, and coded arithmetically at the
//! mixed probability. No parse: there is nothing a byte-level matcher
//! would find that the match model here does not, and the model sees
//! the free text (hashes, identifiers, prose) that matching leaves as
//! literals. Around 1 MB/s per core each way, symmetric; the cost is
//! paid where bytes sit for years and are read rarely.
//!
//! The model (an `lpaq`-style arrangement): seven predictors — the
//! orders 1, 2, 3, 4 and 6 of the byte history, the current word, and
//! a match model that follows the longest earlier occurrence of the last
//! seven bytes — each a bit history (paq's states) in a hash table
//! addressed a nibble at a time (one cache line per predictor per
//! nibble) and mapped to a learned probability; a mixer whose weight
//! set is chosen by the bits of the byte so far; two SSE stages. A unit is coded from an empty model: units
//! decode in parallel.

/// A counter holds a 22-bit probability over a 10-bit count.
const COUNT_LIMIT: u32 = 1023;
/// Log2 buckets of each predictor's table (16 bytes each).
const TABLE_BITS: u32 = 21;
const MATCH_MIN: usize = 7;
const MATCH_MAX: usize = 65535;
/// Context predictors (`Predictor::end_byte` sets their hashes).
const NCTX: usize = 11;
/// Mixer inputs: the context predictors, the match model, a bias.
const N_INPUTS: usize = NCTX + 2;
const MIX_RATE: i32 = 7;

/// The bit-history states (paq's): a state stands for the counts of
/// zeros and ones seen in a context and, while the counts are small,
/// which came last; `NEX[s][b]` is the state after bit `b`, `NEX[s][2]`
/// and `[3]` its counts. A count is discounted when the other bit is
/// seen, so a history that changes its mind is believed quickly.
static NEX: [[u8; 4]; 256] = [
    [1, 2, 0, 0], [3, 5, 1, 0], [4, 6, 0, 1], [7, 10, 2, 0],
    [8, 12, 1, 1], [9, 13, 1, 1], [11, 14, 0, 2], [15, 19, 3, 0],
    [16, 23, 2, 1], [17, 24, 2, 1], [18, 25, 2, 1], [20, 27, 1, 2],
    [21, 28, 1, 2], [22, 29, 1, 2], [26, 30, 0, 3], [31, 33, 4, 0],
    [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1],
    [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2],
    [34, 37, 2, 2], [34, 37, 2, 2], [36, 39, 1, 3], [36, 39, 1, 3],
    [36, 39, 1, 3], [36, 39, 1, 3], [38, 40, 0, 4], [41, 43, 5, 0],
    [42, 45, 4, 1], [42, 45, 4, 1], [44, 47, 3, 2], [44, 47, 3, 2],
    [46, 49, 2, 3], [46, 49, 2, 3], [48, 51, 1, 4], [48, 51, 1, 4],
    [50, 52, 0, 5], [53, 43, 6, 0], [54, 57, 5, 1], [54, 57, 5, 1],
    [56, 59, 4, 2], [56, 59, 4, 2], [58, 61, 3, 3], [58, 61, 3, 3],
    [60, 63, 2, 4], [60, 63, 2, 4], [62, 65, 1, 5], [62, 65, 1, 5],
    [50, 66, 0, 6], [67, 55, 7, 0], [68, 57, 6, 1], [68, 57, 6, 1],
    [70, 73, 5, 2], [70, 73, 5, 2], [72, 75, 4, 3], [72, 75, 4, 3],
    [74, 77, 3, 4], [74, 77, 3, 4], [76, 79, 2, 5], [76, 79, 2, 5],
    [62, 81, 1, 6], [62, 81, 1, 6], [64, 82, 0, 7], [83, 69, 8, 0],
    [84, 71, 7, 1], [84, 71, 7, 1], [86, 73, 6, 2], [86, 73, 6, 2],
    [44, 59, 5, 3], [44, 59, 5, 3], [58, 61, 4, 4], [58, 61, 4, 4],
    [60, 49, 3, 5], [60, 49, 3, 5], [76, 89, 2, 6], [76, 89, 2, 6],
    [78, 91, 1, 7], [78, 91, 1, 7], [80, 92, 0, 8], [93, 69, 9, 0],
    [94, 87, 8, 1], [94, 87, 8, 1], [96, 45, 7, 2], [96, 45, 7, 2],
    [48, 99, 2, 7], [48, 99, 2, 7], [88, 101, 1, 8], [88, 101, 1, 8],
    [80, 102, 0, 9], [103, 69, 10, 0], [104, 87, 9, 1], [104, 87, 9, 1],
    [106, 57, 8, 2], [106, 57, 8, 2], [62, 109, 2, 8], [62, 109, 2, 8],
    [88, 111, 1, 9], [88, 111, 1, 9], [80, 112, 0, 10], [113, 85, 11, 0],
    [114, 87, 10, 1], [114, 87, 10, 1], [116, 57, 9, 2], [116, 57, 9, 2],
    [62, 119, 2, 9], [62, 119, 2, 9], [88, 121, 1, 10], [88, 121, 1, 10],
    [90, 122, 0, 11], [123, 85, 12, 0], [124, 97, 11, 1], [124, 97, 11, 1],
    [126, 57, 10, 2], [126, 57, 10, 2], [62, 129, 2, 10], [62, 129, 2, 10],
    [98, 131, 1, 11], [98, 131, 1, 11], [90, 132, 0, 12], [133, 85, 13, 0],
    [134, 97, 12, 1], [134, 97, 12, 1], [136, 57, 11, 2], [136, 57, 11, 2],
    [62, 139, 2, 11], [62, 139, 2, 11], [98, 141, 1, 12], [98, 141, 1, 12],
    [90, 142, 0, 13], [143, 95, 14, 0], [144, 97, 13, 1], [144, 97, 13, 1],
    [68, 57, 12, 2], [68, 57, 12, 2], [62, 81, 2, 12], [62, 81, 2, 12],
    [98, 147, 1, 13], [98, 147, 1, 13], [100, 148, 0, 14], [149, 95, 15, 0],
    [150, 107, 14, 1], [150, 107, 14, 1], [108, 151, 1, 14], [108, 151, 1, 14],
    [100, 152, 0, 15], [153, 95, 16, 0], [154, 107, 15, 1], [108, 155, 1, 15],
    [100, 156, 0, 16], [157, 95, 17, 0], [158, 107, 16, 1], [108, 159, 1, 16],
    [100, 160, 0, 17], [161, 105, 18, 0], [162, 107, 17, 1], [108, 163, 1, 17],
    [110, 164, 0, 18], [165, 105, 19, 0], [166, 117, 18, 1], [118, 167, 1, 18],
    [110, 168, 0, 19], [169, 105, 20, 0], [170, 117, 19, 1], [118, 171, 1, 19],
    [110, 172, 0, 20], [173, 105, 21, 0], [174, 117, 20, 1], [118, 175, 1, 20],
    [110, 176, 0, 21], [177, 105, 22, 0], [178, 117, 21, 1], [118, 179, 1, 21],
    [110, 180, 0, 22], [181, 115, 23, 0], [182, 117, 22, 1], [118, 183, 1, 22],
    [120, 184, 0, 23], [185, 115, 24, 0], [186, 127, 23, 1], [128, 187, 1, 23],
    [120, 188, 0, 24], [189, 115, 25, 0], [190, 127, 24, 1], [128, 191, 1, 24],
    [120, 192, 0, 25], [193, 115, 26, 0], [194, 127, 25, 1], [128, 195, 1, 25],
    [120, 196, 0, 26], [197, 115, 27, 0], [198, 127, 26, 1], [128, 199, 1, 26],
    [120, 200, 0, 27], [201, 115, 28, 0], [202, 127, 27, 1], [128, 203, 1, 27],
    [120, 204, 0, 28], [205, 125, 29, 0], [206, 127, 28, 1], [128, 207, 1, 28],
    [130, 208, 0, 29], [209, 125, 30, 0], [210, 137, 29, 1], [138, 211, 1, 29],
    [130, 212, 0, 30], [213, 125, 31, 0], [214, 137, 30, 1], [138, 215, 1, 30],
    [130, 216, 0, 31], [217, 125, 32, 0], [218, 137, 31, 1], [138, 219, 1, 31],
    [130, 220, 0, 32], [221, 125, 33, 0], [222, 137, 32, 1], [138, 223, 1, 32],
    [130, 224, 0, 33], [225, 125, 34, 0], [226, 137, 33, 1], [138, 227, 1, 33],
    [130, 228, 0, 34], [229, 125, 35, 0], [230, 137, 34, 1], [138, 231, 1, 34],
    [130, 232, 0, 35], [233, 125, 36, 0], [234, 137, 35, 1], [138, 235, 1, 35],
    [130, 236, 0, 36], [237, 125, 37, 0], [238, 137, 36, 1], [138, 239, 1, 36],
    [130, 240, 0, 37], [241, 135, 38, 0], [242, 137, 37, 1], [138, 243, 1, 37],
    [140, 244, 0, 38], [245, 135, 39, 0], [246, 69, 38, 1], [80, 247, 1, 38],
    [140, 248, 0, 39], [249, 135, 40, 0], [250, 69, 39, 1], [80, 251, 1, 39],
    [140, 252, 0, 40], [249, 135, 41, 0], [250, 69, 40, 1], [80, 251, 1, 40],
    [140, 252, 0, 41], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
];

/// `squash(d) = 4096 / (1 + e^(-d/256))`, d in -2047..=2047.
fn squash(d: i32) -> i32 {
    static T: [i32; 33] = [1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4024, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094];
    if d > 2047 {
        return 4095;
    }
    if d < -2047 {
        return 0;
    }
    let w = d & 127;
    let i = ((d >> 7) + 16) as usize;
    (T[i] * (128 - w) + T[i + 1] * w + 64) >> 7
}

struct Tables {
    /// The inverse of `squash`: probability (12 bits) to logit.
    stretch: [i16; 4096],
    /// Adaptation rates by count: 16384 / (n + n + 3).
    dt: [i32; 1024],
}

impl Tables {
    fn new() -> Box<Tables> {
        let mut t = Box::new(Tables { stretch: [0; 4096], dt: [0; 1024] });
        let mut pi = 0usize;
        for x in -2047..=2047 {
            let v = squash(x) as usize;
            for i in pi..=v {
                t.stretch[i] = x as i16;
            }
            pi = v + 1;
        }
        for i in pi..4096 {
            t.stretch[i] = 2047;
        }
        for (i, d) in t.dt.iter_mut().enumerate() {
            *d = 16384 / (2 * i as i32 + 3);
        }
        t
    }
}

/// An adaptive probability: 22 bits of probability that the next bit
/// is 1 over a 10-bit count, the rate 1 / (count + 1.5) until the count
/// reaches its limit.
#[inline(always)]
fn learn(slot: &mut u32, bit: u32, limit: u32, dt: &[i32; 1024]) {
    let n = *slot & 1023;
    let p = (*slot >> 10) as i32;
    if n < limit {
        *slot += 1;
    } else {
        *slot = (*slot & 0xffff_fc00) | limit;
    }
    let delta = (((bit << 22) as i32 - p) >> 3) * dt[n as usize];
    *slot = slot.wrapping_add(delta as u32 & 0xffff_fc00);
}

/// A state map: the probability of a 1 after each bit-history state,
/// learned; it starts from the state's own counts.
fn state_map() -> Vec<u32> {
    (0..256)
        .map(|s| {
            let (n0, n1) = (NEX[s][2] as u32, NEX[s][3] as u32);
            (((n1 * 2 + 1) << 22) / (n0 * 2 + n1 * 2 + 2)) << 10
        })
        .collect()
}

/// A predictor's table: buckets of 16 bytes, a checksum and the bit
/// histories of the 15 nodes of a nibble, found by the context hash
/// among three buckets of one cache line; when none matches, the
/// least-used of the three (by its first node's state) is replaced.
struct Histories {
    t: Vec<u8>,
    mask: usize,
}

impl Histories {
    fn new(bits: u32) -> Histories {
        Histories { t: vec![0u8; 16 << bits], mask: (1 << bits) - 1 }
    }

    #[inline(always)]
    fn find(&mut self, h: u32) -> usize {
        let chk = (h >> 24) as u8;
        let i = (h as usize & self.mask) * 16;
        let (b0, b1, b2) = (i, i ^ 16, i ^ 32);
        if self.t[b0] == chk {
            return b0;
        }
        if self.t[b1] == chk {
            return b1;
        }
        if self.t[b2] == chk {
            return b2;
        }
        let mut b = b0;
        if self.t[b1 + 1] < self.t[b + 1] {
            b = b1;
        }
        if self.t[b2 + 1] < self.t[b + 1] {
            b = b2;
        }
        self.t[b..b + 16].fill(0);
        self.t[b] = chk;
        b
    }
}

/// An SSE stage: the mixed probability refined by a context, through a
/// table of 33 probabilities per context interpolated on the logit.
struct Apm {
    t: Vec<u16>,
    index: usize,
}

impl Apm {
    fn new(n: usize) -> Apm {
        let mut t = vec![0u16; n * 33];
        for i in 0..n {
            for j in 0..33 {
                t[i * 33 + j] = (squash((j as i32 - 16) * 128) * 16) as u16;
            }
        }
        Apm { t, index: 0 }
    }

    /// The refined probability of `pr` (12 bits) in context `cx`.
    #[inline(always)]
    fn pp(&mut self, pr: i32, cx: usize, tables: &Tables) -> i32 {
        let s = tables.stretch[pr as usize] as i32 + 2048;
        let w = s & 127;
        self.index = (s >> 7) as usize + cx * 33;
        ((self.t[self.index] as i32 * (128 - w) + self.t[self.index + 1] as i32 * w) >> 11) as i32
    }

    #[inline(always)]
    fn update(&mut self, bit: u32, rate: u32) {
        let g = ((bit << 16) + (bit << rate)) as i32 - bit as i32 - bit as i32;
        let t = &mut self.t[self.index];
        *t = (*t as i32 + ((g - *t as i32) >> rate)) as u16;
        let t = &mut self.t[self.index + 1];
        *t = (*t as i32 + ((g - *t as i32) >> rate)) as u16;
    }
}

struct Predictor {
    tables: Box<Tables>,
    /// The context predictors' histories and state maps.
    t: Vec<Histories>,
    sm: Vec<Vec<u32>>,
    /// Each predictor's node for the current bit, and its state.
    slot: [usize; NCTX],
    state: [u8; NCTX],
    /// Each predictor's bucket for the current nibble.
    bucket: [usize; NCTX],
    /// Context hashes for the current byte.
    h: [u32; NCTX],
    /// The byte's bits so far with a leading 1; the last 4 and 8 bytes.
    c0: u32,
    c4: u32,
    c8: u32,
    bitcount: u32,
    /// The current word's hash.
    word: u32,
    /// Where the current and the previous line start in `buf`.
    line_start: usize,
    prev_line_start: usize,
    /// The previous word's hash; the last quoted name before a colon
    /// (a JSON key) and the string being read.
    prev_word: u32,
    key: u32,
    quoted: u32,
    in_quote: bool,
    /// Match model: the history, a hash of the last MATCH_MIN bytes to
    /// the position after them, the position being followed, the
    /// length matched, the byte it predicts, and its counters.
    buf: Vec<u8>,
    ht: Vec<u32>,
    match_ptr: usize,
    match_len: usize,
    expected: u32,
    mm: Vec<u32>,
    mm_slot: usize,
    /// Mixers: inputs, weights (a set per c0 and match length; a set
    /// per previous byte), the chosen sets, their outputs and the mix.
    x: [i32; N_INPUTS],
    w: Vec<i32>,
    w2: Vec<i32>,
    wset: usize,
    wset2: usize,
    pr1: i32,
    pr2: i32,
    pr_mix: i32,
    a1: Apm,
    a2: Apm,
    pr: i32,
}

impl Predictor {
    fn new(len: usize) -> Predictor {
        let bits = (usize::BITS - len.max(1).leading_zeros()).clamp(16, TABLE_BITS);
        let tables = Tables::new();
        let mut p = Predictor {
            tables,
            t: (0..NCTX).map(|_| Histories::new(bits)).collect(),
            sm: (0..NCTX).map(|_| state_map()).collect(),
            slot: [0; NCTX],
            state: [0; NCTX],
            bucket: [0; NCTX],
            h: [0; NCTX],
            c0: 1,
            c4: 0,
            c8: 0,
            bitcount: 0,
            word: 0,
            line_start: 0,
            prev_line_start: 0,
            prev_word: 0,
            key: 0,
            quoted: 0,
            in_quote: false,
            buf: Vec::with_capacity(len + 8),
            ht: vec![0u32; 1 << bits],
            match_ptr: 0,
            match_len: 0,
            expected: 0,
            mm: vec![1 << 31; 64 * 2],
            mm_slot: 0,
            x: [0; N_INPUTS],
            w: vec![0; 4 * 256 * N_INPUTS],
            w2: vec![0; 256 * N_INPUTS],
            wset: 0,
            wset2: 0,
            pr1: 2048,
            pr2: 2048,
            pr_mix: 2048,
            a1: Apm::new(256),
            a2: Apm::new(1 << 16),
            pr: 2048,
        };
        p.begin_nibble();
        p.predict();
        p
    }

    #[inline(always)]
    fn hash(a: u32, b: u32) -> u32 {
        let h = a.wrapping_mul(0x2F0B_3E7D) ^ b.wrapping_mul(0x9E37_79B1);
        h ^ (h >> 15)
    }

    /// Each predictor's bucket of 16 slots for the nibble starting now.
    #[inline(always)]
    fn begin_nibble(&mut self) {
        let nib = if self.bitcount < 4 { 0 } else { self.c0 };
        for i in 0..NCTX {
            let hh = Self::hash(self.h[i], nib.wrapping_add(i as u32 * 0x1000));
            self.bucket[i] = self.t[i].find(hh);
        }
    }

    #[inline(always)]
    fn predict(&mut self) {
        let within = (self.c0 & ((1 << (self.bitcount & 3)) - 1)) | (1 << (self.bitcount & 3));
        let st = &self.tables.stretch;
        for i in 0..NCTX {
            self.slot[i] = self.bucket[i] + within as usize;
            let s = self.t[i].t[self.slot[i]];
            self.state[i] = s;
            self.x[i] = st[(self.sm[i][s as usize] >> 20) as usize] as i32;
        }
        // The match model: the expected byte's next bit, as long as the
        // bits so far agree with it, weighted by the length matched.
        if self.match_len > 0 && (self.expected + 256) >> (8 - self.bitcount) == self.c0 {
            let bit = (self.expected >> (7 - self.bitcount)) & 1;
            let l = self.match_len.min(31);
            self.mm_slot = l * 2 + bit as usize;
            let p = (self.mm[self.mm_slot] >> 20) as i32;
            self.x[NCTX] = st[p as usize] as i32;
        } else {
            self.match_len = 0;
            self.mm_slot = usize::MAX;
            self.x[NCTX] = 0;
        }
        self.x[NCTX + 1] = 256;
        let ml = if self.match_len == 0 { 0 } else if self.match_len < 16 { 1 } else if self.match_len < 32 { 2 } else { 3 };
        self.wset = (ml * 256 + self.c0 as usize) * N_INPUTS;
        self.wset2 = (self.c4 & 0xff) as usize * N_INPUTS;
        let w = &self.w[self.wset..self.wset + N_INPUTS];
        let w2 = &self.w2[self.wset2..self.wset2 + N_INPUTS];
        let (mut dot, mut dot2) = (0i64, 0i64);
        for i in 0..N_INPUTS {
            dot += self.x[i] as i64 * w[i] as i64;
            dot2 += self.x[i] as i64 * w2[i] as i64;
        }
        let (d1, d2) = ((dot >> 16) as i32, (dot2 >> 16) as i32);
        self.pr1 = squash(d1);
        self.pr2 = squash(d2);
        self.pr_mix = squash((d1 + d2) >> 1);
        let p1 = self.a1.pp(self.pr_mix, self.c0 as usize, &self.tables);
        let cx = (self.c0 ^ Self::hash(self.c4 & 0xff_ffff, 0x51)) as usize & 0xffff;
        let p2 = self.a2.pp(self.pr_mix, cx, &self.tables);
        self.pr = (p1 + 3 * p2 + 2) >> 2;
    }

    /// The probability that the next bit is 1, 12 bits.
    #[inline(always)]
    fn p(&self) -> i32 {
        self.pr
    }

    #[inline(always)]
    fn update(&mut self, bit: u32) {
        let dt = &self.tables.dt;
        for i in 0..NCTX {
            let s = self.state[i];
            learn(&mut self.sm[i][s as usize], bit, COUNT_LIMIT, dt);
            self.t[i].t[self.slot[i]] = NEX[s as usize][bit as usize];
        }
        if self.mm_slot != usize::MAX {
            learn(&mut self.mm[self.mm_slot], bit, COUNT_LIMIT, dt);
        }
        let err = ((bit << 12) as i32 - self.pr1) * MIX_RATE;
        let w = &mut self.w[self.wset..self.wset + N_INPUTS];
        for i in 0..N_INPUTS {
            w[i] += (self.x[i] * err + 0x8000) >> 16;
        }
        let err = ((bit << 12) as i32 - self.pr2) * MIX_RATE;
        let w = &mut self.w2[self.wset2..self.wset2 + N_INPUTS];
        for i in 0..N_INPUTS {
            w[i] += (self.x[i] * err + 0x8000) >> 16;
        }
        self.a1.update(bit, 7);
        self.a2.update(bit, 7);

        self.c0 = (self.c0 << 1) | bit;
        self.bitcount += 1;
        if self.bitcount == 8 {
            self.end_byte((self.c0 & 255) as u8);
            self.c0 = 1;
            self.bitcount = 0;
            self.begin_nibble();
        } else if self.bitcount == 4 {
            self.begin_nibble();
        }
        self.predict();
    }

    fn end_byte(&mut self, c: u8) {
        self.buf.push(c);
        self.c8 = (self.c8 << 8) | (self.c4 >> 24);
        self.c4 = (self.c4 << 8) | c as u32;
        let lower = c.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            self.word = Self::hash(self.word, lower as u32 + 1);
        } else if self.word != 0 {
            self.prev_word = self.word;
            self.word = 0;
        }
        // JSON: the key a value belongs to.
        if c == b'"' {
            if self.in_quote {
                self.in_quote = false;
            } else {
                self.in_quote = true;
                self.quoted = 0;
            }
        } else if self.in_quote {
            self.quoted = Self::hash(self.quoted, c as u32 + 1);
        } else if c == b':' {
            self.key = self.quoted;
        }
        self.h[0] = self.c4 & 0xff;
        self.h[1] = self.c4 & 0xffff;
        self.h[2] = self.c4 & 0xff_ffff;
        self.h[3] = Self::hash(self.c4, 4);
        self.h[4] = Self::hash(self.c4, self.c8 & 0xffff);
        self.h[5] = Self::hash(self.word, self.c4 & 0xff);
        self.h[6] = Self::hash(Self::hash(self.c4, self.c8), 8);
        // The column: the position in the line and the byte above it
        // in the line before, for what repeats line after line.
        let pos = self.buf.len();
        if c == b'\n' {
            self.prev_line_start = self.line_start;
            self.line_start = pos;
        }
        let col = pos - self.line_start;
        let above = if self.prev_line_start + col < self.line_start { self.buf[self.prev_line_start + col] as u32 } else { 256 };
        self.h[7] = Self::hash(Self::hash(col.min(255) as u32, above), self.c4 & 0xff);
        self.h[8] = Self::hash(Self::hash(self.word, self.prev_word), 0x88);
        self.h[9] = Self::hash(Self::hash(self.key, self.c4 & 0xffff), 0x99);

        // The match model follows its match while the bytes agree, else
        // looks the last MATCH_MIN bytes up.
        let pos = self.buf.len();
        if self.match_len > 0 && self.match_ptr < pos && self.buf[self.match_ptr] == c {
            self.match_len = (self.match_len + 1).min(MATCH_MAX);
            self.match_ptr += 1;
        } else {
            self.match_len = 0;
        }
        if pos >= MATCH_MIN {
            // The hash of the last MATCH_MIN bytes, whole.
            let mut h = 0u32;
            for &b in &self.buf[pos - MATCH_MIN..pos] {
                h = h.wrapping_mul(0x2F0B_3E7D).wrapping_add(b as u32 + 1);
            }
            let hi = (h ^ (h >> 16)) as usize & (self.ht.len() - 1);
            if self.match_len == 0 {
                let cand = self.ht[hi] as usize;
                if cand > 0 && cand < pos {
                    let mut l = 0usize;
                    while l < MATCH_MAX && l < cand && self.buf[cand - 1 - l] == self.buf[pos - 1 - l] {
                        l += 1;
                    }
                    if l >= MATCH_MIN {
                        self.match_len = l;
                        self.match_ptr = cand;
                    }
                }
            }
            self.ht[hi] = pos as u32;
        }
        self.expected = if self.match_len > 0 { self.buf[self.match_ptr] as u32 } else { 0 };
        self.h[10] = Self::hash(Self::hash(self.expected, self.match_len.min(15) as u32), (self.c4 & 0xff) + 0xaa00);
    }
}

/// Binary arithmetic coder, carry-free: the range [x1, x2] narrows to
/// the part of it the bit's probability gives, and settled leading
/// bytes are shifted out.
struct Encoder<'a> {
    x1: u32,
    x2: u32,
    out: &'a mut Vec<u8>,
}

impl<'a> Encoder<'a> {
    #[inline(always)]
    fn encode(&mut self, bit: u32, p: i32) {
        let xmid = self.x1 + ((self.x2 - self.x1) >> 12) * p as u32;
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
    }

    fn flush(&mut self) {
        self.out.push((self.x1 >> 24) as u8);
        self.out.push((self.x1 >> 16) as u8);
        self.out.push((self.x1 >> 8) as u8);
        self.out.push(self.x1 as u8);
    }
}

struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    src: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(src: &'a [u8]) -> Decoder<'a> {
        let mut d = Decoder { x1: 0, x2: u32::MAX, x: 0, src, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next() as u32;
        }
        d
    }

    #[inline(always)]
    fn next(&mut self) -> u8 {
        let b = self.src.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    #[inline(always)]
    fn decode(&mut self, p: i32) -> u32 {
        let xmid = self.x1 + ((self.x2 - self.x1) >> 12) * p as u32;
        let bit = if self.x <= xmid {
            self.x2 = xmid;
            1
        } else {
            self.x1 = xmid + 1;
            0
        };
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.next() as u32;
        }
        bit
    }
}

/// `input` coded into `out`, from an empty model.
pub fn encode(input: &[u8], out: &mut Vec<u8>) {
    let mut pr = Predictor::new(input.len());
    let mut enc = Encoder { x1: 0, x2: u32::MAX, out };
    for &c in input {
        for i in (0..8).rev() {
            let bit = (c as u32 >> i) & 1;
            enc.encode(bit, pr.p());
            pr.update(bit);
        }
    }
    enc.flush();
}

/// `src` decoded into `out` (its length is the input's), from an empty
/// model. Bytes past the end of `src` read as zero, so a truncated
/// stream decodes to something of the right length: the caller checks.
pub fn decode(src: &[u8], out: &mut [u8]) {
    let mut pr = Predictor::new(out.len());
    let mut dec = Decoder::new(src);
    for o in out.iter_mut() {
        let mut c = 0u32;
        for _ in 0..8 {
            let bit = dec.decode(pr.p());
            pr.update(bit);
            c = (c << 1) | bit;
        }
        *o = c as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let mut x = 7u64;
        let mut text = Vec::new();
        for i in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let w = ["the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog\n", "{\"id\": 12345, ", "\"sha\": \"deadbeef\"}\n"][(x % 10) as usize];
            text.extend_from_slice(w.as_bytes());
            if i % 1000 == 0 {
                text.push((x >> 8) as u8);
            }
        }
        for input in [&text[..], &[][..], b"a", b"abcabcabcabcabcabcabc", &text[..1000]] {
            let mut c = Vec::new();
            encode(input, &mut c);
            let mut back = vec![0u8; input.len()];
            decode(&c, &mut back);
            assert!(back == input, "round trip of {} bytes", input.len());
        }
        let mut c = Vec::new();
        encode(&text, &mut c);
        assert!(c.len() * 4 < text.len(), "{} of {}", c.len(), text.len());
    }
}
