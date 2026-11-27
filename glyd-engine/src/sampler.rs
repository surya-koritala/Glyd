//! Sampling a next token from a step's logits: repetition, presence and frequency penalties over the tokens so far, a bias, allowed tokens, a grammar's mask, temperature, top-k, top-p and
//! min-p, then a draw from the generator the seed starts. Temperature 0 is the largest logit (the first of equals, as torch.argmax chooses).
//!
//! A draw weighs each token by `weight`, exp((logit - largest) / temperature) as a whole number of 2^-40 (WBITS), so that every sum a draw takes (the row's, a cut's, the walk's) is a
//! sum of whole numbers: the same in any order, on the host or on a GPU. Where the logits are made on a GPU, the row is not read: the GPU makes the row's largest logits (and, where
//! the request changes the row, of the row the draw sees: the changes put on it first as `processed` puts them, `Proc`, `Resolved`), and where a cut or the draw's point lies past
//! them, the weights and the tokens of ranges of the row in the draw's order (`View`, `select`); the host draws from those the token it draws from the whole row, seed for seed.

/// What a request asks of the sampler.
#[derive(Clone, Debug)]
pub struct Params {
    pub temperature: f32,
    /// The most probable tokens kept (0: all).
    pub top_k: usize,
    /// The smallest set of tokens whose probabilities add to this (1.0: all).
    pub top_p: f32,
    /// Tokens kept whose probability is at least this share of the largest's (0: all).
    pub min_p: f32,
    /// Divides a positive logit, multiplies a negative one, of every token in the prompt or the answer so far (1.0: none).
    pub repetition_penalty: f32,
    /// Subtracted once from the logit of a token that has been generated, and `frequency_penalty` times the count of it.
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// Added to the logit of each token it names (the token and its bias; a token named twice, its last, as the server's JSON object keeps its last key), before the penalties and the
    /// temperature, as vLLM orders them: it can move a greedy choice, and a token the model's largest logits do not hold can become the choice, so a GPU's candidates of a request with a
    /// bias are of the row with it put on (`Proc`).
    pub logit_bias: Vec<(u32, f32)>,
    /// The only tokens that may be chosen (vLLM's `allowed_token_ids`; empty: any): the row's other logits are taken out before anything else.
    pub allowed: Vec<u32>,
    /// The end tokens (and the request's stop tokens) may not be chosen until this many tokens have been (vLLM's `min_tokens`).
    pub min_tokens: usize,
    /// A grammar the answer must follow (a JSON schema, a regular expression ...): the tokens it does not allow are taken out of each row, as `allowed` does (constrain.rs).
    pub constraint: Option<std::sync::Arc<crate::constrain::Constraint>>,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: Vec::new(),
            allowed: Vec::new(),
            min_tokens: 0,
            constraint: None,
            seed: 0,
        }
    }
}

impl Params {
    /// Whether a step needs no more than the largest logit.
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
            && self.repetition_penalty == 1.0
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && self.logit_bias.is_empty()
            && self.allowed.is_empty()
            && self.min_tokens == 0
            && self.constraint.is_none()
    }
}

/// A token whose logit a request's draw changes (`Proc`): a bias added; the repetition penalty (the token is in the prompt or the answer so far); the presence and frequency penalties
/// (the times the answer has said it, 0: none); and `rescue`, an end token set to 0 first where a grammar and the least length leave no token (`Sampler::processed`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub token: u32,
    pub bias: Option<f32>,
    pub seen: bool,
    pub said: u32,
    pub rescue: bool,
}

/// What a GPU puts on a row of logits for a request's draw before it makes the row's largest logits (`Sampler::proc`, made before the step): the tokens taken out (an allowed list, the
/// grammar's mask of the step, the tokens a least length holds back) and the tokens whose logits change (`Entry`). `resolve` makes it the row's, the grammar's mask waited for.
#[derive(Clone)]
pub struct Proc {
    /// The grammar's mask of the step (a worker makes it beside the step), its end tokens and its width.
    gram: Option<(crate::constrain::MaskHandle, Vec<u32>, usize)>,
    /// The allowed list, where the request has one.
    allowed: Option<std::sync::Arc<Vec<u32>>>,
    /// The tokens held back until the least length (none where it is reached).
    suppress: Vec<u32>,
    entries: Vec<Entry>,
    repetition: f32,
    presence: f32,
    frequency: f32,
}

/// A `Proc` for a row of a vocabulary: the tokens that may be chosen as 32 tokens' bits a word (token t's in word t / 32, bit t % 32; None: every token), and the tokens whose logits
/// change, each once (those in the vocabulary, the grammar's end tokens put back at 0 where nothing else is left), with the penalties.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub words: Option<Vec<u32>>,
    pub entries: Vec<Entry>,
    pub repetition: f32,
    pub presence: f32,
    pub frequency: f32,
}

impl Proc {
    /// The request's changes for a row of `v` logits, as `Sampler::processed` makes them: the allowed list, the grammar's mask (waited for; where its worker died, the end tokens alone,
    /// as the grammar then gives) and the tokens held back taken out together; where that leaves no token under a grammar, its end tokens put back.
    pub fn resolve(&self, v: usize) -> Resolved {
        let n = v.div_ceil(32);
        let mut words: Option<Vec<u32>> = None;
        let mut and = |w: Vec<u32>| match words.as_mut() {
            Some(x) => x.iter_mut().zip(&w).for_each(|(a, b)| *a &= b),
            None => words = Some(w),
        };
        if let Some(a) = &self.allowed {
            let mut w = vec![0u32; n];
            for &t in a.iter().filter(|&&t| (t as usize) < v) {
                w[t as usize / 32] |= 1 << (t % 32);
            }
            and(w);
        }
        if let Some((h, ends, width)) = &self.gram {
            // (the bits `Gram::apply` keeps: a word all set as it is; past the mask's length, none)
            let (bits, len) = match h.words() {
                Some(m) => (m.as_slice().to_vec(), m.len()),
                None => {
                    let mut b = vec![0u32; width.div_ceil(32)];
                    for &e in ends.iter().filter(|&&e| (e as usize) < *width) {
                        b[e as usize / 32] |= 1 << (e % 32);
                    }
                    (b, *width)
                }
            };
            let w = (0..n)
                .map(|i| match bits.get(i).copied().unwrap_or(0) {
                    u32::MAX => u32::MAX,
                    x if x == 0 || i * 32 >= len => 0,
                    x if len - i * 32 < 32 => x & ((1u32 << (len - i * 32)) - 1),
                    x => x,
                })
                .collect();
            and(w);
        }
        if !self.suppress.is_empty() {
            let mut w = vec![u32::MAX; n];
            for &t in self.suppress.iter().filter(|&&t| (t as usize) < v) {
                w[t as usize / 32] &= !(1 << (t % 32));
            }
            and(w);
        }
        let mut entries: Vec<Entry> = self.entries.iter().filter(|e| (e.token as usize) < v).copied().collect();
        // (a grammar and a least length that leave no token: the grammar's end tokens at 0, the reply ends as it cannot go on)
        if let (Some((_, ends, _)), Some(w), false) = (&self.gram, &words, self.suppress.is_empty()) {
            if !(0..v).any(|t| (w[t / 32] >> (t % 32)) & 1 == 1) {
                for &e in ends.iter().filter(|&&e| (e as usize) < v) {
                    match entries.iter_mut().find(|x| x.token == e) {
                        Some(x) => x.rescue = true,
                        None => entries.push(Entry { token: e, bias: None, seen: false, said: 0, rescue: true }),
                    }
                }
            }
        }
        Resolved { words, entries, repetition: self.repetition, presence: self.presence, frequency: self.frequency }
    }
}

impl Resolved {
    /// Whether token `t` may be chosen.
    pub fn allows(&self, t: usize) -> bool {
        self.words.as_ref().is_none_or(|w| (w[t / 32] >> (t % 32)) & 1 == 1)
    }

    /// A changed token's logit from the model's `x`, in `Sampler::processed`'s order, each operation rounded alone: the bias, -inf where the token is taken out, 0 where it is put
    /// back, the repetition penalty, then the presence and frequency penalties.
    pub fn changed(&self, e: &Entry, mut x: f32) -> f32 {
        if let Some(b) = e.bias {
            x += b;
        }
        if !self.allows(e.token as usize) {
            x = f32::NEG_INFINITY;
        }
        if e.rescue {
            x = 0.0;
        }
        if e.seen {
            x = if x > 0.0 { x / self.repetition } else { x * self.repetition };
        }
        if e.said > 0 {
            x -= self.presence + self.frequency * e.said as f32;
        }
        x
    }

    /// The row the draw sees, from the model's row `raw` as the GPU makes it (kernels/engine.cu `proc_fill`, `proc_entries`): each token's logit where it may be chosen, else -inf;
    /// then each changed token's (`changed`). `Sampler::processed`'s row, bit for bit.
    pub fn apply(&self, raw: &[f32]) -> Vec<f32> {
        let mut l: Vec<f32> = raw.iter().enumerate().map(|(t, &x)| if self.allows(t) { x } else { f32::NEG_INFINITY }).collect();
        for e in &self.entries {
            l[e.token as usize] = self.changed(e, raw[e.token as usize]);
        }
        l
    }
}

/// A token's weight in a draw is a whole number of 2^-WBITS: the largest logit's is 2^WBITS (`weight`). A row of up to 2^23 tokens sums to less than 2^64.
pub const WBITS: u32 = 40;

/// `weight`'s float constants, as bits (kernels/engine.cu `draw_weight` has the same): log2(e); ln 2 in two parts (the first's few bits make k ln 2's first part exact); exp's
/// Taylor coefficients 1/2! .. 1/7!.
const LOG2E: u32 = 0x3FB8_AA3B;
const LN2_HI: u32 = 0x3F31_8000;
const LN2_LO: u32 = 0xB95E_8083;
const EXP_C: [u32; 6] = [0x3F00_0000, 0x3E2A_AAAB, 0x3D2A_AAAB, 0x3C08_8889, 0x3AB6_0B61, 0x3950_0D01];

/// A token's weight in a draw: exp(q) for q = (logit - largest) / temperature (in f32, as `cut` forms it), in units of 2^-WBITS, rounded down; 0 for q below -32 (exp(-32) is less
/// than 2^-46) or not a number (a row of -inf). Every float operation rounded alone, in this order (kernels/engine.cu `draw_weight` makes it bit for bit): k = q log2(e) to the
/// nearest whole number (ties to even), r = (q - k LN2_HI) - k LN2_LO, exp(r) by Taylor's terms to the 7th in Horner's order, its 24 bits shifted by k. Within 4e-7 of exp(q) and
/// 2^-40; not always falling with q at its last bit (2,861 of the 1.1 billion floats from -32 to 0 weigh a unit of their 24 bits more than the next larger), so no cut is a floor on
/// weights: min-p is one on the quotient q itself (`min_floor`), which falls along the draw's order.
pub fn weight(q: f32) -> u64 {
    if !(q > -32.0) {
        return 0;
    }
    if q >= 0.0 {
        return 1 << WBITS;
    }
    let f = f32::from_bits;
    let k = (q * f(LOG2E)).round_ties_even();
    let r = (q - k * f(LN2_HI)) - k * f(LN2_LO);
    let mut p = f(EXP_C[5]);
    for &c in EXP_C[..5].iter().rev() {
        p = p * r + f(c);
    }
    p = p * r + 1.0;
    p = p * r + 1.0;
    let b = p.to_bits();
    let m = ((b & 0x7F_FFFF) | 0x80_0000) as u64;
    let s = ((b >> 23) & 0xFF) as i32 - 127 - 23 + k as i32 + WBITS as i32;
    match s {
        0.. => m << s,
        -63..=-1 => m >> -s,
        _ => 0,
    }
}

/// The least whole number at or above p w, exactly (p a float, w a weight): the weight a cumulative sum must reach for a top-p of p over a list of weight w. 0 for p of 0 or below
/// (or not a number); u64::MAX where it is past that.
pub fn share(p: f32, w: u64) -> u64 {
    if !(p > 0.0) || w == 0 {
        return 0;
    }
    let b = p.to_bits();
    let e = ((b >> 23) & 0xFF) as i32;
    // (p = m 2^-s)
    let (m, s) = if e == 0 { ((b & 0x7F_FFFF) as u128, 149) } else { (((b & 0x7F_FFFF) | 0x80_0000) as u128, 150 - e) };
    let x = m * w as u128;
    let n = match s {
        ..=0 => x.checked_shl((-s) as u32).filter(|y| y >> (-s) as u32 == x).unwrap_or(u128::MAX),
        1..=127 => (x + (1u128 << s) - 1) >> s,
        _ => (x > 0) as u128,
    };
    n.min(u64::MAX as u128) as u64
}

/// A min-p's floor on a token's quotient q = (logit - largest) / temperature: ln(min_p) as a float (exp(q) >= min_p, the largest's exp being 1). The quotient falls along the draw's
/// order (a subtraction and a division by a positive number, each rounded, never turn two values about), so the tokens at or above it are a first part of it.
pub fn min_floor(min_p: f32) -> f32 {
    (min_p as f64).ln() as f32
}

/// The point a draw's number `u` (the generator's next) lands on in a list of weight `total`: floor(u' total / 2^53), u' the number's top 53 bits (u' / 2^53 a uniform number in
/// [0, 1)): a whole number below `total` (0 where it is 0).
fn pick(u: u64, total: u64) -> u64 {
    (((u >> 11) as u128 * total as u128) >> 53) as u64
}

/// A float's bits as the number that grows with the value (-0 and +0 the same), as kernels/engine.cu's `f32_key` orders them.
pub fn key32(x: f32) -> u32 {
    let b = x.to_bits();
    let b = if b == 0x8000_0000 { 0 } else { b };
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// A token's place in a draw's order as a number that falls along it: the larger logit first, and of equal ones the smaller token (`dist`'s order). kernels/engine.cu `draw_key`.
pub fn key(x: f32, token: u32) -> u64 {
    ((key32(x) as u64) << 32) | (u32::MAX - token) as u64
}

/// Draws made from rows read on the host and from what a GPU made of them (src/sched.rs `draw_row`, `check_row`).
pub static DRAWS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What a GPU is asked of a row (the row a draw sees) beside its candidates (`View`): the tokens whose logit is not -inf, whose keys (`key`) lie in [lo, hi] and whose quotients
/// (x - max) / temperature are `floor` or more (a min-p's: `min_floor`; -inf: any), in 256 bins by the bits `shift` .. `shift` + 8 of their keys; their weights about `max`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Query {
    pub max: f32,
    pub temperature: f32,
    pub lo: u64,
    pub hi: u64,
    pub floor: f32,
    pub shift: u32,
}

impl Query {
    /// Every token of the row (not -inf), binned by its keys' first 8 bits.
    pub fn all(max: f32, temperature: f32) -> Query {
        Query { max, temperature, lo: 0, hi: u64::MAX, floor: f32::NEG_INFINITY, shift: 56 }
    }

    /// Token `t` of logit `x`'s key and weight, where it is the query's.
    pub fn holds(&self, x: f32, t: u32) -> Option<(u64, u64)> {
        if !(x > f32::NEG_INFINITY) {
            return None;
        }
        let (k, q) = (key(x, t), (x - self.max) / self.temperature);
        (k >= self.lo && k <= self.hi && q >= self.floor).then(|| (k, weight(q)))
    }
}

/// A query's tokens in its 256 bins: each bin's count, weight (the sum of its tokens'), and least and greatest key (u64::MAX and 0 where it has none).
#[derive(Clone, Debug, PartialEq)]
pub struct Bins {
    pub count: Vec<u32>,
    pub weight: Vec<u64>,
    pub least: Vec<u64>,
    pub most: Vec<u64>,
}

impl Bins {
    pub fn empty() -> Bins {
        Bins { count: vec![0; 256], weight: vec![0; 256], least: vec![u64::MAX; 256], most: vec![0; 256] }
    }

    /// A token of the query (its key and weight) put in its bin.
    pub fn add(&mut self, k: u64, w: u64, shift: u32) {
        let b = ((k >> shift) & 0xFF) as usize;
        self.count[b] += 1;
        self.weight[b] += w;
        self.least[b] = self.least[b].min(k);
        self.most[b] = self.most[b].max(k);
    }
}

/// What a GPU gives of a step's row (the row a request's draw sees: the model's, or the one its changes are put on) in place of the row: its candidates, and the weights and
/// tokens of ranges of it (`Query`). None from any of them: it has none (the row is then read).
pub trait View {
    /// The row's `kern::TOPK` largest logits (token and logit, largest first, equal ones by the smaller token).
    fn cands(&mut self) -> Option<Vec<(u32, f32)>>;
    /// The sum of every token's weight about `max` at `temperature` (made with the step where it was asked for).
    fn total(&mut self, max: f32, temperature: f32) -> Option<u64> {
        self.bins(&Query::all(max, temperature)).map(|b| b.weight.iter().sum())
    }
    fn bins(&mut self, q: &Query) -> Option<Bins>;
    /// The query's tokens (token and logit, in any order), where they are `cap` or fewer.
    fn tokens(&mut self, q: &Query, cap: usize) -> Option<Vec<(u32, f32)>>;
    /// Token `t`'s logit (a drafted token that is not among the candidates).
    fn logit(&mut self, t: u32) -> Option<f32>;
}

/// The most tokens a draw asks a view for at once (`select`): a range of more is split by its keys' bits again.
pub const SELECT_CAP: usize = 4096;

/// What `select` finds: the first place of a query's tokens in the draw's order where they hold so many tokens (`Count`) or so much weight (`Weight`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    Count(u64),
    Weight(u64),
}

/// The token `select` finds: its key, logit and weight, and the count and weight of the query's tokens before it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct At {
    pub key: u64,
    pub token: u32,
    pub x: f32,
    pub w: u64,
    pub before_n: u64,
    pub before_w: u64,
}

/// The token of `q`'s at which its tokens, in the draw's order (the larger key first), first hold `target`, found range by range (the bin that holds it, then that bin's range
/// binned by the bits its keys differ in, until a range of SELECT_CAP tokens or fewer, whose tokens are read and ordered); `skip`: a token of `q`'s (its key and weight) left out.
/// Some(None) where all of them hold less; None where the view gives none, or gives tokens that are not its bins'.
pub fn select(v: &mut dyn View, q: &Query, target: Target, skip: Option<(u64, u64)>) -> Option<Option<At>> {
    let reached = |n: u64, w: u64| match target {
        Target::Count(k) => n >= k,
        Target::Weight(x) => w >= x,
    };
    let mut q = Query { shift: 56, ..*q };
    let (mut n, mut w) = (0u64, 0u64);
    loop {
        let mut b = v.bins(&q)?;
        if let Some((k, kw)) = skip.filter(|s| s.0 >= q.lo && s.0 <= q.hi) {
            let i = ((k >> q.shift) & 0xFF) as usize;
            b.count[i] = b.count[i].checked_sub(1)?;
            b.weight[i] = b.weight[i].checked_sub(kw)?;
        }
        let mut found = None;
        for i in (0..256).rev() {
            let (cn, cw) = (n + b.count[i] as u64, w + b.weight[i]);
            if b.count[i] > 0 && reached(cn, cw) {
                found = Some(i);
                break;
            }
            (n, w) = (cn, cw);
        }
        let Some(i) = found else { return Some(None) };
        let (least, most) = (b.least[i], b.most[i]);
        if b.count[i] as usize <= SELECT_CAP || least == most {
            let toks = v.tokens(&Query { lo: least, hi: most, ..q }, SELECT_CAP + 1)?;
            let mut ks: Vec<(u64, u32, f32, u64)> = toks
                .iter()
                .filter_map(|&(t, x)| q.holds(x, t).map(|(k, wt)| (k, t, x, wt)))
                .filter(|&(k, ..)| k >= least && k <= most && Some(k) != skip.map(|s| s.0))
                .collect();
            if ks.len() != b.count[i] as usize {
                return None;
            }
            ks.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            for (k, t, x, wt) in ks {
                let (cn, cw) = (n + 1, w + wt);
                if reached(cn, cw) {
                    return Some(Some(At { key: k, token: t, x, w: wt, before_n: n, before_w: w }));
                }
                (n, w) = (cn, cw);
            }
            return None;
        }
        let h = 63 - (least ^ most).leading_zeros();
        q = Query { lo: least, hi: most, shift: h.saturating_sub(7), ..q };
    }
}

/// The model's own sampling defaults, from its generation_config.json (temperature, top_k, top_p, repetition_penalty), as vLLM takes them; the API's (temperature 1) where it has none.
pub fn generation_defaults(dir: &std::path::Path) -> Params {
    let mut p = Params { temperature: 1.0, ..Params::default() };
    let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json")) else { return p };
    let Ok(v) = glyd_gpu::json::parse(&text) else { return p };
    let num = |k: &str| match v.get(k) {
        Some(glyd_gpu::json::Value::Number(n)) => n.parse::<f64>().ok(),
        _ => None,
    };
    if v.get("do_sample").and_then(glyd_gpu::json::Value::as_bool) == Some(false) {
        return p;
    }
    if let Some(x) = num("temperature") {
        p.temperature = x as f32;
    }
    if let Some(x) = num("top_k") {
        p.top_k = x.max(0.0) as usize;
    }
    if let Some(x) = num("top_p") {
        p.top_p = x as f32;
    }
    if let Some(x) = num("repetition_penalty") {
        p.repetition_penalty = x as f32;
    }
    p
}

/// A seed for a request that gives none (so that a draw is not the same twice, as vLLM's are not): the clock, the process and a counter, mixed.
pub fn fresh_seed() -> u64 {
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let mut z = t ^ ((std::process::id() as u64) << 32) ^ COUNT.fetch_add(0x9E37_79B9_7F4A_7C15, std::sync::atomic::Ordering::Relaxed);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ from a splitmix64 seed: a generator of its own, the same on every platform.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut next = || {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        };
        Rng([next(), next(), next(), next()])
    }

    fn u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }
}

pub struct Sampler {
    p: Params,
    rng: Rng,
    /// How many times each token is in the prompt or the answer (the repetition penalty's) and in the answer (the others').
    seen: std::collections::HashMap<u32, u32>,
    said: std::collections::HashMap<u32, u32>,
    /// Tokens chosen so far, and the tokens that may not be chosen until `min_tokens` of them (set by the scheduler: the end tokens and the request's stop tokens).
    n_said: usize,
    suppress: Vec<u32>,
    /// Whether each token of the row is allowed (made at the first row, from `allowed`).
    mask: Vec<bool>,
    /// Where the answer is in the request's grammar, if it has one.
    gram: Option<crate::constrain::Gram>,
    /// The allowed list for a GPU (`Proc`), ascending, each once.
    allowed: Option<std::sync::Arc<Vec<u32>>>,
    /// The tokens whose logits the penalties and the bias change (a token the bias names twice: its last bias), kept as the answer grows (`note`), and where each is.
    entries: Vec<Entry>,
    entry_at: std::collections::HashMap<u32, usize>,
}

/// The list a draw from a view walks (`Sampler::kept`): all of it among the candidates (its tokens and weights, as `cut` gives them), or the candidates and the tokens of a query
/// past them (the candidates are its first) and its weight.
enum Kept {
    List(Vec<u32>, Vec<u64>),
    Past(Query, u64),
}

/// The token of the list `idx` (weights `w`, in the draw's order) that point `r` lands on: the first whose cumulative weight passes it; the first where none does (a list of weight 0).
fn land(idx: &[u32], w: &[u64], r: u64) -> u32 {
    let mut acc = 0u64;
    for (i, &x) in w.iter().enumerate() {
        acc += x;
        if acc > r {
            return idx[i];
        }
    }
    idx[0]
}

impl Sampler {
    /// A sampler for a request whose prompt is `prompt`.
    pub fn new(p: Params, prompt: &[u32]) -> Sampler {
        let mut seen = std::collections::HashMap::new();
        if p.repetition_penalty != 1.0 {
            for &t in prompt {
                *seen.entry(t).or_insert(0) += 1;
            }
        }
        let rng = Rng::new(p.seed);
        let gram = p.constraint.as_ref().map(|c| c.start());
        let allowed = (!p.allowed.is_empty()).then(|| {
            let mut a = p.allowed.clone();
            a.sort_unstable();
            a.dedup();
            std::sync::Arc::new(a)
        });
        let mut s = Sampler {
            p,
            rng,
            seen,
            said: std::collections::HashMap::new(),
            n_said: 0,
            suppress: Vec::new(),
            mask: Vec::new(),
            gram,
            allowed,
            entries: Vec::new(),
            entry_at: std::collections::HashMap::new(),
        };
        for &(t, b) in &s.p.logit_bias.clone() {
            s.entry(t).bias = Some(b);
        }
        for t in s.seen.keys().copied().collect::<Vec<_>>() {
            s.entry(t).seen = true;
        }
        s
    }

    /// Token `t`'s entry, made where it has none.
    fn entry(&mut self, t: u32) -> &mut Entry {
        let at = *self.entry_at.entry(t).or_insert_with(|| {
            self.entries.push(Entry { token: t, bias: None, seen: false, said: 0, rescue: false });
            self.entries.len() - 1
        });
        &mut self.entries[at]
    }

    pub fn params(&self) -> &Params {
        &self.p
    }

    /// The generator moved on `draws` draws: where it is in a request whose answer holds that many sampled tokens (each took one draw, `walk`'s), so that
    /// a caller that knows only the seed and the step draws the step's token as `sample` would have (range.rs). The penalties' counts are not replayed.
    pub fn skip(&mut self, draws: u64) {
        for _ in 0..draws {
            self.rng.u64();
        }
    }

    /// The tokens that may not be chosen until `min_tokens` tokens have been.
    pub fn set_suppress(&mut self, tokens: Vec<u32>) {
        self.suppress = tokens;
    }

    fn suppressing(&self) -> bool {
        self.p.min_tokens > self.n_said && !self.suppress.is_empty()
    }

    /// The next token from `logits` (one per token of the vocabulary), which the answer then holds.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let mut l = self.processed(logits);
        let tok = if self.p.temperature <= 0.0 { argmax(&l) } else { self.draw(&mut l) };
        self.note(tok);
        tok
    }

    /// A drafted token checked against `logits` (speculative sampling with a draft that proposes one token): kept with the probability the draw gives it, else a token drawn from the
    /// draw's distribution without it. The token is distributed exactly as `sample`'s (P(draft) = p(draft); P(x) = (1 - p(draft)) p(x) / (1 - p(draft)) = p(x) for every other x), though
    /// the generator is used differently, so a seed gives other tokens than steps would. Greedy: the largest logit, kept where it is the draft. Returns the token and whether it is the draft.
    pub fn check_draft(&mut self, logits: &[f32], draft: u32) -> (u32, bool) {
        let mut l = self.processed(logits);
        let tok = if self.p.temperature <= 0.0 {
            argmax(&l)
        } else {
            let (idx, w) = self.dist(&mut l);
            self.check_list(&idx, &w, draft)
        };
        self.note(tok);
        (tok, tok == draft)
    }

    /// `check_draft`'s draw from the list `idx` (weights `w`): the draft where the first number lands within its weight (none drawn where it has none), else the rest's walk.
    fn check_list(&mut self, idx: &[u32], w: &[u64], draft: u32) -> u32 {
        let total: u64 = w.iter().sum();
        let wd = idx.iter().position(|&t| t == draft).map_or(0, |i| w[i]);
        if wd > 0 && pick(self.rng.u64(), total) < wd {
            return draft;
        }
        // (the rest, its weight the list's but the draft's)
        let (idx, w): (Vec<u32>, Vec<u64>) = idx.iter().zip(w).filter(|(t, _)| **t != draft).map(|(t, x)| (*t, *x)).unzip();
        if idx.is_empty() {
            draft
        } else {
            self.walk(&idx, &w)
        }
    }

    /// The token chosen, held by the answer (the penalties' counts).
    fn note(&mut self, tok: u32) {
        if let Some(g) = &mut self.gram {
            g.consume(tok);
        }
        self.n_said += 1;
        let n = self.said.entry(tok).or_insert(0);
        *n += 1;
        let n = *n;
        let (rep, said) = (self.p.repetition_penalty != 1.0, self.p.presence_penalty != 0.0 || self.p.frequency_penalty != 0.0);
        if rep {
            *self.seen.entry(tok).or_insert(0) += 1;
        }
        if rep || said {
            let e = self.entry(tok);
            e.seen |= rep;
            if said {
                e.said = n;
            }
        }
    }

    /// `logits` after the bias, the allowed tokens, the suppressed ones and the penalties: what the temperature and the cuts then see (what a GPU makes of the row, `Proc`).
    pub fn processed(&mut self, logits: &[f32]) -> Vec<f32> {
        let mut l: Vec<f32> = logits.to_vec();
        for e in &self.entries {
            if let (Some(b), Some(x)) = (e.bias, l.get_mut(e.token as usize)) {
                *x += b;
            }
        }
        if !self.p.allowed.is_empty() {
            if self.mask.len() != l.len() {
                self.mask = vec![false; l.len()];
                for &t in &self.p.allowed {
                    if let Some(m) = self.mask.get_mut(t as usize) {
                        *m = true;
                    }
                }
            }
            for (x, &ok) in l.iter_mut().zip(&self.mask) {
                if !ok {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(g) = &mut self.gram {
            g.apply(&mut l);
        }
        if self.suppressing() {
            for &t in &self.suppress {
                if let Some(x) = l.get_mut(t as usize) {
                    *x = f32::NEG_INFINITY;
                }
            }
            // (a grammar that is whole and allows the end tokens alone, before the least length: the reply ends, as it cannot go on)
            if let Some(g) = self.gram.as_ref().filter(|_| !l.iter().any(|&x| x > f32::NEG_INFINITY)) {
                for &e in g.ends() {
                    if let Some(x) = l.get_mut(e as usize) {
                        *x = 0.0;
                    }
                }
            }
        }
        if self.p.repetition_penalty != 1.0 {
            for &t in self.seen.keys() {
                if let Some(x) = l.get_mut(t as usize) {
                    *x = if *x > 0.0 { *x / self.p.repetition_penalty } else { *x * self.p.repetition_penalty };
                }
            }
        }
        if self.p.presence_penalty != 0.0 || self.p.frequency_penalty != 0.0 {
            for (&t, &n) in self.said.iter() {
                if let Some(x) = l.get_mut(t as usize) {
                    *x -= self.p.presence_penalty + self.p.frequency_penalty * n as f32;
                }
            }
        }
        l
    }

    /// Whether the request changes the row before its draw: penalties, a bias, allowed tokens, a grammar, or tokens held back until the least length. The candidates a draw takes
    /// are then of the row it sees (`Proc` put on it), not the model's.
    pub fn changes(&self) -> bool {
        let p = &self.p;
        p.repetition_penalty != 1.0
            || p.presence_penalty != 0.0
            || p.frequency_penalty != 0.0
            || !p.logit_bias.is_empty()
            || !p.allowed.is_empty()
            || self.gram.is_some()
            || self.suppressing()
    }

    /// What a GPU puts on the row before the next draw's candidates (`Proc`): None where the request changes nothing on it (`changes`). A grammar's mask is the one its worker
    /// makes beside the step (made here where no worker has it).
    pub fn proc(&mut self) -> Option<Proc> {
        if !self.changes() {
            return None;
        }
        let gram = self.gram.as_mut().map(|g| {
            let h = g.handle().unwrap_or_else(|| crate::constrain::MaskHandle::Ready(std::sync::Arc::new(g.mask())));
            (h, g.ends().to_vec(), g.width())
        });
        let p = &self.p;
        Some(Proc {
            gram,
            allowed: self.allowed.clone(),
            suppress: if self.suppressing() { self.suppress.clone() } else { Vec::new() },
            entries: self.entries.clone(),
            repetition: p.repetition_penalty,
            presence: p.presence_penalty,
            frequency: p.frequency_penalty,
        })
    }

    /// The temperature at which the next draw takes the total of the row's weights (`View::total`; a step can make it with its candidates): a draw with no top-k, but a min-p alone
    /// (whose walk is of the tokens it keeps). None for the largest logit or a top-k.
    pub fn total_need(&self) -> Option<f32> {
        let p = &self.p;
        (p.temperature > 0.0 && p.top_k == 0 && (p.top_p < 1.0 || p.min_p <= 0.0)).then_some(p.temperature)
    }

    /// What the next draw asks of a GPU before the step: the total's temperature (`total_need`), and what it puts on the row first (`proc`).
    pub fn gpu_ask(&mut self) -> (Option<f32>, Option<Proc>) {
        (self.total_need(), self.proc())
    }

    /// The next token from a GPU's view of the row the draw sees (`View`: its candidates, and past them its weights and tokens by range), never the row: the token `sample` gives
    /// from the row, seed for seed. The view is of the row the request's changes are put on where it `changes` it. None where the view has none (the generator untouched).
    pub fn sample_view(&mut self, v: &mut dyn View) -> Option<u32> {
        let c = v.cands().filter(|c| !c.is_empty())?;
        let tok = self.draw_view(v, &c)?;
        self.note(tok);
        Some(tok)
    }

    /// `check_draft` from a GPU's view of the drafted row the draw sees (`sample_view`'s): the same token, seed for seed. None where the view has none (the generator untouched).
    pub fn check_view(&mut self, v: &mut dyn View, draft: u32) -> Option<(u32, bool)> {
        let c = v.cands().filter(|c| !c.is_empty())?;
        let tok = if self.p.temperature <= 0.0 {
            c[0].0
        } else {
            let saved = Rng(self.rng.0);
            match self.check_past(v, &c, draft) {
                Some(t) => t,
                None => {
                    self.rng = saved;
                    return None;
                }
            }
        };
        self.note(tok);
        Some((tok, tok == draft))
    }

    /// A draw at a temperature from the candidates `c` of the row the draw sees (the largest first) and the view past them.
    fn draw_view(&mut self, v: &mut dyn View, c: &[(u32, f32)]) -> Option<u32> {
        if self.p.temperature <= 0.0 {
            return Some(c[0].0);
        }
        let saved = Rng(self.rng.0);
        let tok = match self.kept(v, c) {
            Some(Kept::List(idx, w)) => Some(self.walk(&idx, &w)),
            Some(Kept::Past(q, wk)) => {
                let r = pick(self.rng.u64(), wk);
                // (the point among the candidates, the list's first tokens, else past them)
                let wc: Vec<u64> = c.iter().map(|x| weight((x.1 - q.max) / q.temperature)).collect();
                let mut acc = 0u64;
                match wc.iter().position(|&x| {
                    acc += x;
                    acc > r
                }) {
                    Some(i) => Some(c[i].0),
                    None => select(v, &q, Target::Weight(r + 1), None).flatten().map(|a| a.token),
                }
            }
            None => None,
        };
        if tok.is_none() {
            self.rng = saved;
        }
        tok
    }

    /// `check_list` from the candidates `c` and the view past them.
    fn check_past(&mut self, v: &mut dyn View, c: &[(u32, f32)], draft: u32) -> Option<u32> {
        let (q, wk) = match self.kept(v, c)? {
            Kept::List(idx, w) => return Some(self.check_list(&idx, &w, draft)),
            Kept::Past(q, wk) => (q, wk),
        };
        // the draft's place: among the candidates, else its logit read; its weight in the list (0 where the list does not hold it)
        let x = match c.iter().find(|x| x.0 == draft) {
            Some(x) => x.1,
            None => v.logit(draft)?,
        };
        let held = q.holds(x, draft);
        let wd = held.map_or(0, |h| h.1);
        if wd > 0 && pick(self.rng.u64(), wk) < wd {
            return Some(draft);
        }
        // the rest's walk, the draft left out
        let rest = wk - wd;
        let wc: Vec<(u32, u64)> = c.iter().filter(|x| x.0 != draft).map(|x| (x.0, weight((x.1 - q.max) / q.temperature))).collect();
        let r = pick(self.rng.u64(), rest);
        if rest == 0 {
            return Some(wc[0].0);
        }
        let mut acc = 0u64;
        if let Some(&(t, _)) = wc.iter().find(|&&(_, w)| {
            acc += w;
            acc > r
        }) {
            return Some(t);
        }
        select(v, &q, Target::Weight(r + 1), held).flatten().map(|a| a.token)
    }

    /// The list a draw walks (`dist`, then `cut`) from the candidates `c` of the row it sees (the largest first; their largest not -inf) and the view past them: of the candidates
    /// alone where the top-k is within them, or the row's tokens are all among them, or the min-p or the top-p cut falls among them; else the query of its tokens past them (the
    /// top-k's k-th token found by count, the min-p a floor on the quotient, the top-p's token found by weight). None where the view gives none.
    fn kept(&self, v: &mut dyn View, c: &[(u32, f32)]) -> Option<Kept> {
        let p = &self.p;
        let (max, t, k) = (c[0].1, p.temperature, p.top_k);
        let gram = self.gram.is_some();
        // (a row of -inf: the list's first alone has a place, its weight 0; under a grammar the first of the row, as `dist` takes it)
        if !(max > f32::NEG_INFINITY) {
            let n = if gram || k == 1 { 1 } else { 2.min(c.len()) };
            return Some(Kept::List(c[..n].iter().map(|x| x.0).collect(), vec![0; n]));
        }
        let fin = c.iter().take_while(|x| x.1 > f32::NEG_INFINITY).count();
        let list = |n: usize| {
            // (as `dist` gives it: under a grammar the tokens taken out are not in it)
            let l: Vec<&(u32, f32)> = c[..n].iter().filter(|x| !gram || x.1 > f32::NEG_INFINITY).collect();
            self.cut(l.iter().map(|x| x.0).collect(), l.iter().map(|x| x.1).collect())
        };
        if (1..=c.len()).contains(&k) {
            let (idx, w) = list(k);
            return Some(Kept::List(idx, w));
        }
        if fin < c.len() {
            // (every token of the row that is not -inf is a candidate: the row's -inf ones weigh 0, and under a grammar are not in it)
            let (idx, w) = list(fin);
            return Some(Kept::List(idx, w));
        }
        let mut q = Query::all(max, t);
        if k > 0 {
            if let Some(at) = select(v, &q, Target::Count(k as u64), None)? {
                q.lo = at.key;
            }
        }
        let f = if p.min_p > 0.0 { min_floor(p.min_p) } else { f32::NEG_INFINITY };
        if f > 0.0 {
            // (a floor above the largest's quotient: `cut` keeps the largest alone)
            return Some(Kept::List(vec![c[0].0], vec![weight(0.0)]));
        }
        let qc: Vec<f32> = c.iter().map(|x| (x.1 - max) / t).collect();
        let wc: Vec<u64> = qc.iter().map(|&x| weight(x)).collect();
        let m = qc.iter().take_while(|&&x| x >= f).count();
        let (top_p, min_p) = (p.top_p < 1.0, p.min_p > 0.0);
        let total = match (top_p || !min_p, q.lo) {
            (false, _) => 0,
            (true, 0) => v.total(max, t)?,
            (true, _) => v.bins(&q)?.weight.iter().sum(),
        };
        let cut = top_p.then(|| share(p.top_p, total));
        if let Some(cut) = cut {
            let mut acc = 0u64;
            if let Some(i) = wc[..m].iter().position(|&w| {
                acc += w;
                acc >= cut
            }) {
                return Some(Kept::List(c[..=i].iter().map(|x| x.0).collect(), wc[..=i].to_vec()));
            }
        }
        if m < c.len() {
            return Some(Kept::List(c[..m].iter().map(|x| x.0).collect(), wc[..m].to_vec()));
        }
        let q = Query { floor: f, ..q };
        let whole = |v: &mut dyn View| v.bins(&q).map(|b| b.weight.iter().sum::<u64>());
        Some(match cut {
            Some(cut) => match select(v, &q, Target::Weight(cut), None)? {
                Some(at) => Kept::Past(Query { lo: at.key, ..q }, at.before_w + at.w),
                None => Kept::Past(q, whole(v)?),
            },
            None if !min_p => Kept::Past(q, total),
            None => Kept::Past(q, whole(v)?),
        })
    }

    fn draw(&mut self, l: &mut [f32]) -> u32 {
        let (idx, w) = self.dist(l);
        self.walk(&idx, &w)
    }

    /// The draw's list and its tokens' weights (in the draw's order: the larger logit first, the smaller token of equal ones): top-k, temperature, min-p and top-p over the processed row.
    fn dist(&self, l: &mut [f32]) -> (Vec<u32>, Vec<u64>) {
        // the candidates by probability, largest first, cut by top-k, then min-p, then top-p
        if self.gram.is_some() && self.p.top_k > 0 {
            // (a grammar leaves a share of the row, often most of it: the top-k is taken in one pass, the logits with their tokens beside them, a buffer of twice k cut to k each time it fills, and the k-th
            // largest logit so far a floor that the rest must pass; the order is the one below's, the larger logit first and, of equal ones, the smaller token)
            // (a request's top_k is any number: no more than the row has)
            let (k, by) =
                (self.p.top_k.min(l.len()), |a: &(f32, u32), b: &(f32, u32)| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            let mut c: Vec<(f32, u32)> = Vec::with_capacity(2 * k + 1);
            let mut floor = f32::NEG_INFINITY;
            for (i, &x) in l.iter().enumerate() {
                if x > floor {
                    c.push((x, i as u32));
                    if c.len() >= 2 * k {
                        c.select_nth_unstable_by(k - 1, by);
                        c.truncate(k);
                        floor = c[k - 1].0;
                    }
                }
            }
            if !c.is_empty() {
                if c.len() > k {
                    c.select_nth_unstable_by(k - 1, by);
                    c.truncate(k);
                }
                c.sort_by(by);
                return self.cut(c.iter().map(|x| x.1).collect(), c.iter().map(|x| x.0).collect());
            }
        }
        let mut idx: Vec<u32> = (0..l.len() as u32).collect();
        if self.gram.is_some() {
            // (a grammar leaves a few tokens of the row: the rest are not sorted)
            idx.retain(|&i| l[i as usize] > f32::NEG_INFINITY);
            if idx.is_empty() {
                idx.push(0);
            }
        }
        let by_logit = |a: &u32, b: &u32| l[*b as usize].partial_cmp(&l[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b));
        if self.p.top_k > 0 && self.p.top_k < idx.len() {
            idx.select_nth_unstable_by(self.p.top_k - 1, by_logit);
            idx.truncate(self.p.top_k);
        }
        idx.sort_by(by_logit);
        let vals: Vec<f32> = idx.iter().map(|&i| l[i as usize]).collect();
        self.cut(idx, vals)
    }

    /// The candidates `idx` (their logits `vals`, largest first) after the min-p and top-p cuts, with their weights (`weight` of their quotients at the temperature, about the first): the
    /// tokens whose quotient is at least min-p's floor (`min_floor`; the first at least), then up to the first whose cumulative weight reaches top-p of the whole list's (all of those
    /// where none does).
    fn cut(&self, mut idx: Vec<u32>, vals: Vec<f32>) -> (Vec<u32>, Vec<u64>) {
        let max = vals[0];
        let qs: Vec<f32> = vals.iter().map(|&v| (v - max) / self.p.temperature).collect();
        let mut w: Vec<u64> = qs.iter().map(|&q| weight(q)).collect();
        let total: u64 = w.iter().sum();
        if self.p.min_p > 0.0 {
            let floor = min_floor(self.p.min_p);
            let keep = qs.iter().take_while(|&&q| q >= floor).count().max(1);
            w.truncate(keep);
            idx.truncate(keep);
        }
        if self.p.top_p < 1.0 {
            let cut = share(self.p.top_p, total);
            let mut acc = 0u64;
            let keep = w
                .iter()
                .position(|&x| {
                    acc += x;
                    acc >= cut
                })
                .map_or(w.len(), |i| i + 1);
            w.truncate(keep.max(1));
            idx.truncate(keep.max(1));
        }
        (idx, w)
    }

    /// The generator's draw among the tokens `idx` of weights `w` (in the draw's order): one number, landed on their sum (`pick`, `land`).
    fn walk(&mut self, idx: &[u32], w: &[u64]) -> u32 {
        let r = pick(self.rng.u64(), w.iter().sum());
        land(idx, w, r)
    }
}

/// What a step says of a token: its log-probability and the likeliest tokens', all of the row's raw logits as the model made them (the log-softmax before the bias, the penalties, the temperature and
/// top-k or top-p: what vLLM's default `raw_logprobs` mode gives, whatever the draw looked at).
#[derive(Clone, Debug, PartialEq)]
pub struct Lp {
    pub token: u32,
    pub logprob: f64,
    /// The `k` likeliest tokens and their log-probabilities, the likeliest first (equal ones by the smaller token).
    pub top: Vec<(u32, f64)>,
}

/// `token`'s log-probability under `row` (one logit per token) and the `k` likeliest tokens'. `cands`: the GPU's largest logits of the row, largest first (a list of at least `k` gives the `k` likeliest without
/// a search of the row; the row's own log-sum-exp is still read from it).
pub fn logprobs_of(row: &[f32], token: u32, k: usize, cands: Option<&[(u32, f32)]>) -> Lp {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = row.iter().map(|&x| x as f64 - max).filter(|&d| d > -745.0).map(f64::exp).sum();
    let lse = max + sum.ln();
    let logprob = row.get(token as usize).map_or(f64::NEG_INFINITY, |&x| x as f64 - lse);
    let top = match cands {
        Some(c) if k <= c.len() => c[..k].iter().map(|&(t, x)| (t, x as f64 - lse)).collect(),
        _ => {
            let mut idx: Vec<u32> = (0..row.len() as u32).collect();
            let by_logit = |a: &u32, b: &u32| row[*b as usize].partial_cmp(&row[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b));
            if k < idx.len() && k > 0 {
                idx.select_nth_unstable_by(k - 1, by_logit);
            }
            idx.truncate(k);
            idx.sort_by(by_logit);
            idx.into_iter().map(|t| (t, row[t as usize] as f64 - lse)).collect()
        }
    };
    Lp { token, logprob, top }
}

/// The first largest of `l` (torch.argmax's choice among equals).
pub fn argmax(l: &[f32]) -> u32 {
    let (mut best, mut at) = (f32::NEG_INFINITY, 0usize);
    for (i, &x) in l.iter().enumerate() {
        if x > best {
            best = x;
            at = i;
        }
    }
    at as u32
}

/// What a GPU's kernels make of a row of logits, made on the host (the tests, and the kernels' checks: kcheck.rs): its candidates (`topk64_*`), its total, bins and tokens of a
/// query (`draw_total_*`, `draw_bins_*`, `draw_tokens_*`), a row as their view, and the draw an engine makes from them.
#[cfg(any(test, feature = "dev"))]
pub mod emu {
    use super::*;

    /// The `kern::TOPK` largest of a row, largest first, equal ones by the smaller token (`topk64_f32`, `topk64_bf16`).
    pub fn top(w: &[f32]) -> Vec<(u32, f32)> {
        let mut i: Vec<u32> = (0..w.len() as u32).collect();
        i.sort_by(|&a, &b| key32(w[b as usize]).cmp(&key32(w[a as usize])).then(a.cmp(&b)));
        i[..crate::kern::TOPK.min(w.len())].iter().map(|&t| (t, w[t as usize])).collect()
    }

    /// The sum of the weights of the row's tokens (not -inf) about `max` at temperature `t`.
    pub fn total(w: &[f32], max: f32, t: f32) -> u64 {
        bins(w, &Query::all(max, t)).weight.iter().sum()
    }

    /// The query's tokens of the row in its bins.
    pub fn bins(w: &[f32], q: &Query) -> Bins {
        let mut b = Bins::empty();
        for (t, &x) in w.iter().enumerate() {
            if let Some((k, wt)) = q.holds(x, t as u32) {
                b.add(k, wt, q.shift);
            }
        }
        b
    }

    /// The query's tokens of the row (token and logit), in the row's order.
    pub fn tokens(w: &[f32], q: &Query) -> Vec<(u32, f32)> {
        w.iter().enumerate().filter(|(t, &x)| q.holds(x, *t as u32).is_some()).map(|(t, &x)| (t as u32, x)).collect()
    }

    /// A row on the host as a GPU's view of it, counting the queries past the candidates it answers (its total is made with the step's candidates, as a GPU's is: not counted).
    pub struct Row<'a> {
        pub row: &'a [f32],
        pub asks: usize,
    }

    impl View for Row<'_> {
        fn cands(&mut self) -> Option<Vec<(u32, f32)>> {
            Some(top(self.row))
        }
        fn total(&mut self, max: f32, t: f32) -> Option<u64> {
            Some(total(self.row, max, t))
        }
        fn bins(&mut self, q: &Query) -> Option<Bins> {
            self.asks += 1;
            Some(bins(self.row, q))
        }
        fn tokens(&mut self, q: &Query, cap: usize) -> Option<Vec<(u32, f32)>> {
            self.asks += 1;
            Some(tokens(self.row, q)).filter(|v| v.len() <= cap)
        }
        fn logit(&mut self, t: u32) -> Option<f32> {
            self.row.get(t as usize).copied()
        }
    }

    /// The token an engine draws for `s` from `row` (the model's logits) by a GPU's view of the row it sees, and whether it read the row. Where the request changes the row, that
    /// row as the GPU makes it is held to `processed`'s bit for bit first.
    pub fn draw(s: &mut Sampler, row: &[f32]) -> (u32, bool) {
        let w = match s.proc() {
            Some(p) => {
                let w = p.resolve(row.len()).apply(row);
                let host = s.processed(row);
                assert!(w.iter().zip(&host).all(|(a, b)| a.to_bits() == b.to_bits()), "the row the GPU makes is not the host's");
                w
            }
            None => row.to_vec(),
        };
        match s.sample_view(&mut Row { row: &w, asks: 0 }) {
            Some(t) => (t, false),
            None => (s.sample(row), true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(mut x: u64) -> impl FnMut() -> u64 {
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
    }

    /// A weight is exp(q) to within its float's rounding (4e-7 of it) and a unit of 2^-40 of the largest's, at the floats from -32 to 0 (every 61st, each k's boundaries on the
    /// way); 0 at and below -32, for -inf and for not a number.
    #[test]
    fn a_weight_is_exp_to_its_rounding() {
        assert_eq!(
            (weight(0.0), weight(-0.0), weight(f32::NEG_INFINITY), weight(f32::NAN), weight(-32.0), weight(-40.0)),
            (1 << WBITS, 1 << WBITS, 0, 0, 0, 0)
        );
        for b in (0x8000_0001u32..0xC200_0000).step_by(61) {
            let q = f32::from_bits(b);
            let (w, e) = (weight(q), (q as f64).exp() * (1u64 << WBITS) as f64);
            assert!((w as f64 - e).abs() <= e * 4e-7 + 1.0, "q {q}: weight {w}, exp {e}");
        }
    }

    /// `share` is the ceiling of p w, exactly: against the product in whole numbers of p's own fraction, for floats of every size and weights up to a row's.
    #[test]
    fn a_share_is_the_ceiling_of_the_product() {
        let mut next = xorshift(0x2545_F491_4F6C_DD1D);
        for i in 0..20_000 {
            let p = match i % 4 {
                0 => (next() % 1_000_000) as f32 / 1_000_000.0,
                1 => f32::from_bits(0x3000_0000 + (next() % 0x0F80_0000) as u32),
                2 => [0.95, 0.9, 0.5, 0.05, 0.1, 1.0, 0.999, 1e-30, 1.5][i / 4 % 9],
                _ => f32::from_bits((next() % 0x0080_0000) as u32),
            };
            let w = next() >> (next() % 64);
            // (p = m / 2^s: the ceiling of m w / 2^s)
            let (m, s) = {
                let b = p.to_bits();
                let e = (b >> 23) & 0xFF;
                if e == 0 {
                    ((b & 0x7F_FFFF) as u128, 149)
                } else {
                    (((b & 0x7F_FFFF) | 0x80_0000) as u128, 150 - e as i32)
                }
            };
            let x = m * w as u128;
            let want = if s >= 128 { (x > 0) as u128 } else { x.div_ceil(1u128 << s) };
            assert_eq!(share(p, w) as u128, want.min(u64::MAX as u128), "p {p} w {w}");
        }
        assert_eq!((share(0.0, 9), share(-1.0, 9), share(f32::NAN, 9), share(0.5, 0), share(0.5, 9), share(1.0, 9)), (0, 0, 0, 0, 5, 9));
        assert_eq!(share(1e30, u64::MAX), u64::MAX);
    }

    /// Rows of every kind a draw meets, of `v` tokens: a sure token, a few alike, a hundred alike, a flat row (its 95% nucleus thousands of tokens), all equal, ramps, a tail of -inf,
    /// fewer finite tokens than the candidates, and bf16 values with many equal ones.
    fn kinds_of_rows(v: usize, seed: u64) -> Vec<(String, Vec<f32>)> {
        let mut next = xorshift(seed);
        let bf = |x: f32| f32::from_bits(x.to_bits() & 0xFFFF_0000);
        let mut rows = Vec::new();
        let floor = |next: &mut dyn FnMut() -> u64| -> Vec<f32> { (0..v).map(|_| bf(-12.0 + (next() % 400) as f32 / 100.0)).collect() };
        for (name, hot, level) in [("a sure token", 4, 8.0f32), ("thirty alike", 30, 4.0), ("a hundred alike", 100, 3.0)] {
            let mut r = floor(&mut next);
            for _ in 0..hot {
                r[(next() % v as u64) as usize] = bf(level + (next() % 300) as f32 / 100.0);
            }
            rows.push((name.to_string(), r));
        }
        rows.push(("flat".into(), (0..v).map(|_| bf((next() % 1000) as f32 / 4000.0)).collect()));
        rows.push(("flat, f32".into(), (0..v).map(|_| (next() % 1_000_000) as f32 / 3e6).collect()));
        rows.push(("all equal".into(), vec![1.5; v]));
        rows.push(("a ramp".into(), (0..v).map(|t| -(t as f32) / v as f32 * 6.0).collect()));
        rows.push(("a falling ramp of ties".into(), (0..v).map(|t| bf(-((t / 37) as f32) * 0.125)).collect()));
        let mut r = floor(&mut next);
        for x in r.iter_mut().skip(v / 3) {
            *x = f32::NEG_INFINITY;
        }
        rows.push(("a third, the rest -inf".into(), r));
        let mut r = vec![f32::NEG_INFINITY; v];
        for _ in 0..40 {
            r[(next() % v as u64) as usize] = bf((next() % 100) as f32 / 10.0);
        }
        rows.push(("40 finite".into(), r));
        rows.push(("both zeros and far ones".into(), (0..v).map(|t| [0.0, -0.0, -40.0, -25.0][(next() % 4) as usize] + (t % 3) as f32 * 0.0).collect()));
        rows
    }

    /// The samplings a draw meets: greedy; top-k within the candidates and past them (200, 5,000, more than the row); top-p and min-p alone and together, near 1 and far; temperatures.
    fn samplings() -> Vec<Params> {
        let d = Params::default();
        let mut out = vec![Params { temperature: 0.0, ..d.clone() }];
        for (temperature, top_k, top_p, min_p) in [
            (1.0, 0, 1.0, 0.0),
            (0.7, 0, 0.95, 0.0),
            (1.0, 0, 0.999, 0.0),
            (1.3, 0, 0.5, 0.0),
            (1.0, 0, 1.0, 0.05),
            (0.9, 0, 1.0, 0.5),
            (1.1, 0, 0.8, 0.02),
            (1.0, 20, 0.9, 0.0),
            (1.0, 64, 1.0, 0.1),
            (1.0, 65, 1.0, 0.0),
            (0.8, 200, 0.95, 0.0),
            (1.0, 200, 1.0, 0.01),
            (1.0, 5000, 0.9, 0.05),
            (2.0, 1 << 40, 0.99, 0.0),
            (0.3, 0, 0.9, 0.0),
            (1.0, 1, 1.0, 0.0),
            (1.0, 0, 1.0, 1.5),
            (1.0, 0, 0.0, 0.0),
        ] {
            out.push(Params { temperature, top_k, top_p, min_p, ..d.clone() });
        }
        out
    }

    /// A draw from a GPU's view of the row (its candidates, and past them the weights and tokens of ranges: `View`, `select`) is the draw from the row, seed for seed, for every kind
    /// of row and every sampling, and never reads the row: a flat row's 95% nucleus of thousands of tokens, a top-k of 200 or 5,000, a min-p past the candidates. The queries past
    /// the candidates are asked where the cut or the point lies past them, and none where it does not.
    #[test]
    fn a_draw_from_a_view_is_the_draw_from_the_row() {
        let (mut draws, mut past) = (0usize, 0usize);
        for v in [3000usize, 30_000] {
            for (name, row) in kinds_of_rows(v, 0x9E37_79B9_7F4A_7C15 ^ v as u64) {
                for (i, base) in samplings().into_iter().enumerate() {
                    let p = Params { seed: (i * 7 + v) as u64, ..base };
                    let (mut whole, mut gpu) = (Sampler::new(p.clone(), &[]), Sampler::new(p.clone(), &[]));
                    let mut asks = 0;
                    for step in 0..6 {
                        let mut view = emu::Row { row: &row, asks: 0 };
                        let got = gpu.sample_view(&mut view);
                        let want = whole.sample(&row);
                        assert_eq!(got, Some(want), "{name}, {v} tokens, {p:?}, step {step}");
                        asks += view.asks;
                        draws += 1;
                    }
                    past += (asks > 0) as usize;
                    // (a draw the candidates hold asks nothing past them: greedy, a top-k within them, a sure row's top-p)
                    if p.temperature <= 0.0 || (1..=64).contains(&p.top_k) || (name == "a sure token" && p.top_k == 0 && p.top_p == 0.95) {
                        assert_eq!(asks, 0, "{name}, {p:?}");
                    }
                }
            }
        }
        assert!(past > 100, "{past} samplings of {draws} draws asked past the candidates");
    }

    /// A flat row's nucleus: thousands of tokens hold 95% of its weight, and the draw lands among them as the row's does, from what a GPU gives of a few ranges, not the row.
    #[test]
    fn a_flat_rows_nucleus_of_thousands_is_drawn_without_the_row() {
        let v = 262_144usize;
        let mut next = xorshift(0x51ED_270B_27C3_B1A5);
        let row: Vec<f32> = (0..v).map(|_| f32::from_bits((((next() % 2000) as f32 / 1000.0).to_bits()) & 0xFFFF_0000)).collect();
        for (temperature, top_p, min_p, top_k) in
            [(1.0f32, 0.95f32, 0.0f32, 0usize), (0.7, 0.9, 0.0, 0), (1.0, 1.0, 0.3, 0), (1.0, 0.95, 0.0, 20_000), (1.0, 1.0, 0.0, 0)]
        {
            let p = Params { temperature, top_p, min_p, top_k, seed: 11, ..Params::default() };
            let (mut whole, mut gpu) = (Sampler::new(p.clone(), &[]), Sampler::new(p.clone(), &[]));
            // (the nucleus: the tokens the row's own cut keeps)
            let (kept, _) = whole.dist(&mut row.clone());
            assert!(kept.len() > 1000, "{p:?}: a nucleus of {}", kept.len());
            for step in 0..8 {
                let mut view = emu::Row { row: &row, asks: 0 };
                assert_eq!(gpu.sample_view(&mut view), Some(whole.sample(&row)), "{p:?}, step {step}");
                assert!(view.asks <= 24, "{p:?}: {} queries", view.asks);
            }
        }
    }

    /// The draw an engine makes from the GPU's view of the row the draw sees (the penalties, a bias, allowed tokens and the tokens held back put on the row first, `Proc`) is the
    /// draw from the model's whole row (`sample`), seed for seed and step by step as the penalties' counts grow, whatever the sampling; the row the GPU makes is `processed`'s bit for
    /// bit at every step, and the row is never read.
    #[test]
    fn a_draw_from_the_row_the_gpu_makes_is_the_rows() {
        let v = 3000usize;
        let mut seed = 0x51ed_270b_27c3_b1a5u64;
        let mut unit = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        let draws = [
            Params { temperature: 0.0, ..Params::default() },
            Params { temperature: 0.8, top_k: 20, top_p: 0.9, ..Params::default() },
            Params { temperature: 1.0, top_k: 64, min_p: 0.05, ..Params::default() },
            Params { temperature: 0.7, top_k: 1, ..Params::default() },
            Params { temperature: 1.0, top_p: 0.95, ..Params::default() },
            Params { temperature: 0.9, min_p: 0.1, ..Params::default() },
            Params { temperature: 1.1, top_p: 0.8, min_p: 0.02, ..Params::default() },
            Params { temperature: 1.0, ..Params::default() },
            Params { temperature: 0.6, top_k: 100, ..Params::default() },
            Params { temperature: 1.0, top_k: 200, top_p: 0.9, ..Params::default() },
        ];
        let changes = [
            Params { repetition_penalty: 1.3, ..Params::default() },
            Params { repetition_penalty: 0.8, ..Params::default() },
            Params { presence_penalty: 0.6, frequency_penalty: 0.4, ..Params::default() },
            Params { logit_bias: vec![(5, 3.0), (2999, -2.0), (17, 9.0), (1234, 100.0)], ..Params::default() },
            Params {
                logit_bias: vec![(7, -100.0), (9, 1.0), (9, 2.5)],
                repetition_penalty: 1.2,
                frequency_penalty: -0.5,
                presence_penalty: 0.25,
                ..Params::default()
            },
            Params { allowed: (0..v as u32).step_by(7).chain([4, 4, 9999]).collect(), presence_penalty: 1.0, ..Params::default() },
            Params { min_tokens: 30, repetition_penalty: 1.1, ..Params::default() },
        ];
        for (i, d) in draws.iter().enumerate() {
            for (j, c) in changes.iter().enumerate() {
                let p = Params { temperature: d.temperature, top_k: d.top_k, top_p: d.top_p, min_p: d.min_p, seed: (i * 31 + j) as u64, ..c.clone() };
                let prompt: Vec<u32> = (0..40).map(|i| (i * 37 % v) as u32).collect();
                let (mut whole, mut gpu) = (Sampler::new(p.clone(), &prompt), Sampler::new(p.clone(), &prompt));
                for s in [&mut whole, &mut gpu] {
                    s.set_suppress(vec![0, 1, 7]);
                }
                for step in 0..60 {
                    // a row of bf16 values (as the GPU's logits are): a low floor, a few tied, both zeros, a few tokens far above it and the tokens said so far among them;
                    // every fifth step flat
                    let bf = |x: f32| f32::from_bits(x.to_bits() & 0xffff_0000);
                    let spread = if step % 5 == 4 { 0.5 } else { 8.0 };
                    let mut row: Vec<f32> = (0..v).map(|_| bf(unit() * spread - 12.0)).collect();
                    for t in (0..v).step_by(97) {
                        row[t] = row[(t + 1) % v];
                    }
                    row[11] = -0.0;
                    row[12] = 0.0;
                    if step % 5 != 4 {
                        for _ in 0..5 {
                            row[(unit() * v as f32) as usize % v] = bf(unit() * 5.0 + 3.0);
                        }
                        for (&t, _) in whole.said.iter().take(6) {
                            row[t as usize] = 7.5;
                        }
                    }
                    let (got, read) = emu::draw(&mut gpu, &row);
                    let want = whole.sample(&row);
                    assert_eq!(got, want, "{p:?}, step {step}");
                    assert!(!read, "{p:?}, step {step}: the row was read");
                }
            }
        }
    }

    /// A drafted token checked from a GPU's view of its row (`check_view`) is `check_draft`'s, seed for seed: drafts among the candidates, past them, and out of the draw's list,
    /// on rows whose list ends among the candidates and past them, with and without a request's changes.
    #[test]
    fn a_draft_checked_from_a_view_is_checked_as_from_the_row() {
        let v = 5000usize;
        let mut next = xorshift(0xD1B5_4A32_D192_ED03);
        let mut past = 0;
        for (name, row) in kinds_of_rows(v, 77) {
            for (i, base) in samplings().into_iter().enumerate() {
                for change in [Params::default(), Params { repetition_penalty: 1.2, logit_bias: vec![(3, 2.0)], ..Params::default() }] {
                    let p = Params { temperature: base.temperature, top_k: base.top_k, top_p: base.top_p, min_p: base.min_p, seed: i as u64, ..change };
                    let (mut whole, mut gpu) = (Sampler::new(p.clone(), &[3, 9]), Sampler::new(p.clone(), &[3, 9]));
                    for step in 0..5 {
                        let top = emu::top(&row);
                        let draft = match step {
                            0 => top[0].0,
                            1 => top[(next() % 64) as usize].0,
                            _ => (next() % v as u64) as u32,
                        };
                        let w = match gpu.proc() {
                            Some(pr) => pr.resolve(v).apply(&row),
                            None => row.clone(),
                        };
                        let mut view = emu::Row { row: &w, asks: 0 };
                        let got = gpu.check_view(&mut view, draft);
                        assert_eq!(got, Some(whole.check_draft(&row, draft)), "{name}, {p:?}, step {step}, draft {draft}");
                        past += (view.asks > 0) as usize;
                    }
                }
            }
        }
        assert!(past > 100, "{past} checks asked past the candidates");
    }

    /// A view that cannot answer (a query it has no bins for) leaves the generator where it was: the draw is then made from the row, as if the view had not been asked.
    #[test]
    fn a_view_that_fails_leaves_the_draw_to_the_row() {
        struct Cands<'a>(&'a [f32]);
        impl View for Cands<'_> {
            fn cands(&mut self) -> Option<Vec<(u32, f32)>> {
                Some(emu::top(self.0))
            }
            fn bins(&mut self, _: &Query) -> Option<Bins> {
                None
            }
            fn tokens(&mut self, _: &Query, _: usize) -> Option<Vec<(u32, f32)>> {
                None
            }
            fn logit(&mut self, _: u32) -> Option<f32> {
                None
            }
        }
        let row: Vec<f32> = (0..3000).map(|t| ((t * 7919) % 1000) as f32 / 5000.0).collect();
        let p = Params { temperature: 1.0, top_p: 0.95, seed: 5, ..Params::default() };
        let (mut a, mut b) = (Sampler::new(p.clone(), &[]), Sampler::new(p, &[]));
        for _ in 0..5 {
            assert_eq!(a.sample_view(&mut Cands(&row)), None);
            assert_eq!(a.sample(&row), b.sample(&row));
        }
        assert_eq!(a.check_view(&mut Cands(&row), 3), None);
    }

    /// What a GPU is asked to put on the row: nothing where the request changes nothing; the bias, the penalties' tokens as the answer grows (the prompt's for the repetition penalty), the allowed
    /// list and the tokens held back until the least length (and not after); a bias that names a token twice: its last (as the server's JSON object keeps its last key).
    #[test]
    fn the_changes_a_gpu_puts_on_the_row() {
        assert!(Sampler::new(Params { temperature: 0.7, top_k: 5, ..Params::default() }, &[3]).proc().is_none());
        let mut s = Sampler::new(Params { repetition_penalty: 1.5, presence_penalty: 0.5, logit_bias: vec![(9, 2.0)], ..Params::default() }, &[3, 4, 3]);
        let r = s.proc().unwrap().resolve(16);
        assert_eq!(r.words, None);
        let mut e = r.entries.clone();
        e.sort_by_key(|e| e.token);
        assert_eq!(
            e,
            vec![
                Entry { token: 3, bias: None, seen: true, said: 0, rescue: false },
                Entry { token: 4, bias: None, seen: true, said: 0, rescue: false },
                Entry { token: 9, bias: Some(2.0), seen: false, said: 0, rescue: false },
            ]
        );
        let row = vec![1.0f32; 16];
        s.sample(&row);
        s.sample(&row);
        // (greedy: token 0 twice, said twice; token 9 is the bias's)
        let r = s.proc().unwrap().resolve(16);
        assert!(r.entries.contains(&Entry { token: 9, bias: Some(2.0), seen: true, said: 2, rescue: false }), "{:?}", r.entries);
        // the allowed list and the tokens held back (past the vocabulary: none)
        let mut s = Sampler::new(Params { allowed: vec![2, 40, 5, 2, 100], min_tokens: 1, ..Params::default() }, &[]);
        s.set_suppress(vec![5, 70]);
        let r = s.proc().unwrap().resolve(64);
        assert_eq!(r.words, Some(vec![1 << 2, 1 << 8]));
        s.sample(&[0.0; 64]);
        assert_eq!(s.proc().unwrap().resolve(64).words, Some(vec![(1 << 2) | (1 << 5), 1 << 8]), "the least length reached: nothing held back");
        let mut twice = Sampler::new(Params { logit_bias: vec![(1, 1.0), (1, 2.0)], ..Params::default() }, &[]);
        assert_eq!(twice.proc().unwrap().resolve(4).entries, vec![Entry { token: 1, bias: Some(2.0), seen: false, said: 0, rescue: false }]);
        assert_eq!(twice.processed(&[0.0; 4]), vec![0.0, 2.0, 0.0, 0.0]);
    }

    fn logits() -> Vec<f32> {
        // token 3 the largest, then 1, then 7, the rest low
        let mut l = vec![-5.0f32; 10];
        l[3] = 4.0;
        l[1] = 3.0;
        l[7] = 2.0;
        l
    }

    #[test]
    fn a_models_defaults() {
        let dir = std::env::temp_dir().join(format!("glyd-engine-gen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(generation_defaults(&dir).temperature, 1.0);
        std::fs::write(dir.join("generation_config.json"), r#"{"do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95}"#).unwrap();
        let p = generation_defaults(&dir);
        assert_eq!((p.temperature, p.top_k, p.top_p, p.repetition_penalty), (0.6, 20, 0.95, 1.0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn temperature_zero_is_the_first_largest() {
        let mut s = Sampler::new(Params::default(), &[]);
        assert_eq!(s.sample(&logits()), 3);
        let mut tie = logits();
        tie[5] = 4.0;
        assert_eq!(s.sample(&tie), 3);
    }

    #[test]
    fn a_request_with_no_seed_gets_its_own() {
        let seeds: std::collections::HashSet<u64> = (0..1000).map(|_| fresh_seed()).collect();
        assert_eq!(seeds.len(), 1000);
    }

    #[test]
    fn the_same_seed_draws_the_same_tokens() {
        let p = Params { temperature: 1.0, seed: 42, ..Params::default() };
        let (mut a, mut b) = (Sampler::new(p.clone(), &[]), Sampler::new(p, &[]));
        let (x, y): (Vec<u32>, Vec<u32>) = ((0..50).map(|_| a.sample(&logits())).collect(), (0..50).map(|_| b.sample(&logits())).collect());
        assert_eq!(x, y);
        let mut c = Sampler::new(Params { temperature: 1.0, seed: 43, ..Params::default() }, &[]);
        assert_ne!(x, (0..50).map(|_| c.sample(&logits())).collect::<Vec<_>>());
    }

    #[test]
    fn top_k_top_p_and_min_p_cut_the_candidates() {
        let draws = |p: Params| -> std::collections::HashSet<u32> {
            let mut s = Sampler::new(p, &[]);
            (0..400).map(|_| s.sample(&logits())).collect()
        };
        let base = Params { temperature: 2.0, seed: 1, ..Params::default() };
        assert!(draws(base.clone()).len() > 3);
        assert_eq!(draws(Params { top_k: 1, ..base.clone() }), [3].into());
        assert!(draws(Params { top_k: 2, ..base.clone() }).is_subset(&[3, 1].into()));
        assert!(draws(Params { top_p: 0.5, ..base.clone() }).len() <= 3);
        assert!(draws(Params { min_p: 0.5, ..base }).is_subset(&[3, 1].into()));
    }

    /// The draws keep the distribution the weights say: a token's share of many draws is its weight's share of the list's, for a top-p, a min-p and a whole row.
    #[test]
    fn the_draws_are_distributed_as_the_weights() {
        let row = vec![2.0f32, 1.5, 0.3, -1.0, 0.9, 1.4, -2.0, 0.0];
        for p in [
            Params { temperature: 0.8, seed: 7, ..Params::default() },
            Params { temperature: 1.0, top_p: 0.9, seed: 8, ..Params::default() },
            Params { temperature: 1.2, min_p: 0.2, seed: 9, ..Params::default() },
        ] {
            let mut s = Sampler::new(p.clone(), &[]);
            let (idx, w) = s.dist(&mut row.clone());
            let total: u64 = w.iter().sum();
            let n = 200_000;
            let mut c = vec![0f64; row.len()];
            for _ in 0..n {
                c[s.sample(&row) as usize] += 1.0;
            }
            for (t, x) in idx.iter().zip(&w) {
                let (got, want) = (c[*t as usize] / n as f64, *x as f64 / total as f64);
                assert!((got - want).abs() < 0.005, "{p:?}: token {t} drawn {got:.4} of the time, its weight {want:.4}");
            }
        }
    }

    #[test]
    fn what_a_draw_asks_of_the_gpu() {
        let ask = |p: Params| Sampler::new(p, &[]).total_need();
        let d = Params { temperature: 0.7, ..Params::default() };
        for (p, want) in [
            (Params { top_k: 20, ..d.clone() }, None),
            (Params { top_k: 200, top_p: 0.9, ..d.clone() }, None),
            (Params { top_p: 0.95, ..d.clone() }, Some(0.7)),
            (d.clone(), Some(0.7)),
            (Params { top_p: 0.95, min_p: 0.05, ..d.clone() }, Some(0.7)),
            (Params { min_p: 0.05, ..d.clone() }, None),
            (Params { temperature: 0.0, ..d.clone() }, None),
        ] {
            assert_eq!(ask(p.clone()), want, "{p:?}");
        }
        // (a least length with tokens to take out: the row the draw sees, until it is reached)
        let mut s = Sampler::new(Params { temperature: 1.0, top_k: 20, min_tokens: 1, ..Params::default() }, &[]);
        s.set_suppress(vec![3]);
        assert!(s.changes() && s.gpu_ask().1.is_some());
        s.sample(&logits());
        assert!(!s.changes() && s.gpu_ask().1.is_none());
    }

    #[test]
    fn penalties_move_what_was_said() {
        // the largest, said once, is overtaken by the second under a presence penalty greater than the gap
        let mut s = Sampler::new(Params { temperature: 0.0, presence_penalty: 1.5, ..Params::default() }, &[]);
        assert_eq!(s.sample(&logits()), 3);
        assert_eq!(s.sample(&logits()), 1);
        // the repetition penalty acts on the prompt's tokens too
        let mut r = Sampler::new(Params { temperature: 0.0, repetition_penalty: 2.0, ..Params::default() }, &[3]);
        assert_eq!(r.sample(&logits()), 1, "token 3's 4.0 is halved to 2.0, so token 1's 3.0 is the largest");
    }

    /// vLLM's logit_bias: the bias is added to the logits of the row before the penalties and the temperature, so it moves greedy choices too, and a token outside the largest 64 can
    /// be made the choice (a request with a bias is drawn from the row with the bias put on it: `Proc`).
    #[test]
    fn a_logit_bias_is_added_to_the_row_before_anything_else() {
        let mut row = logits();
        row.extend(std::iter::repeat_n(-5.0f32, 300)); // 310 tokens, the first three of them the likely ones
        let biased = |bias: Vec<(u32, f32)>, t: f32, seed: u64| Sampler::new(Params { temperature: t, seed, logit_bias: bias, ..Params::default() }, &[]);
        // greedy: a bias of 100 on a token that is last of the row makes it the choice; one of -100 on the largest takes it out
        assert_eq!(biased(vec![(309, 100.0)], 0.0, 0).sample(&row), 309);
        assert_eq!(biased(vec![(3, -100.0)], 0.0, 0).sample(&row), 1);
        assert_eq!(biased(vec![], 0.0, 0).sample(&row), 3);
        // a bias too small to overtake does not change the greedy choice; the gap 4.0 - 3.0 is closed by 1.5 on the second
        assert_eq!(biased(vec![(1, 0.5)], 0.0, 0).sample(&row), 3);
        assert_eq!(biased(vec![(1, 1.5)], 0.0, 0).sample(&row), 1);
        // drawn: +100 on one token is every draw, -100 on the likeliest never one of them
        let mut s = biased(vec![(250, 100.0)], 1.0, 5);
        assert!((0..50).all(|_| s.sample(&row) == 250));
        let mut s = biased(vec![(3, -100.0)], 3.0, 6);
        assert!((0..300).all(|_| s.sample(&row) != 3));
        // the same bias gives the same draws as no bias on a row that holds the bias in it (bias added by hand): the order of the steps is bias, then temperature
        let mut hand = row.clone();
        hand[7] += 2.0;
        let (mut a, mut b) = (biased(vec![(7, 2.0)], 1.3, 11), biased(vec![], 1.3, 11));
        assert!((0..100).all(|_| a.sample(&row) == b.sample(&hand)));
        // a token the row has no place for is ignored (the server refuses it before)
        assert_eq!(biased(vec![(100_000, 50.0)], 0.0, 0).sample(&row), 3);
        // the penalties come after the bias: a presence penalty acts on what the bias made the choice
        let mut p = Sampler::new(Params { temperature: 0.0, presence_penalty: 1.5, logit_bias: vec![(250, 100.0)], ..Params::default() }, &[]);
        assert_eq!(p.sample(&row), 250);
        assert_eq!(p.sample(&row), 250, "100 is more than a penalty of 1.5");
        // the GPU's candidates are of the row with the bias on it
        assert!(biased(vec![(250, 100.0)], 1.0, 1).proc().is_some() && biased(vec![], 1.0, 1).proc().is_none());
        let mut s = biased(vec![(250, 100.0)], 1.0, 1);
        let w = s.proc().unwrap().resolve(row.len()).apply(&row);
        assert_eq!(s.sample_view(&mut emu::Row { row: &w, asks: 0 }), Some(250));
        assert!(!Params { logit_bias: vec![(1, 1.0)], ..Params::default() }.greedy());
        assert!(Params::default().greedy());
    }

    /// The log-probabilities of a step as vLLM's default (`raw_logprobs`) gives them: the log-softmax of the row's raw logits (before the bias, the penalties, the temperature, top-k and top-p),
    /// for the token chosen and for the `k` likeliest, against a plain softmax; the GPU's candidates give the same list as the row.
    #[test]
    fn logprobs_are_the_log_softmax_of_the_raw_row() {
        let mut next = xorshift(0x9E37_79B9_7F4A_7C15);
        // a row of bf16 values with many equal ones (as the model's are), 5,000 tokens
        let row: Vec<f32> = (0..5000).map(|_| f32::from_bits((((next() % 6000) as f32 / 300.0 - 8.0).to_bits()) & 0xFFFF_0000)).collect();
        // the reference: the plain softmax in f64, tokens by probability (equal ones by the smaller token)
        let m = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let z: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
        let want = |t: u32| row[t as usize] as f64 - m - z.ln();
        let mut by: Vec<u32> = (0..row.len() as u32).collect();
        by.sort_by(|&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap().then(a.cmp(&b)));
        let cands: Vec<(u32, f32)> = by[..64].iter().map(|&i| (i, row[i as usize])).collect();
        for (token, k) in [(by[0], 0usize), (by[3], 5), (4999, 20), (17, 1), (by[0], 20)] {
            let lp = logprobs_of(&row, token, k, None);
            assert_eq!(lp.token, token);
            assert!((lp.logprob - want(token)).abs() < 1e-9, "token {token}: {} against {}", lp.logprob, want(token));
            assert_eq!(lp.top.iter().map(|t| t.0).collect::<Vec<_>>(), by[..k].to_vec(), "the {k} likeliest");
            for &(t, l) in &lp.top {
                assert!((l - want(t)).abs() < 1e-9);
            }
            assert!(lp.top.windows(2).all(|w| w[0].1 >= w[1].1));
            // the candidates of the GPU: the same list and the same numbers, for every k they cover
            assert_eq!(logprobs_of(&row, token, k, Some(&cands)), lp);
        }
        // more than the candidates cover: read from the row
        assert_eq!(logprobs_of(&row, by[1], 70, Some(&cands)).top.iter().map(|t| t.0).collect::<Vec<_>>(), by[..70].to_vec());
        // the probabilities of all tokens add to one
        assert!((by.iter().map(|&t| want(t).exp()).sum::<f64>() - 1.0).abs() < 1e-9);
        // a one-hot row: the chosen token is certain, the others are far below (and a token the row has no place for has no log-probability)
        let mut hot = vec![-30.0f32; 8];
        hot[2] = 30.0;
        let lp = logprobs_of(&hot, 2, 2, None);
        assert!(lp.logprob.abs() < 1e-12 && lp.top[1].1 < -59.0);
        assert_eq!(logprobs_of(&hot, 8, 0, None).logprob, f64::NEG_INFINITY);
    }

    /// vLLM's allowed_token_ids and min_tokens: the row is cut to the allowed tokens (greedy and drawn, a bias cannot bring one back), and the end tokens are out until the tokens asked for
    /// have been chosen; both are put on the row a GPU's candidates are made of (`Proc`), until the least length is reached.
    #[test]
    fn allowed_tokens_and_a_least_length() {
        let mut row = logits();
        row.extend(std::iter::repeat_n(-5.0f32, 20));
        // allowed: only those; the largest of them, whatever the row says; drawn, nothing else
        let mut s = Sampler::new(Params { temperature: 0.0, allowed: vec![7, 1], ..Params::default() }, &[]);
        assert_eq!(s.sample(&row), 1);
        let mut s = Sampler::new(Params { temperature: 2.0, allowed: vec![7, 12], seed: 3, ..Params::default() }, &[]);
        assert!((0..200).all(|_| matches!(s.sample(&row), 7 | 12)));
        let mut s = Sampler::new(Params { temperature: 0.0, allowed: vec![7], logit_bias: vec![(3, 100.0)], ..Params::default() }, &[]);
        assert_eq!(s.sample(&row), 7, "a bias does not bring back a token that is not allowed");
        assert!(!Params { allowed: vec![1], ..Params::default() }.greedy() && !Params { min_tokens: 1, ..Params::default() }.greedy());
        assert!(Sampler::new(Params { temperature: 1.0, top_k: 3, allowed: vec![1], ..Params::default() }, &[]).proc().is_some());
        // min_tokens: the end token (3, the largest) is out for the first two choices and back after
        let mut s = Sampler::new(Params { temperature: 0.0, min_tokens: 2, ..Params::default() }, &[]);
        s.set_suppress(vec![3]);
        assert_eq!((s.sample(&row), s.sample(&row), s.sample(&row)), (1, 1, 3));
        let mut s = Sampler::new(Params { temperature: 1.0, top_k: 3, min_tokens: 1, seed: 2, ..Params::default() }, &[]);
        s.set_suppress(vec![3]);
        assert!(s.proc().is_some(), "a token taken out: the candidates of the row with it out");
        assert_ne!(s.sample(&row), 3);
        assert!(s.proc().is_none(), "and after the least length, the model's own");
    }

    /// check_draft's tokens are distributed as sample's, whatever the draft (one the draw favours, one it seldom gives, one its cuts leave out), with top-k, top-p and a temperature.
    #[test]
    fn check_draft_keeps_the_distribution() {
        let row = vec![2.0f32, 1.5, 0.3, -1.0, 0.9, 1.4, -2.0, 0.0];
        let p = Params { temperature: 0.8, top_k: 6, top_p: 0.95, seed: 7, ..Params::default() };
        let n = 200_000;
        let freq = |f: &mut dyn FnMut(&mut Sampler) -> u32| {
            let mut s = Sampler::new(p.clone(), &[]);
            let mut c = vec![0f64; row.len()];
            for _ in 0..n {
                c[f(&mut s) as usize] += 1.0;
            }
            c.iter().map(|x| x / n as f64).collect::<Vec<_>>()
        };
        let want = freq(&mut |s| s.sample(&row));
        for draft in [0u32, 4, 3, 6] {
            let got = freq(&mut |s| s.check_draft(&row, draft).0);
            for (t, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 0.006, "draft {draft}: token {t} drawn {a:.4} of the time, where sample draws it {b:.4}");
            }
        }
        // greedy: the largest logit, the draft kept where it is that
        let mut g = Sampler::new(Params::default(), &[]);
        assert_eq!(g.check_draft(&row, 0), (0, true));
        assert_eq!(g.check_draft(&row, 5), (0, false));
    }
}
