//! Sampling a next token from a step's logits, on the host (a step's logits are 300 KB, a copy of some 30 us beside a step of 46 ms): repetition, presence and frequency penalties over the
//! tokens so far, temperature, top-k, top-p and min-p, then a draw from the generator the seed starts. Temperature 0 is the largest logit (the first of equals, as torch.argmax chooses).

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
    /// Added to the logit of each token it names (the token and its bias), before the penalties and the temperature, as vLLM orders them: it can move a greedy choice, and a token the GPU's largest
    /// logits do not hold can become the choice, so a request with a bias is drawn from the whole row.
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

/// What a draw asks of the GPU before the step it draws from: the row's largest logits (`kern::TOPK` of them) and, for a top-p with no top-k, the sum over the whole row of exp((logit - largest) / temperature).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Need {
    pub temperature: f32,
    pub mass: bool,
}

/// What the GPU gives for a `Need`: the largest logits of a step's row (token and logit, largest first, equal logits by the smaller token) and the sum asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct Top {
    pub cands: Vec<(u32, f32)>,
    pub mass: Option<f64>,
}

/// Draws made by `Sampler::sample_with` from the whole row and from the GPU's largest logits alone.
pub static DRAWS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// How far the GPU's sum of a row's exponentials may be from the host's, relative (topkcheck holds the kernel to 2e-6); a top-p cut that is not the same for every sum within it is not made from the largest logits.
const MASS_ERROR: f64 = 1e-5;

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

    /// A uniform number in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 53) as f64
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
        Sampler { p, rng, seen, said: std::collections::HashMap::new(), n_said: 0, suppress: Vec::new(), mask: Vec::new(), gram }
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
            let (idx, probs) = self.dist(&mut l);
            let total: f64 = probs.iter().sum();
            let pd = idx.iter().position(|&t| t == draft).map_or(0.0, |i| probs[i] / total);
            if pd > 0.0 && self.rng.unit() < pd {
                draft
            } else {
                // (the rest, renormalized by walk: the draft's share taken out)
                let (idx, probs): (Vec<u32>, Vec<f64>) = idx.iter().zip(&probs).filter(|(t, _)| **t != draft).map(|(t, p)| (*t, *p)).unzip();
                if idx.is_empty() {
                    draft
                } else {
                    self.walk(&idx, &probs)
                }
            }
        };
        self.note(tok);
        (tok, tok == draft)
    }

    /// The token chosen, held by the answer (the penalties' counts).
    fn note(&mut self, tok: u32) {
        if let Some(g) = &mut self.gram {
            g.consume(tok);
        }
        self.n_said += 1;
        *self.said.entry(tok).or_insert(0) += 1;
        if self.p.repetition_penalty != 1.0 {
            *self.seen.entry(tok).or_insert(0) += 1;
        }
    }

    /// `logits` after the bias, the allowed tokens, the suppressed ones and the penalties: what the temperature and the cuts then see.
    fn processed(&mut self, logits: &[f32]) -> Vec<f32> {
        let mut l: Vec<f32> = logits.to_vec();
        for &(t, b) in &self.p.logit_bias {
            if let Some(x) = l.get_mut(t as usize) {
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

    /// What a draw from the row's `n` largest logits asks of the GPU, or None where it needs the whole row. It looks at them alone with a temperature above 0, no penalties, no bias, no allowed tokens and none taken out, and
    /// either a top-k of at most `n` or, with no top-k, a top-p below 1 and no min-p (the cut is then of the row's probabilities: their sum is asked for).
    fn need(&self, n: usize) -> Option<Need> {
        let p = &self.p;
        if n == 0
            || p.temperature <= 0.0
            || p.repetition_penalty != 1.0
            || p.presence_penalty != 0.0
            || p.frequency_penalty != 0.0
            || !p.logit_bias.is_empty()
            || !p.allowed.is_empty()
            || self.gram.is_some()
            || self.suppressing()
        {
            return None;
        }
        match p.top_k {
            k if (1..=n).contains(&k) => Some(Need { temperature: p.temperature, mass: false }),
            0 if p.top_p < 1.0 && p.min_p == 0.0 => Some(Need { temperature: p.temperature, mass: true }),
            _ => None,
        }
    }

    /// What the next draw asks of a GPU that makes `kern::TOPK` candidates (None: it reads the whole row).
    pub fn top_need(&self) -> Option<Need> {
        self.need(crate::kern::TOPK)
    }

    /// The next token from the largest logits of the step's row alone (`cands`: token and logit, largest first, equal logits by the smaller token: what the GPU returns, with `mass`, the sum over the whole row of
    /// exp((logit - largest) / temperature), where a top-p with no top-k is asked), where that is all the draw looks at (`need`). The same token `sample` gives from the whole row. None where it is not (the row's logits are
    /// then to be had and `sample` used), the generator untouched.
    pub fn sample_top(&mut self, cands: &[(u32, f32)], mass: Option<f64>) -> Option<u32> {
        let need = self.need(cands.len())?;
        let tok = if need.mass {
            self.draw_mass(cands, mass?)?
        } else {
            let k = self.p.top_k;
            self.finish_draw(cands[..k].iter().map(|c| c.0).collect(), cands[..k].iter().map(|c| c.1).collect())
        };
        self.n_said += 1;
        *self.said.entry(tok).or_insert(0) += 1;
        Some(tok)
    }

    /// The next token for an engine that makes the row's largest logits on the GPU (`top`, as `top_need` asked): drawn from them where that is all the draw looks at (`sample_top`), else from the whole row, which `row`
    /// reads then and not before. The same token either way.
    pub fn sample_with(&mut self, top: Option<&Top>, row: impl FnOnce() -> crate::ffi::Result<Vec<f32>>) -> crate::ffi::Result<u32> {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(tok) = top.and_then(|t| self.sample_top(&t.cands, t.mass)) {
            DRAWS[1].fetch_add(1, Relaxed);
            return Ok(tok);
        }
        DRAWS[0].fetch_add(1, Relaxed);
        Ok(self.sample(&row()?))
    }

    /// A draw with a top-p and no top-k from the largest logits `cands` and `mass`, the sum over the row of their exponentials: the token the whole row gives, or None where the cut is not the same for every sum within
    /// MASS_ERROR of `mass`, that is where it lies past the candidates or too near a cumulative probability to tell (and where `mass` is no sum that holds the candidates').
    fn draw_mass(&mut self, cands: &[(u32, f32)], mass: f64) -> Option<u32> {
        let max = cands[0].1;
        let e: Vec<f64> = cands.iter().map(|c| (((c.1 - max) / self.p.temperature) as f64).exp()).collect();
        if !mass.is_finite() || mass < e.iter().sum::<f64>() * (1.0 - MASS_ERROR) {
            return None;
        }
        // the tokens kept, as `finish_draw` keeps them: up to the first whose cumulative probability reaches top-p
        let cut = |z: f64| {
            let mut acc = 0.0;
            e.iter().position(|&x| {
                acc += x / z;
                acc >= self.p.top_p as f64
            })
        };
        let keep = cut(mass * (1.0 - MASS_ERROR)).filter(|&i| Some(i) == cut(mass * (1.0 + MASS_ERROR)))? + 1;
        let probs: Vec<f64> = e[..keep].iter().map(|x| x / mass).collect();
        Some(self.walk(&cands[..keep].iter().map(|c| c.0).collect::<Vec<_>>(), &probs))
    }

    fn draw(&mut self, l: &mut [f32]) -> u32 {
        let (idx, probs) = self.dist(l);
        self.walk(&idx, &probs)
    }

    /// The draw's candidates and their probabilities (largest first; walk renormalizes them): top-k, temperature, min-p and top-p over the processed row.
    fn dist(&self, l: &mut [f32]) -> (Vec<u32>, Vec<f64>) {
        // the candidates by probability, largest first, cut by top-k, then min-p, then top-p
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

    /// A token drawn from the candidates `idx` (their logits `vals`, largest first): temperature, then min-p and top-p cuts, then the generator's draw.
    fn finish_draw(&mut self, idx: Vec<u32>, vals: Vec<f32>) -> u32 {
        let (idx, probs) = self.cut(idx, vals);
        self.walk(&idx, &probs)
    }

    /// The candidates `idx` (their logits `vals`, largest first) after the temperature and the min-p and top-p cuts, with their probabilities.
    fn cut(&self, mut idx: Vec<u32>, vals: Vec<f32>) -> (Vec<u32>, Vec<f64>) {
        let max = vals[0];
        let mut probs: Vec<f64> = vals.iter().map(|&v| (((v - max) / self.p.temperature) as f64).exp()).collect();
        let total: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= total;
        }
        if self.p.min_p > 0.0 {
            let floor = self.p.min_p as f64 * probs[0];
            let keep = probs.iter().take_while(|&&p| p >= floor).count().max(1);
            probs.truncate(keep);
            idx.truncate(keep);
        }
        if self.p.top_p < 1.0 {
            let (mut acc, mut keep) = (0.0, 0);
            for &p in &probs {
                acc += p;
                keep += 1;
                if acc >= self.p.top_p as f64 {
                    break;
                }
            }
            probs.truncate(keep.max(1));
            idx.truncate(keep.max(1));
        }
        (idx, probs)
    }

    /// The generator's draw among the tokens `idx` of probabilities `probs` (largest first): one number, walked down their sum.
    fn walk(&mut self, idx: &[u32], probs: &[f64]) -> u32 {
        let total: f64 = probs.iter().sum();
        let mut r = self.rng.unit() * total;
        for (i, &p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[idx.len() - 1]
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The candidates the GPU returns (the largest logits of a row, equal ones by the smaller token) give the token the whole row gives, draw for draw, for every top-k they cover, and say when they do not.
    #[test]
    fn the_largest_logits_alone_give_the_same_draws() {
        // a row of bf16 values with many equal ones
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let row: Vec<f32> = (0..5000).map(|_| f32::from_bits((((next() % 4000) as f32 / 400.0 - 5.0).to_bits()) & 0xFFFF_0000)).collect();
        let mut by: Vec<u32> = (0..row.len() as u32).collect();
        by.sort_by(|&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap().then(a.cmp(&b)));
        let cands: Vec<(u32, f32)> = by[..64].iter().map(|&i| (i, row[i as usize])).collect();
        for (k, t, tp) in [(20usize, 0.6f32, 0.95f32), (1, 1.0, 1.0), (64, 1.3, 0.8), (5, 0.7, 1.0)] {
            let p = Params { temperature: t, top_k: k, top_p: tp, seed: 9, ..Params::default() };
            let (mut a, mut b) = (Sampler::new(p.clone(), &[]), Sampler::new(p, &[]));
            for _ in 0..200 {
                assert_eq!(a.sample(&row), b.sample_top(&cands, None).unwrap(), "top-k {k}, temperature {t}, top-p {tp}");
            }
        }
        // where the candidates do not cover what the draw looks at: no top-k, more than the candidates, penalties, or no temperature
        for p in [
            Params { temperature: 1.0, ..Params::default() },
            Params { temperature: 1.0, top_k: 65, ..Params::default() },
            Params { temperature: 1.0, top_k: 20, repetition_penalty: 1.1, ..Params::default() },
            Params { temperature: 1.0, top_k: 20, presence_penalty: 0.5, ..Params::default() },
            Params { temperature: 0.0, top_k: 20, ..Params::default() },
        ] {
            assert_eq!(Sampler::new(p, &[]).sample_top(&cands, None), None);
        }
    }

    /// The GPU's answer for `row` as the kernels make it: the 64 largest (equal values by the smaller token) and the sum of the exponentials at temperature `t` as expsum_bf16 forms it (each exponential in fp32, eight of them added
    /// in fp32, the eights in fp64: another order than the host's), `off` further from the host's sum, relative.
    fn gpu_top(row: &[f32], t: f32, off: f64) -> Top {
        let mut by: Vec<u32> = (0..row.len() as u32).collect();
        by.sort_by(|&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap().then(a.cmp(&b)));
        let max = row[by[0] as usize];
        let mass: f64 = row.chunks(8).map(|c| c.iter().map(|&v| ((v - max) / t).exp()).sum::<f32>() as f64).sum();
        Top { cands: by[..64].iter().map(|&i| (i, row[i as usize])).collect(), mass: Some(mass * (1.0 + off)) }
    }

    /// Rows of 5,000 bf16 logits: a few large values on a low floor (a sure next token), thirty alike, a hundred alike (a doubtful one), and a floor alone (nearly flat).
    fn rows(n: usize) -> Vec<Vec<f32>> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let bf = |v: f32| f32::from_bits(v.to_bits() & 0xFFFF_0000);
        (0..n)
            .map(|i| {
                let mut row: Vec<f32> = (0..5000).map(|_| bf(-12.0 + (next() % 400) as f32 / 100.0)).collect();
                let (hot, level) = [(4, 8.0), (30, 4.0), (100, 3.0), (0, 0.0)][i % 4];
                for _ in 0..hot {
                    row[(next() % 5000) as usize] = bf(level + (next() % 300) as f32 / 100.0);
                }
                row
            })
            .collect()
    }

    /// What a lossy engine does with a step, on the host: ask `top_need`, draw by `sample_with` from the GPU's answer (here `gpu_top`, its sum `off` out), reading the row only where the draw needs it. The same tokens as the draws from the whole row,
    /// seed for seed, in steps over rows of every kind, whatever the sampling: the Nemotron 3 Super checkpoint's (temperature 1, top-p 0.95, no top-k), the Qwen3.5 ones (top-k 20), and the samplings the candidates cannot serve (the row is read
    /// every step: a top-k past them, no top-k and no top-p, a min-p with no top-k, penalties, a bias, allowed tokens, no temperature). Returns how many steps read the row.
    fn steps_like_an_engine(p: &Params, rows: &[Vec<f32>]) -> usize {
        let (mut whole, mut gpu) = (Sampler::new(p.clone(), &[7]), Sampler::new(p.clone(), &[7]));
        let mut reads = 0;
        for (i, row) in rows.iter().enumerate() {
            let top = gpu.top_need().map(|n| {
                let t = gpu_top(row, n.temperature, [0.0, 7e-7, -7e-7][i % 3]);
                Top { mass: t.mass.filter(|_| n.mass), ..t }
            });
            let got = gpu.sample_with(top.as_ref(), || {
                reads += 1;
                Ok(row.clone())
            });
            assert_eq!(got.unwrap(), whole.sample(row), "step {i} of {p:?}");
        }
        reads
    }

    #[test]
    fn the_gpus_largest_logits_give_the_draws_of_the_row() {
        let rows = rows(120);
        let n = rows.len();
        let base = Params { temperature: 1.0, seed: 17, ..Params::default() };
        // the checkpoints' own: the rows the 64 largest do not hold 95% of (a hundred alike, a flat one) are read, the others are not
        let reads = steps_like_an_engine(&Params { top_p: 0.95, ..base.clone() }, &rows);
        assert!(reads > 0 && reads < n, "Nemotron's sampling reads {reads} of {n} rows");
        let sure: Vec<Vec<f32>> = rows.iter().step_by(4).cloned().chain(rows.iter().skip(1).step_by(4).cloned()).collect();
        assert_eq!(steps_like_an_engine(&Params { top_p: 0.95, ..base.clone() }, &sure), 0, "a sure row is not read");
        assert_eq!(steps_like_an_engine(&Params { temperature: 0.6, top_p: 0.8, ..base.clone() }, &sure), 0);
        // a top-p that the hundred alike are reached by within 64 (half of the probability) needs no row either
        assert!(steps_like_an_engine(&Params { top_p: 0.5, ..base.clone() }, &rows) < reads);
        // a top-k within the candidates never reads the row, with or without top-p and min-p
        for p in [
            Params { top_k: 20, top_p: 0.95, temperature: 0.6, ..base.clone() },
            Params { top_k: 64, temperature: 1.3, ..base.clone() },
            Params { top_k: 1, ..base.clone() },
            Params { top_k: 5, min_p: 0.1, top_p: 0.9, ..base.clone() },
        ] {
            assert_eq!(steps_like_an_engine(&p, &rows), 0, "{p:?}");
        }
        // the samplings the candidates cannot serve read the row at every step
        for p in [
            Params { top_k: 65, ..base.clone() },
            base.clone(),
            Params { top_p: 0.95, min_p: 0.05, ..base.clone() },
            Params { top_k: 20, repetition_penalty: 1.1, ..base.clone() },
            Params { top_p: 0.95, presence_penalty: 0.5, ..base.clone() },
            Params { top_k: 20, frequency_penalty: 0.3, ..base.clone() },
            Params { top_k: 20, logit_bias: vec![(4000, 100.0)], ..base.clone() },
            Params { top_p: 0.95, allowed: vec![5, 6, 7, 4999], ..base.clone() },
            Params { temperature: 0.0, presence_penalty: 1.0, ..base.clone() },
        ] {
            assert_eq!(steps_like_an_engine(&p, &rows), n, "{p:?}");
        }
    }

    /// A top-p that a cumulative probability of the candidates comes within the GPU's error of is not drawn from them: the sum could put the cut a token to either side. Two tokens alike on a floor 40 below have 0.5 of the probability
    /// each less 1e-14, which the host's sum (the floor added to 2, a term at a time) loses, so a top-p of 0.5 keeps one token for the host and would keep both for a sum 1e-6 too high: the row is read, whatever the sum is within the
    /// error. A top-p of 0.3 keeps the first token for every sum within it, and is drawn from the candidates. A sum that is no sum of the row (missing, zero, not a number, less than the candidates') is not trusted either.
    #[test]
    fn a_top_p_too_near_a_cumulative_probability_is_drawn_from_the_row() {
        let mut row = vec![-40.0f32; 5000];
        row[10] = 0.0;
        row[20] = 0.0;
        let run = |top_p: f32, mass: &dyn Fn(f64) -> Option<f64>| {
            let p = Params { temperature: 1.0, top_p, seed: 3, ..Params::default() };
            let (mut a, mut b) = (Sampler::new(p.clone(), &[]), Sampler::new(p, &[]));
            let (mut reads, mut seen) = (0, std::collections::HashSet::new());
            for _ in 0..200 {
                let top = gpu_top(&row, 1.0, 0.0);
                let top = Top { mass: top.mass.and_then(mass), ..top };
                let tok = b.sample_with(Some(&top), || {
                    reads += 1;
                    Ok(row.clone())
                });
                assert_eq!(tok.as_ref().unwrap(), &a.sample(&row));
                seen.insert(tok.unwrap());
            }
            (reads, seen)
        };
        for off in [0.0, -1e-6, 1e-6, -9e-6] {
            let (reads, seen) = run(0.5, &|m| Some(m * (1.0 + off)));
            assert_eq!((reads, seen), (200, [10].into()), "off {off}");
        }
        let (reads, seen) = run(0.3, &|m| Some(m * (1.0 - 9e-6)));
        assert_eq!((reads, seen), (0, [10].into()), "the first token keeps 0.5 for any sum within the error");
        for bad in [None, Some(0.0), Some(f64::NAN), Some(f64::INFINITY), Some(0.5)] {
            let (reads, _) = run(0.3, &|_| bad);
            assert_eq!(reads, 200, "a sum of {bad:?} is not trusted");
        }
    }

    #[test]
    fn what_a_draw_asks_of_the_gpu() {
        let ask = |p: Params| Sampler::new(Params { temperature: 0.7, ..p }, &[]).top_need();
        let (k, m) = (Some(Need { temperature: 0.7, mass: false }), Some(Need { temperature: 0.7, mass: true }));
        assert_eq!((ask(Params { top_k: 20, ..Params::default() }), ask(Params { top_k: 64, top_p: 0.9, min_p: 0.1, ..Params::default() })), (k, k));
        assert_eq!((ask(Params { top_p: 0.95, ..Params::default() }), ask(Params { top_k: 0, top_p: 0.01, ..Params::default() })), (m, m));
        for none in [
            Params::default(),
            Params { top_k: 65, ..Params::default() },
            Params { top_p: 0.95, min_p: 0.05, ..Params::default() },
            Params { min_p: 0.05, ..Params::default() },
        ] {
            assert_eq!(ask(none.clone()), None, "{none:?}");
        }
        assert_eq!(Sampler::new(Params { temperature: 0.0, top_k: 20, ..Params::default() }, &[]).top_need(), None);
        assert_eq!(Sampler::new(Params { temperature: 0.0, ..Params::default() }, &[]).top_need(), None, "greedy asks for nothing");
        // a least length with tokens to take out: the row until it is reached
        let mut s = Sampler::new(Params { temperature: 1.0, top_k: 20, min_tokens: 1, ..Params::default() }, &[]);
        s.set_suppress(vec![3]);
        assert_eq!(s.top_need(), None);
        s.sample(&logits());
        assert!(s.top_need().is_some());
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
    /// be made the choice, which the GPU's candidates cannot show (a request with a bias is drawn from the whole row).
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
        // the candidates of the GPU cover a draw without a bias only
        let cands: Vec<(u32, f32)> = vec![(3, 4.0), (1, 3.0), (7, 2.0)];
        assert_eq!(biased(vec![(250, 100.0)], 1.0, 1).sample_top(&cands, None), None);
        assert!(Sampler::new(Params { temperature: 1.0, top_k: 3, ..Params::default() }, &[]).sample_top(&cands, None).is_some());
        assert!(!Params { logit_bias: vec![(1, 1.0)], ..Params::default() }.greedy());
        assert!(Params::default().greedy());
    }

    /// The log-probabilities of a step as vLLM's default (`raw_logprobs`) gives them: the log-softmax of the row's raw logits (before the bias, the penalties, the temperature, top-k and top-p),
    /// for the token chosen and for the `k` likeliest, against a plain softmax; the GPU's candidates give the same list as the row.
    #[test]
    fn logprobs_are_the_log_softmax_of_the_raw_row() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
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

    /// vLLM's allowed_token_ids and min_tokens: the row is cut to the allowed tokens (greedy and drawn, a bias cannot bring one back), and the end tokens are out until the tokens asked for have been chosen; neither
    /// is a draw from the GPU's candidates.
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
        assert_eq!(
            Sampler::new(Params { temperature: 1.0, top_k: 3, allowed: vec![1], ..Params::default() }, &[]).sample_top(&[(3, 4.0), (1, 3.0), (7, 2.0)], None),
            None
        );
        // min_tokens: the end token (3, the largest) is out for the first two choices and back after
        let mut s = Sampler::new(Params { temperature: 0.0, min_tokens: 2, ..Params::default() }, &[]);
        s.set_suppress(vec![3]);
        assert_eq!((s.sample(&row), s.sample(&row), s.sample(&row)), (1, 1, 3));
        let mut s = Sampler::new(Params { temperature: 1.0, top_k: 3, min_tokens: 1, seed: 2, ..Params::default() }, &[]);
        s.set_suppress(vec![3]);
        assert_eq!(s.sample_top(&[(3, 4.0), (1, 3.0), (7, 2.0)], None), None, "no draw from the candidates where a token is taken out");
        assert_ne!(s.sample(&row), 3);
        assert!(s.sample_top(&[(3, 4.0), (1, 3.0), (7, 2.0)], None).is_some(), "and after the least length, as any other");
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
