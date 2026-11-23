//! The scheduler's pages (`Pager`): the blocks of 16 tokens lent to sequences, the full ones kept by their names (`block_hash`) for a later prompt that begins with
//! the same tokens, and what a step lost or made heavy (`Lost`). The dense Qwen3 engine's 12-bit cache, which this file held, went with that engine (R3c, #139): the
//! description executor's 12-bit pages are src/dynamic/kvpages.rs.

/// What the cache lost or made heavy in the steps since it was last asked: blocks with a void layer-page (what they hold is not what was written), blocks with a raw layer-page (held exactly, in the scarce second
/// extra memory: not to be kept for a later prompt), and whether more blocks were named than the status's rings hold (then any block may be lost).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lost {
    pub void: Vec<i32>,
    pub raw: Vec<i32>,
    pub overflow: bool,
}

impl Lost {
    pub fn is_empty(&self) -> bool {
        self.void.is_empty() && self.raw.is_empty() && !self.overflow
    }

    pub fn merge(&mut self, o: Lost) {
        self.void.extend(o.void);
        self.raw.extend(o.raw);
        self.overflow |= o.overflow;
    }
}

/// What a request is told where it cannot be held even when it runs alone.
pub const ALONE_MESSAGE: &str = "the lossless KV cache cannot hold this request, even alone: send a shorter text";

/// Tokens in a page.
pub const PAGE: usize = 16;

/// A cached page's place in a chain: the name of its tokens after those before it, the name before, and its tokens (to tell a collision).
struct Entry {
    key: u128,
    prev: u128,
    toks: [u32; PAGE],
}

/// A full block's name (R4c; docs/design/runtime.md "Content-addressed blocks and the radix index"): the first 16 bytes, read little-endian, of the SHA-256 of the name of the
/// block before it (16 bytes little-endian; for a sequence's first block its scope's root, `block_root`) and the block's token ids (4 bytes little-endian each). A block
/// has this one name whatever holds its keys and values (this cache's 12-bit pages, the description executor's blocks) and whoever is told of it, so the function is part of
/// what others read: changing it breaks them. A name says what tokens came before as well as the block's own: two blocks of the same name hold the same keys and values (as
/// far as two runs of the same tokens do), and a prompt's names are found one after another until the first that is not there, which is the longest start that is held.
pub fn block_hash(parent: u128, toks: &[u32]) -> u128 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(parent.to_le_bytes());
    for t in toks {
        h.update(t.to_le_bytes());
    }
    let d: [u8; 32] = h.finalize().into();
    u128::from_le_bytes(d[..16].try_into().expect("16 bytes"))
}

/// The name a scope's chains begin with: `block_hash`'s construction over the label "glyd-kv-1" and the scope's key (8 bytes little-endian; a request's `cache_salt` as the
/// server keys it, 0 for none), so no block of one scope has a name of another's. What else must keep two requests' blocks apart (an adapter, an image's bytes) is to be
/// hashed in here, or with the block it is in.
pub fn block_root(scope: u64) -> u128 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"glyd-kv-1");
    h.update(scope.to_le_bytes());
    let d: [u8; 32] = h.finalize().into();
    u128::from_le_bytes(d[..16].try_into().expect("16 bytes"))
}

/// The names of the full blocks `tokens` begins with, in the scope whose chains begin at `root` (`block_root`).
pub fn block_names(root: u128, tokens: &[u32]) -> Vec<u128> {
    let mut prev = root;
    tokens
        .chunks_exact(PAGE)
        .map(|c| {
            prev = block_hash(prev, c);
            prev
        })
        .collect()
}

/// The pages of the cache lent to sequences: a free list, the number of holders of every page, and the full pages of sequences that have ended kept by what their tokens were (a chain of keys), so a
/// prompt that begins with the same tokens finds them. Of the cached pages no one holds, the oldest is the first taken back, a chain's last page before its first.
/// A page the cache found lost (void) or heavy (a raw layer-page, held in the scarce second extra memory) is out of the cache and is not kept for a later prompt: it is cleared as soon as no one holds it.
pub struct Pager {
    free: Vec<i32>,
    refs: Vec<u32>,
    entry: Vec<Option<Entry>>,
    heavy: Vec<bool>,
    cache: std::collections::HashMap<u128, i32>,
    idle: std::collections::BTreeMap<u64, i32>,
    age: Vec<u64>,
    tick: u64,
}

impl Pager {
    pub fn new(pages: usize) -> Pager {
        Pager {
            free: (0..pages as i32).rev().collect(),
            refs: vec![0; pages],
            entry: (0..pages).map(|_| None).collect(),
            heavy: vec![false; pages],
            cache: Default::default(),
            idle: Default::default(),
            age: vec![0; pages],
            tick: 0,
        }
    }

    pub fn pages(&self) -> usize {
        self.refs.len()
    }

    /// Pages that can be had now: the free and the cached that no one holds.
    pub fn available(&self) -> usize {
        self.free.len() + self.idle.len()
    }

    /// A page for a new holder (the free first, else the oldest cached that no one holds), None where there is none.
    pub fn alloc(&mut self) -> Option<i32> {
        let p = match self.free.pop() {
            Some(p) => p,
            None => {
                let (&age, &p) = self.idle.iter().next()?;
                self.idle.remove(&age);
                if let Some(e) = self.entry[p as usize].take() {
                    self.cache.remove(&e.key);
                }
                p
            }
        };
        self.refs[p as usize] = 1;
        self.heavy[p as usize] = false;
        Some(p)
    }

    /// The run of cached pages `tokens` begins with (at most `max` of them), each now held by the caller.
    pub fn lookup(&mut self, tokens: &[u32], max: usize) -> Vec<i32> {
        self.lookup_in(0, tokens, max)
    }

    /// `lookup` among the pages of the scope `from` (a request's `cache_salt`, 0 for none): the key its chains begin with, so a chain of one scope is never found by
    /// another's tokens (`publish` with `prev` the scope for a sequence's first page).
    #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
    pub fn lookup_in(&mut self, from: u128, tokens: &[u32], max: usize) -> Vec<i32> {
        let mut out = Vec::new();
        let mut prev = from;
        for chunk in tokens.chunks_exact(PAGE).take(max) {
            let key = block_hash(prev, chunk);
            let Some(&p) = self.cache.get(&key) else { break };
            match &self.entry[p as usize] {
                Some(e) if e.prev == prev && e.toks[..] == *chunk => {}
                _ => break,
            }
            if self.refs[p as usize] == 0 {
                self.idle.remove(&self.age[p as usize]);
            }
            self.refs[p as usize] += 1;
            out.push(p);
            prev = key;
        }
        out
    }

    /// A page written in full with `toks`, after the pages whose chain ends in `prev`: kept under its key (unless the same tokens are cached on another page, or the page is lost or heavy). Returns the key, for the next page's chain.
    pub fn publish(&mut self, page: i32, prev: u128, toks: &[u32]) -> u128 {
        let key = block_hash(prev, toks);
        if !self.heavy[page as usize] && !self.cache.contains_key(&key) && self.entry[page as usize].is_none() {
            let mut t = [0u32; PAGE];
            t.copy_from_slice(toks);
            self.entry[page as usize] = Some(Entry { key, prev, toks: t });
            self.cache.insert(key, page);
        }
        key
    }

    /// A holder gives its pages back: the cached ones that no one holds now wait to be taken back, the others (a last page not full, a page whose tokens another page holds, a lost or heavy page) are returned: the caller clears those on the GPU.
    pub fn release(&mut self, pages: &[i32]) -> Vec<i32> {
        let mut freed = Vec::new();
        for &p in pages.iter().rev() {
            let r = &mut self.refs[p as usize];
            debug_assert!(*r > 0, "page {p} released twice");
            *r = r.saturating_sub(1);
            if *r == 0 {
                if self.entry[p as usize].is_some() && !self.heavy[p as usize] {
                    self.tick += 1;
                    self.age[p as usize] = self.tick;
                    self.idle.insert(self.tick, p);
                } else {
                    if let Some(e) = self.entry[p as usize].take() {
                        self.cache.remove(&e.key);
                    }
                    self.heavy[p as usize] = false;
                    self.free.push(p);
                    freed.push(p);
                }
            }
        }
        freed
    }

    /// A page the cache found lost or heavy: out of the cache, so that no prompt finds it, and, where `heavy`, cleared as soon as no one holds it (not kept for a later prompt). True where no one held it and it was waiting
    /// in the cache: it is free now, and the caller clears it on the GPU.
    pub fn evict(&mut self, page: i32, heavy: bool) -> bool {
        let p = page as usize;
        if p >= self.refs.len() {
            return false;
        }
        if let Some(e) = self.entry[p].take() {
            self.cache.remove(&e.key);
        }
        if heavy {
            self.heavy[p] = true;
        }
        if self.refs[p] == 0 && self.idle.get(&self.age[p]) == Some(&page) {
            self.idle.remove(&self.age[p]);
            self.heavy[p] = false;
            self.free.push(page);
            return true;
        }
        false
    }

    /// Every cached page that no one holds taken out of the cache and made free: returned, for the caller to clear on the GPU.
    pub fn drop_cache(&mut self) -> Vec<i32> {
        let idle: Vec<i32> = self.idle.values().copied().collect();
        self.idle.clear();
        for &p in &idle {
            if let Some(e) = self.entry[p as usize].take() {
                self.cache.remove(&e.key);
            }
            self.heavy[p as usize] = false;
            self.free.push(p);
        }
        idle
    }

    /// Pages that are cached and that no one holds.
    pub fn idle_pages(&self) -> usize {
        self.idle.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scope's pages are its own: tokens another scope published are not found, by no scope's lookup but its own, though they are the same tokens.
    #[test]
    fn a_scopes_pages_are_found_by_that_scope_alone() {
        let mut g = Pager::new(8);
        let toks: Vec<u32> = (0..2 * PAGE as u32).collect();
        let scope = 0x1234_5678_9abc_def1;
        let (a, b) = (g.alloc().unwrap(), g.alloc().unwrap());
        let k = g.publish(a, scope, &toks[..PAGE]);
        g.publish(b, k, &toks[PAGE..]);
        assert!(g.lookup(&toks, 4).is_empty() && g.lookup_in(7, &toks, 4).is_empty());
        assert_eq!(g.lookup_in(scope, &toks, 4), vec![a, b]);
        // the same tokens with no scope: pages of their own
        let c = g.alloc().unwrap();
        g.publish(c, 0, &toks[..PAGE]);
        assert_eq!((g.lookup(&toks, 4), g.lookup_in(scope, &toks, 1)), (vec![c], vec![a]));
    }

    /// The pages lent out, kept and taken back: a run of cached pages is found by the tokens it was written for, shared pages go back when their last holder does, and the oldest idle page is taken first, a chain's end before its start.
    #[test]
    fn pager() {
        let mut g = Pager::new(6);
        assert_eq!((g.pages(), g.available()), (6, 6));
        let toks: Vec<u32> = (0..48).collect();
        // a sequence of three full pages and a partial one, published as it goes
        let held: Vec<i32> = (0..4).map(|_| g.alloc().unwrap()).collect();
        let mut prev = 0;
        for (i, c) in toks.chunks(PAGE).enumerate() {
            prev = g.publish(held[i], prev, c);
        }
        assert_eq!(g.available(), 2);
        // a second prompt that begins with the same 40 tokens (two full pages, then others) finds two pages and holds them with the first
        let mut other: Vec<u32> = toks[..40].to_vec();
        other.extend(100..110);
        let got = g.lookup(&other, 10);
        assert_eq!(got, held[..2].to_vec());
        // the first ends: its partial page goes back, the cached ones that the second still holds stay held, the third waits
        let freed = g.release(&held);
        assert_eq!(freed, vec![held[3]]);
        assert_eq!(g.available(), 2 + 1 + 1);
        // the second ends too: all three cached pages wait; the oldest idle is the chain's last
        assert!(g.release(&got).is_empty());
        assert_eq!(g.available(), 6);
        let p1 = g.alloc().unwrap();
        let p2 = g.alloc().unwrap();
        let p3 = g.alloc().unwrap();
        assert!(!held[..3].contains(&p1) && !held[..3].contains(&p2) && !held[..3].contains(&p3), "the free pages come first");
        // now the cached ones: the chain's last page first
        assert_eq!(g.alloc(), Some(held[2]));
        assert_eq!(g.lookup(&toks, 10), held[..2].to_vec());
        assert_eq!(g.alloc(), None, "the two cached pages that are held again are not for the taking");
        // the same tokens published on another page are not cached twice
        g.release(&held[..2]);
        let dup = g.release(&[p1]);
        assert_eq!(dup, vec![p1]);
        let q = g.alloc().unwrap();
        assert_eq!(q, p1);
        let k = g.publish(q, 0, &toks[..16]);
        assert_eq!(k, block_hash(0, &toks[..16]));
        assert_eq!(g.lookup(&toks, 10), vec![held[0], held[1]], "the first page of the run stays the cached one");
    }

    /// A page found lost or heavy is out of the cache and is never kept for a later prompt: a held page is cleared when its holder goes, an idle one at once, and every idle cached page can be dropped.
    #[test]
    fn lost_and_heavy_pages_are_not_kept() {
        let mut g = Pager::new(8);
        let toks: Vec<u32> = (0..64).collect();
        let held: Vec<i32> = (0..4).map(|_| g.alloc().unwrap()).collect();
        let mut prev = 0;
        for (i, c) in toks.chunks(PAGE).enumerate() {
            prev = g.publish(held[i], prev, c);
        }
        // the second page is found heavy while its sequence still holds it: out of the cache, still held
        assert!(!g.evict(held[1], true));
        let found = g.lookup(&toks, 4);
        assert_eq!(found, vec![held[0]], "the run of cached pages ends where a page left the cache");
        g.release(&found);
        // a heavy page is not published again
        g.publish(held[1], block_hash(0, &toks[..16]), &toks[16..32]);
        assert_eq!(g.lookup(&toks, 4), vec![held[0]]);
        g.release(&[held[0]]);
        // its holder gives it back: it goes back to be cleared, the others wait in the cache
        let freed = g.release(&held);
        assert_eq!(freed, vec![held[1]]);
        assert_eq!((g.idle_pages(), g.available()), (3, 8));
        // an idle cached page found lost is free at once, and returned for clearing
        assert!(g.evict(held[2], true));
        assert_eq!((g.idle_pages(), g.available()), (2, 8));
        let found = g.lookup(&toks, 4);
        assert_eq!(found, vec![held[0]]);
        g.release(&found);
        // dropping the cache frees every idle cached page
        let mut dropped = g.drop_cache();
        dropped.sort_unstable();
        let mut want = vec![held[0], held[3]];
        want.sort_unstable();
        assert_eq!(dropped, want);
        assert_eq!((g.idle_pages(), g.available()), (0, 8));
        assert!(g.lookup(&toks, 4).is_empty());
        // a page that is free again is a page like any other: published and found
        let p = g.alloc().unwrap();
        g.publish(p, 0, &toks[..16]);
        g.release(&[p]);
        assert_eq!(g.lookup(&toks, 4), vec![p]);
    }
}
