//! The level-3 match finder: zstd 1.5.5's
//! `ZSTD_compressBlock_doubleFast_noDict_generic` (`zstd_double_fast.c`)
//! step for step: two tables, one hashing eight bytes (the "long"
//! table, `hashLog` bits) and one hashing `minMatch` bytes (the
//! "small" table, `chainLog` bits); at each position the repcode is
//! tried one byte ahead, then a long match here, then a short match
//! here upgraded to a long match one byte ahead when there is one; the
//! step grows every 256 bytes without a match; four table entries are
//! filled after a match and the repcode chained right after it. Its
//! `extDict` variant, run by the streaming API after its input ring
//! wraps, is the older loop: one position at a time, accelerated by
//! the distance from the last match, with the dictionary segment's
//! rules on repcodes and backward extension and no repcode check at
//! the block start.

use super::fast::{count, hash, idx, pos, read32, read64, SeqStore, Window, HASH_READ_SIZE, SEARCH_STRENGTH};
use super::CParams;

/// `_match_stored`: the four complementary entries, then repcode 2
/// matches chained at `ip` (`ext` applies the dictionary segment's
/// rules; `prefix_start_index` and `dict_start_index` only matter then).
#[allow(clippy::too_many_arguments)]
fn after_match(input: &[u8], store: &mut SeqStore, long: &mut [u32], small: &mut [u32], p: &CParams, curr: u32, ip: &mut usize, anchor: &mut usize, end: usize, rep1: &mut u32, rep2: &mut u32, ext: Option<(u32, u32)>) {
    let (hbl, hbs, mls) = (p.hash_log, p.chain_log, p.min_match);
    let ilimit = end as isize - HASH_READ_SIZE as isize;
    if *ip as isize > ilimit {
        return;
    }
    let c2 = pos(curr) + 2;
    long[hash(input, c2, hbl, 8)] = curr + 2;
    long[hash(input, *ip - 2, hbl, 8)] = idx(*ip - 2);
    small[hash(input, c2, hbs, mls)] = curr + 2;
    small[hash(input, *ip - 1, hbs, mls)] = idx(*ip - 1);
    while *ip as isize <= ilimit {
        let current2 = idx(*ip);
        let rep_index2 = current2.wrapping_sub(*rep2);
        let valid = match ext {
            None => *rep2 > 0,
            Some((prefix_start_index, dict_start_index)) => (prefix_start_index - 1).wrapping_sub(rep_index2) >= 3 && *rep2 <= current2 - dict_start_index,
        };
        if !(valid && read32(input, pos(rep_index2)) == read32(input, *ip)) {
            break;
        }
        let rlen = count(input, *ip + 4, pos(rep_index2) + 4, end) + 4;
        std::mem::swap(rep1, rep2);
        small[hash(input, *ip, hbs, mls)] = current2;
        long[hash(input, *ip, hbl, 8)] = current2;
        store.store(&[], 1, rlen);
        *ip += rlen;
        *anchor = *ip;
    }
}

/// The block `input[start..end]` parsed into `SeqStore`, the tables
/// and the repcodes updated for the next block (`noDict`).
#[allow(clippy::too_many_arguments)]
pub fn compress_block(input: &[u8], start: usize, end: usize, long: &mut [u32], small: &mut [u32], p: &CParams, window: &Window, rep: &mut [u32; 3]) -> SeqStore {
    let mut store = SeqStore::new();
    let (hbl, hbs, mls) = (p.hash_log, p.chain_log, p.min_match);
    let step_incr = 1usize << SEARCH_STRENGTH;
    let end_index = idx(end);
    let prefix_lowest_index = window.lowest_prefix(end_index, p.window_log);
    let prefix_lowest = pos(prefix_lowest_index);
    let ilimit = end as isize - HASH_READ_SIZE as isize;
    let mut anchor = start;
    let mut ip = start;
    let (mut rep1, mut rep2) = (rep[0], rep[1]);
    let (mut saved1, mut saved2) = (0u32, 0u32);
    if ip == prefix_lowest {
        ip += 1;
    }
    {
        let current = idx(ip);
        let window_low = window.lowest_prefix(current, p.window_log);
        let max_rep = current - window_low;
        if rep2 > max_rep {
            saved2 = rep2;
            rep2 = 0;
        }
        if rep1 > max_rep {
            saved1 = rep1;
            rep1 = 0;
        }
    }
    // A match: its start, offset (0 for a repcode) and length.
    struct Found {
        ip: usize,
        offset: u32,
        len: usize,
    }
    loop {
        let mut step = 1usize;
        let mut next_step = ip + step_incr;
        let mut ip1 = ip + step;
        let mut curr = idx(ip);
        let mut hl1 = 0usize;
        let found: Option<Found> = 'search: {
            if ip1 as isize > ilimit {
                break 'search None;
            }
            let mut hl0 = hash(input, ip, hbl, 8);
            let mut idxl0 = long[hl0];
            loop {
                let hs0 = hash(input, ip, hbs, mls);
                let idxs0 = small[hs0];
                curr = idx(ip);
                long[hl0] = curr;
                small[hs0] = curr;
                // The repcode one byte ahead.
                if rep1 > 0 && read32(input, ip + 1 - rep1 as usize) == read32(input, ip + 1) {
                    let len = count(input, ip + 5, ip + 5 - rep1 as usize, end) + 4;
                    break 'search Some(Found { ip: ip + 1, offset: 0, len });
                }
                hl1 = hash(input, ip1, hbl, 8);
                // A long match here.
                if idxl0 > prefix_lowest_index && read64(input, pos(idxl0)) == read64(input, ip) {
                    let mut m = pos(idxl0);
                    let mut len = count(input, ip + 8, m + 8, end) + 8;
                    let offset = (ip - m) as u32;
                    let mut ip0 = ip;
                    while ip0 > anchor && m > prefix_lowest && input[ip0 - 1] == input[m - 1] {
                        ip0 -= 1;
                        m -= 1;
                        len += 1;
                    }
                    break 'search Some(Found { ip: ip0, offset, len });
                }
                let idxl1 = long[hl1];
                // A short match here, upgraded to a long one a byte ahead when there is one.
                if idxs0 > prefix_lowest_index && read32(input, pos(idxs0)) == read32(input, ip) {
                    let (mut ip0, mut m, mut len) = if idxl1 > prefix_lowest_index && read64(input, pos(idxl1)) == read64(input, ip1) {
                        let m = pos(idxl1);
                        (ip1, m, count(input, ip1 + 8, m + 8, end) + 8)
                    } else {
                        let m = pos(idxs0);
                        (ip, m, count(input, ip + 4, m + 4, end) + 4)
                    };
                    let offset = (ip0 - m) as u32;
                    while ip0 > anchor && m > prefix_lowest && input[ip0 - 1] == input[m - 1] {
                        ip0 -= 1;
                        m -= 1;
                        len += 1;
                    }
                    break 'search Some(Found { ip: ip0, offset, len });
                }
                if ip1 >= next_step {
                    step += 1;
                    next_step += step_incr;
                }
                ip = ip1;
                ip1 += step;
                hl0 = hl1;
                idxl0 = idxl1;
                if ip1 as isize > ilimit {
                    break 'search None;
                }
            }
        };
        let Some(f) = found else {
            let saved2 = if saved1 != 0 && rep1 != 0 { saved1 } else { saved2 };
            rep[0] = if rep1 != 0 { rep1 } else { saved1 };
            rep[1] = if rep2 != 0 { rep2 } else { saved2 };
            break;
        };
        if f.offset == 0 {
            store.store(&input[anchor..f.ip], 1, f.len);
        } else {
            rep2 = rep1;
            rep1 = f.offset;
            // The entry for ip1 is written only while ip1 stays below
            // where the search resumes after the match.
            if step < 4 {
                long[hl1] = idx(ip1);
            }
            store.store(&input[anchor..f.ip], f.offset + 3, f.len);
        }
        ip = f.ip + f.len;
        anchor = ip;
        after_match(input, &mut store, long, small, p, curr, &mut ip, &mut anchor, end, &mut rep1, &mut rep2, None);
    }
    store.lits.extend_from_slice(&input[anchor..end]);
    store
}

/// `ZSTD_compressBlock_doubleFast_extDict_generic`: the block parsed
/// with a dictionary segment below `window.dict_limit`.
#[allow(clippy::too_many_arguments)]
pub fn compress_block_ext(input: &[u8], start: usize, end: usize, long: &mut [u32], small: &mut [u32], p: &CParams, window: &Window, rep: &mut [u32; 3]) -> SeqStore {
    let end_index = idx(end);
    let dict_start_index = window.lowest_match(end_index, p.window_log);
    let prefix_start_index = window.dict_limit.max(dict_start_index);
    if prefix_start_index == dict_start_index {
        return compress_block(input, start, end, long, small, p, window, rep);
    }
    let mut store = SeqStore::new();
    let (hbl, hbs, mls) = (p.hash_log, p.chain_log, p.min_match);
    let prefix_start = pos(prefix_start_index);
    let dict_start = pos(dict_start_index);
    let ilimit = end as isize - HASH_READ_SIZE as isize;
    let low_match = |index: u32| if index < prefix_start_index { dict_start } else { prefix_start };
    let mut anchor = start;
    let mut ip = start;
    let (mut rep1, mut rep2) = (rep[0], rep[1]);
    while (ip as isize) < ilimit {
        let hs = hash(input, ip, hbs, mls);
        let match_index = small[hs];
        let hl = hash(input, ip, hbl, 8);
        let match_long_index = long[hl];
        let curr = idx(ip);
        let rep_index = (curr + 1).wrapping_sub(rep1);
        small[hs] = curr;
        long[hl] = curr;
        let len;
        if (prefix_start_index - 1).wrapping_sub(rep_index) >= 3 && rep1 <= curr + 1 - dict_start_index && read32(input, pos(rep_index)) == read32(input, ip + 1) {
            len = count(input, ip + 5, pos(rep_index) + 4, end) + 4;
            ip += 1;
            store.store(&input[anchor..ip], 1, len);
        } else if match_long_index > dict_start_index && read64(input, pos(match_long_index)) == read64(input, ip) {
            let mut m = pos(match_long_index);
            let low = low_match(match_long_index);
            let mut l = count(input, ip + 8, m + 8, end) + 8;
            let offset = curr - match_long_index;
            while ip > anchor && m > low && input[ip - 1] == input[m - 1] {
                ip -= 1;
                m -= 1;
                l += 1;
            }
            len = l;
            rep2 = rep1;
            rep1 = offset;
            store.store(&input[anchor..ip], offset + 3, len);
        } else if match_index > dict_start_index && read32(input, pos(match_index)) == read32(input, ip) {
            let h3 = hash(input, ip + 1, hbl, 8);
            let match_index3 = long[h3];
            long[h3] = curr + 1;
            let (mut m, low, mut l, offset) = if match_index3 > dict_start_index && read64(input, pos(match_index3)) == read64(input, ip + 1) {
                let m = pos(match_index3);
                let l = count(input, ip + 9, m + 8, end) + 8;
                ip += 1;
                (m, low_match(match_index3), l, curr + 1 - match_index3)
            } else {
                let m = pos(match_index);
                (m, low_match(match_index), count(input, ip + 4, m + 4, end) + 4, curr - match_index)
            };
            while ip > anchor && m > low && input[ip - 1] == input[m - 1] {
                ip -= 1;
                m -= 1;
                l += 1;
            }
            len = l;
            rep2 = rep1;
            rep1 = offset;
            store.store(&input[anchor..ip], offset + 3, len);
        } else {
            ip += ((ip - anchor) >> SEARCH_STRENGTH) + 1;
            continue;
        }
        ip += len;
        anchor = ip;
        after_match(input, &mut store, long, small, p, curr, &mut ip, &mut anchor, end, &mut rep1, &mut rep2, Some((prefix_start_index, dict_start_index)));
    }
    rep[0] = rep1;
    rep[1] = rep2;
    store.lits.extend_from_slice(&input[anchor..end]);
    store
}
