//! GLY-360: guesses for the next tokens from the request's own tokens (prompt and answer so far).
//! Answers often repeat a run of tokens seen before (a name, a code line, a quoted phrase): after the
//! latest earlier match of the last few tokens, the tokens that followed it are the guess.

/// The longest match tried first; shorter ones only when it finds nothing.
const MAX_N: usize = 3;

/// Up to `k` guesses for the tokens after `history`: what followed the latest earlier occurrence of
/// its last `n` tokens, for n from MAX_N down to 1. Empty when nothing matches.
// ponytail: a linear scan per step (~MAX_N x the history); index n-grams if contexts get long.
pub fn draft(history: &[u32], k: usize) -> Vec<u32> {
    for n in (1..=MAX_N).rev() {
        let Some(last) = history.len().checked_sub(n) else {
            continue;
        };
        let tail = &history[last..];
        // An earlier occurrence ends before the tail does, so at least one token follows it.
        if let Some(at) = (0..last).rev().find(|&i| &history[i..i + n] == tail) {
            let from = at + n;
            return history[from..(from + k).min(history.len())].to_vec();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::draft;
    #[test]
    fn guesses_what_followed_the_latest_longest_match() {
        // ... 1 2 3 [4 5 6] ... 1 2 3 -> 4 5 6
        assert_eq!(draft(&[1, 2, 3, 4, 5, 6, 9, 1, 2, 3], 3), [4, 5, 6]);
        // Only k guesses, and only tokens that exist.
        assert_eq!(draft(&[1, 2, 3, 4, 5, 6, 9, 1, 2, 3], 2), [4, 5]);
        assert_eq!(draft(&[7, 1, 2, 3, 4, 1, 2, 3], 8), [4, 1, 2, 3]);
        // The latest match wins over an earlier one.
        assert_eq!(draft(&[2, 3, 5, 2, 3, 6, 2, 3], 1), [6]);
        // A longer match wins over a later shorter one: [1 2 3] -> 4, not [2 3] -> 8 or [3] -> 9.
        assert_eq!(draft(&[1, 2, 3, 4, 2, 3, 8, 3, 9, 1, 2, 3], 1), [4]);
        // Falls back to shorter tails.
        assert_eq!(draft(&[5, 6, 7, 6], 2), [7, 6]);
        // Nothing seen before, too short, or no room.
        assert!(draft(&[1, 2, 3], 4).is_empty());
        assert!(draft(&[], 4).is_empty());
        assert!(draft(&[1, 1], 0).is_empty());
        assert_eq!(draft(&[1, 1], 4), [1]);
    }
}
