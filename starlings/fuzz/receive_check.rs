#[path = "../crates/starlings-security/src/data/flow/cursor.rs"]
mod cursor;
use cursor::ReceiveCursor;

/// At most eight 256-byte tensors and 64 actions; no sockets or payload contents.
/// Returns true only for a complete trace, not native execution or ownership.
pub fn check(data: &[u8]) -> bool {
    let Some(&count) = data.first() else {
        return false;
    };
    let count = usize::from(count % 8 + 1);
    let Some(lengths) = data.get(1..=count) else {
        return false;
    };
    let mut written: Vec<Vec<bool>> = lengths
        .iter()
        .map(|&length| vec![false; usize::from(length) + 1])
        .collect();
    let total: usize = written.iter().map(Vec::len).sum();
    let mut state = ReceiveCursor::default();
    let mut granted = 0u128;
    let mut spent = 0u128;
    for action in data[count + 1..].chunks_exact(18).take(64) {
        let offset = u64::from_le_bytes(action[1..9].try_into().unwrap());
        let bytes = u64::from_le_bytes(action[9..17].try_into().unwrap());
        let ended = action[17] & 1 != 0;
        match action[0] {
            0 => {
                let next = state.grant(bytes, total as u64);
                let valid = bytes != 0 && granted + u128::from(bytes) <= total as u128;
                assert_eq!(next.is_some(), valid);
                let Some(next) = next else { return false };
                state = next;
                granted += u128::from(bytes);
            }
            1 => {
                let Some(tensor) = written.get_mut(state.tensor) else {
                    return false;
                };
                let contiguous = tensor.iter().take_while(|&&byte| byte).count();
                let end = u128::from(offset) + u128::from(bytes);
                let valid = bytes != 0
                    && offset as u128 == contiguous as u128
                    && end <= tensor.len() as u128
                    && u128::from(bytes) <= granted - spent
                    && ended == (end == tensor.len() as u128);
                let next = state.chunk(offset, bytes, tensor.len(), ended);
                assert_eq!(next.is_some(), valid);
                let Some((range, next)) = next else {
                    return false;
                };
                assert_eq!(range.start, contiguous);
                assert_eq!(range.end as u128, end);
                assert!(tensor[range.clone()].iter().all(|&byte| !byte));
                tensor[range].fill(true);
                spent += u128::from(bytes);
                // Reading and authorizing bytes does not advance past an unsent receipt.
                assert_eq!(next.tensor, state.tensor);
                assert_eq!(next.offset as u128, end);
                state = next;
                state.acknowledge(ended);
            }
            2 => {
                let complete = written.iter().flatten().all(|&byte| byte) && granted == spent;
                assert_eq!(state.complete(count), complete);
                return complete;
            }
            _ => return false,
        }
        assert_eq!(u128::from(state.granted), granted);
        assert_eq!(u128::from(state.credit) + spent, granted);
        assert_eq!(
            state.tensor,
            written.iter().take_while(|t| t.iter().all(|&b| b)).count()
        );
        assert_eq!(
            state.complete(count),
            written.iter().flatten().all(|&byte| byte) && granted == spent
        );
    }
    state.complete(count)
}
