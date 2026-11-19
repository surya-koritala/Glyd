use std::ops::Range;

/// Byte accounting only; the caller retains authorization, I/O and ownership.
#[derive(Clone, Copy, Default)]
pub(super) struct ReceiveCursor {
    pub(super) tensor: usize,
    pub(super) offset: u64,
    pub(super) credit: u64,
    pub(super) granted: u64,
}

impl ReceiveCursor {
    pub(super) fn grant(self, bytes: u64, total: u64) -> Option<Self> {
        let granted = self.granted.checked_add(bytes)?;
        if bytes == 0 || granted > total {
            return None;
        }
        Some(Self {
            credit: self.credit.checked_add(bytes)?,
            granted,
            ..self
        })
    }

    pub(super) fn chunk(
        self,
        offset: u64,
        bytes: u64,
        length: usize,
        end_of_tensor: bool,
    ) -> Option<(Range<usize>, Self)> {
        if bytes == 0 || offset != self.offset || bytes > self.credit {
            return None;
        }
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(usize::try_from(bytes).ok()?)?;
        if end > length || end_of_tensor != (end == length) {
            return None;
        }
        Some((
            start..end,
            Self {
                offset: end as u64,
                credit: self.credit - bytes,
                ..self
            },
        ))
    }

    /// Advance only after the caller has successfully sent the receipt.
    pub(super) fn acknowledge(&mut self, end_of_tensor: bool) {
        if end_of_tensor {
            self.tensor += 1;
            self.offset = 0;
        }
    }

    pub(super) fn complete(self, tensors: usize) -> bool {
        self.tensor == tensors && self.offset == 0 && self.credit == 0
    }
}
