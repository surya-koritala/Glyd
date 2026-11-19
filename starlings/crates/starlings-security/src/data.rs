//! Persistent GCP1 over enrolled TLS. No backend invocation or resource release.
use crate::{AuthorityEpoch, DataAdmission, Error, PeerStream, Result, transport};
use serde::{Deserialize, Serialize};
use starlings_protocol::{
    BoundedU32, DataHeader, DataPrefix, Digest, FixtureMode, NodeId, Version,
};
use std::{
    io::{Read, Write},
    time::Instant,
};
mod flow;
pub use flow::{FlowReceiver, FlowSender, ReceiveBudget, ReceivedTensors};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DataBinding {
    pub version: Version,
    pub epoch: AuthorityEpoch,
    pub plan: Digest,
    pub source: NodeId,
    pub destination: NodeId,
    pub source_stage: BoundedU32<0, 7>,
    pub destination_stage: BoundedU32<0, 7>,
}
pub struct DataFrame {
    header: DataHeader,
    payload: Vec<u8>,
}
impl DataFrame {
    pub fn header(&self) -> &DataHeader {
        &self.header
    }
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}
pub struct DataStream {
    peer: PeerStream,
    binding: DataBinding,
}
impl DataStream {
    fn read_header(io: &mut impl Read, header_budget: usize) -> Result<(DataHeader, usize)> {
        let mut prefix = [0; 16];
        io.read_exact(&mut prefix)?;
        let prefix = DataPrefix::parse(prefix)?;
        if prefix.header_len() > header_budget {
            return Err(Error::Capacity);
        }
        let mut bytes = vec![0; prefix.header_len()];
        io.read_exact(&mut bytes)?;
        Ok((
            DataHeader::decode(prefix, &bytes, FixtureMode::Reject)?,
            bytes.len(),
        ))
    }
    pub fn connect(peer: PeerStream, admission: &DataAdmission<'_>) -> Result<Self> {
        Self::negotiate(peer, admission, true)
    }
    pub fn accept(peer: PeerStream, admission: &DataAdmission<'_>) -> Result<Self> {
        Self::negotiate(peer, admission, false)
    }
    fn negotiate(
        mut peer: PeerStream,
        admission: &DataAdmission<'_>,
        connecting: bool,
    ) -> Result<Self> {
        let binding = admission.binding().clone();
        peer.admitted_frame(admission, Instant::now() + transport::TIMEOUT, |io| {
            admission.check(io, None, false, || ())?;
            if connecting {
                crate::send(io, &binding)?;
            }
            let remote: DataBinding = crate::receive(io)?;
            if remote != binding {
                return Err(Error::Denied);
            }
            if !connecting {
                crate::send(io, &binding)?;
            }
            admission.check(io, None, true, || ())
        })?;
        Ok(Self { peer, binding })
    }
    /// The authenticated peer, for the owner's fresh per-invocation admissions.
    pub fn peer(&self) -> &PeerStream {
        &self.peer
    }
    fn bound(&self, admission: &DataAdmission<'_>) -> Result<()> {
        if admission.binding() != &self.binding {
            return Err(Error::Denied);
        }
        Ok(())
    }
    /// The owner supplies fresh admission for each invocation, without reconnecting.
    pub fn send(
        &mut self,
        admission: &DataAdmission<'_>,
        header: &DataHeader,
        payload: &[u8],
    ) -> Result<()> {
        self.send_until(
            admission,
            header,
            payload,
            Instant::now() + transport::TIMEOUT,
        )
    }
    fn send_until(
        &mut self,
        admission: &DataAdmission<'_>,
        header: &DataHeader,
        payload: &[u8],
        deadline: Instant,
    ) -> Result<()> {
        let bound = self.bound(admission);
        self.peer.admitted_frame(admission, deadline, |io| {
            bound?;
            admission.check(io, Some(header), false, || ())?;
            let (prefix, bytes) = header.encode(FixtureMode::Reject)?;
            if prefix.payload_len() != payload.len() {
                return Err(Error::Invalid);
            }
            // One write: one checked call and one TLS flight, not three small segments.
            let mut frame = Vec::new();
            frame
                .try_reserve_exact(16 + bytes.len() + payload.len())
                .map_err(|_| Error::Capacity)?;
            frame.extend_from_slice(&prefix.bytes());
            frame.extend_from_slice(&bytes);
            frame.extend_from_slice(payload);
            io.write_all(&frame)?;
            io.flush()?;
            admission.check(io, Some(header), false, || ())
        })
    }
    /// Prefix, header and payload share one deadline. Authorization precedes payload
    /// allocation and the short enqueue callback runs under policy/runtime/fence locks.
    pub fn receive<T>(
        &mut self,
        admission: &DataAdmission<'_>,
        enqueue: impl FnOnce(DataFrame) -> T,
    ) -> Result<T> {
        self.receive_until(admission, Instant::now() + transport::TIMEOUT, enqueue)
    }
    fn receive_until<T>(
        &mut self,
        admission: &DataAdmission<'_>,
        deadline: Instant,
        enqueue: impl FnOnce(DataFrame) -> T,
    ) -> Result<T> {
        let bound = self.bound(admission);
        let deadline = deadline.min(Instant::now() + transport::TIMEOUT);
        self.peer.admitted_frame(admission, deadline, |io| {
            bound?;
            let frame = Self::read_frame(io, deadline, |peer, header| {
                admission.check(peer, header, true, || ())
            })?;
            admission.with_frame(io, frame, true, |frame| {
                if Instant::now() >= deadline {
                    Err(Error::Expired)
                } else {
                    Ok(enqueue(frame))
                }
            })?
        })
    }
    fn read_frame<R: Read>(
        io: &mut R,
        deadline: Instant,
        admit: impl Fn(&R, Option<&DataHeader>) -> Result<()>,
    ) -> Result<DataFrame> {
        admit(io, None)?;
        let (header, _) = Self::read_header(io, starlings_protocol::DATA_HEADER_LIMIT)?;
        admit(io, Some(&header))?;
        if Instant::now() >= deadline {
            return Err(Error::Expired);
        }
        let payload_len = match &header {
            DataHeader::Tensor(h) => {
                usize::try_from(h.chunk_bytes.0).map_err(|_| Error::Capacity)?
            }
            DataHeader::Control(_) => 0,
        };
        let mut payload = vec![0; payload_len];
        io.read_exact(&mut payload)?;
        Ok(DataFrame { header, payload })
    }
}

#[cfg(test)]
mod tests;
