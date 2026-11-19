//! Directed placement inputs only. No model selection or uncalibrated speed prediction.
use crate::CoreError;
use serde::{Deserialize, Serialize};
use starlings_protocol::{Decimal, FixtureMode, NodeId, RequestId, Version};
use std::collections::{BTreeMap, BTreeSet};

pub const LINK_TTL_MS: u64 = 30_000;
const MAX_LINKS: usize = 128;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkInput {
    pub version: Version,
    pub fixture: bool,
    pub sample_id: RequestId,
    pub authority_epoch: Decimal,
    pub authority_incarnation: RequestId,
    pub source: NodeId,
    pub destination: NodeId,
    pub payload_bytes: u32,
    pub elapsed_us: Decimal,
    pub acknowledged_bytes_per_second: Decimal,
    pub rtt_mean_us: Option<u32>,
    pub rtt_variance_us2: Option<Decimal>,
    pub inter_ping_gap_ms: u32,
    pub timing_basis: String,
    pub age_ms: u32,
}
impl LinkInput {
    pub fn validate(&self, mode: FixtureMode) -> Result<(), CoreError> {
        if (self.fixture && mode != FixtureMode::Simulation)
            || self.authority_epoch.0 == 0
            || self.source == self.destination
            || !(4096..=262_144).contains(&self.payload_bytes)
            || !self.payload_bytes.is_multiple_of(4096)
            || self.elapsed_us.0 == 0
            || self.elapsed_us.0 >= 5_000_000
            || self.acknowledged_bytes_per_second.0 == 0
            || u64::from(self.payload_bytes) * 1_000_000 / self.elapsed_us.0
                != self.acknowledged_bytes_per_second.0
            || self.rtt_mean_us.is_some_and(|n| n == 0 || n >= 5_000_000)
            || self.rtt_mean_us.is_some() != self.rtt_variance_us2.is_some()
            || self.inter_ping_gap_ms > 100
            || u64::from(self.age_ms) >= LINK_TTL_MS
            || !matches!(
                self.timing_basis.as_str(),
                "sender-start-to-ack-received"
                    | "receiver-start-to-ack-sent"
                    | "start-request-to-receive-and-ack-sent"
            )
        {
            return Err(CoreError::Invalid);
        }
        Ok(())
    }
}
/// Explicit synthetic replay only; live inputs must come from a current authenticated view.
/// Receiver owns original capture age. Exact replay never changes its timestamp.
#[derive(Default)]
pub struct LinkFixtureReplay {
    links: BTreeMap<(NodeId, NodeId), (LinkInput, u64)>,
    seen: BTreeSet<(NodeId, NodeId, RequestId)>,
    now: u64,
}
impl LinkFixtureReplay {
    fn clock(&mut self, now: u64) -> Result<(), CoreError> {
        if now < self.now {
            return Err(CoreError::Clock);
        }
        self.now = now;
        Ok(())
    }
    pub fn observe(&mut self, input: LinkInput, now: u64) -> Result<bool, CoreError> {
        input.validate(FixtureMode::Simulation)?;
        if !input.fixture {
            return Err(CoreError::Invalid);
        }
        self.clock(now)?;
        let pair = (input.source, input.destination);
        let sample = (input.source, input.destination, input.sample_id);
        if let Some((old, received)) = self.links.get_mut(&pair)
            && old.sample_id == input.sample_id
        {
            let mut measurements = input.clone();
            measurements.age_ms = old.age_ms;
            if old != &measurements {
                return Err(CoreError::Conflict);
            }
            // New transport delay may only shorten eligibility; receiver time never changes.
            let elapsed = now.checked_sub(*received).ok_or(CoreError::Clock)?;
            old.age_ms = old
                .age_ms
                .max(u64::from(input.age_ms).saturating_sub(elapsed) as u32);
            return Ok(false);
        }
        if self.seen.contains(&sample) {
            return Err(CoreError::ReplayExpired);
        }
        if self.seen.len() >= 256 {
            return Err(CoreError::Limit);
        }
        if self.links.len() >= MAX_LINKS && !self.links.contains_key(&pair) {
            return Err(CoreError::Limit);
        }
        self.seen.insert(sample);
        self.links.insert(pair, (input, now));
        Ok(true)
    }
    pub fn current(
        &mut self,
        source: NodeId,
        destination: NodeId,
        now: u64,
    ) -> Result<Option<LinkInput>, CoreError> {
        self.clock(now)?;
        let Some((input, received)) = self.links.get(&(source, destination)) else {
            return Ok(None);
        };
        let age = now
            .checked_sub(*received)
            .and_then(|n| n.checked_add(u64::from(input.age_ms)))
            .ok_or(CoreError::Clock)?;
        if age >= LINK_TTL_MS {
            return Ok(None);
        }
        let mut current = input.clone();
        current.age_ms = age as u32;
        Ok(Some(current))
    }
    /// Invalidate both directions, retaining IDs so replay cannot revive withdrawal.
    pub fn withdraw(&mut self, node: NodeId) {
        for ((source, destination), (input, _)) in &mut self.links {
            if *source == node || *destination == node {
                input.age_ms = LINK_TTL_MS as u32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_replay_preserves_asymmetry_congestion_age_and_withdrawal() {
        let [forward, reverse, congested]: [LinkInput; 3] =
            serde_json::from_str(include_str!("../../../examples/directed-links.json")).unwrap();
        let source = forward.source;
        let destination = forward.destination;
        let bytes = serde_json::to_vec(&forward).unwrap();
        let replay: LinkInput = serde_json::from_slice(&bytes).unwrap();
        assert!(replay.validate(FixtureMode::Reject).is_err());
        let mut inputs = LinkFixtureReplay::default();
        assert!(inputs.observe(replay, 0).unwrap());
        inputs.observe(reverse, 0).unwrap();
        assert_eq!(
            inputs
                .current(source, destination, 100)
                .unwrap()
                .unwrap()
                .acknowledged_bytes_per_second
                .0,
            inputs
                .current(destination, source, 100)
                .unwrap()
                .unwrap()
                .acknowledged_bytes_per_second
                .0
                * 4
        );
        assert!(!inputs.observe(forward.clone(), 200).unwrap());
        assert_eq!(
            inputs
                .current(source, destination, 200)
                .unwrap()
                .unwrap()
                .age_ms,
            200
        );
        let mut delayed = forward.clone();
        delayed.age_ms = 400;
        assert!(!inputs.observe(delayed, 200).unwrap());
        assert_eq!(
            inputs
                .current(source, destination, 200)
                .unwrap()
                .unwrap()
                .age_ms,
            400
        );
        let mut mutation = forward.clone();
        mutation.rtt_mean_us = Some(101);
        assert_eq!(inputs.observe(mutation, 200), Err(CoreError::Conflict));
        inputs.observe(congested.clone(), 300).unwrap();
        assert_eq!(inputs.observe(forward, 300), Err(CoreError::ReplayExpired));
        assert_eq!(
            inputs
                .current(source, destination, 300)
                .unwrap()
                .unwrap()
                .acknowledged_bytes_per_second
                .0,
            512_000
        );
        assert!(
            inputs
                .current(destination, source, LINK_TTL_MS)
                .unwrap()
                .is_none()
        );
        inputs.withdraw(destination);
        assert!(!inputs.observe(congested, LINK_TTL_MS).unwrap());
        assert!(
            inputs
                .current(source, destination, LINK_TTL_MS)
                .unwrap()
                .is_none()
        );
    }
}
