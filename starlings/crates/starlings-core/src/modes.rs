//! Offline tier-cost ranking, not native qualification or transactional admission.
use crate::{
    CoreError,
    planner::{self, Candidate, Family, PreviewInput, PreviewStage},
};
use serde::{Deserialize, Serialize};
use starlings_protocol::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigurationMode {
    Resident,
    ExpertOffload,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementMode {
    WholeRequest,
    ExpertOffload,
    LayerSplit,
}

/// Declared observations; file contents cannot establish trusted measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub basis: CostBasis,
    pub evidence_digest: Digest,
    pub captured_at_ms: Decimal,
    pub age_at_capture_ms: Decimal,
    pub valid_for_ms: BoundedU32<1, 86400000>,
}
impl Observation {
    fn fresh(&self, now: u64, fixture: bool) -> bool {
        (fixture || self.basis == CostBasis::Measured)
            && !zero(&self.evidence_digest)
            && now
                .checked_sub(self.captured_at_ms.0)
                .and_then(|n| n.checked_add(self.age_at_capture_ms.0))
                .is_some_and(|age| age < u64::from(self.valid_for_ms.get()))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierBandwidth {
    pub node: NodeId,
    pub profile_digest: Digest,
    pub pool_id: String,
    pub bytes_per_second: Decimal,
    pub observation: Observation,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkLatency {
    pub source: NodeId,
    pub destination: NodeId,
    pub link_evidence_digest: Digest,
    pub p50_us: Option<Decimal>,
    pub p99_us: Option<Decimal>,
    pub idle_wake_us: Option<Decimal>,
    pub observation: Observation,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum StorageTraffic {
    NotUsed,
    Unavailable,
    Available {
        bytes_per_second: Decimal,
        layer_bytes_per_token: Vec<Decimal>,
        first_stage_bytes_per_token: Decimal,
        last_stage_bytes_per_token: Decimal,
        per_stage_bytes_per_token: Decimal,
        observation: Observation,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageRead {
    pub node: NodeId,
    pub profile_digest: Digest,
    pub traffic: StorageTraffic,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeCosts {
    pub tiers: Vec<TierBandwidth>,
    pub storage: Vec<StorageRead>,
    pub links: Vec<LinkLatency>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub configuration: ConfigurationMode,
    pub placement: PreviewInput,
    pub costs: ModeCosts,
    /// Approval of content_hash((configuration, costs)), supplied by trusted local review.
    pub approved_costs_digest: Option<Digest>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkLimits {
    pub p50_us: Decimal,
    pub p99_us: Decimal,
    pub idle_wake_us: Decimal,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeInput {
    pub version: Version,
    pub fixture: bool,
    pub family: Family,
    pub variants: Vec<Variant>,
    pub link_limits: LinkLimits,
}
#[derive(Debug, Serialize)]
pub struct DecodeCost {
    pub memory_read_us: Decimal,
    pub storage_read_us: Decimal,
    pub local_copy_us: Decimal,
    pub link_rtt_us: Decimal,
    pub total_us: Decimal,
    pub predicted_tokens_per_second: f64,
    pub evidence: Vec<Digest>,
}
#[derive(Debug, Serialize)]
pub struct Choice {
    pub mode: PlacementMode,
    pub variant_digest: Digest,
    pub costs_digest: Digest,
    pub placement_digest: Digest,
    pub profiles: Vec<ProfileKey>,
    pub stages: Vec<PreviewStage>,
    pub decode: DecodeCost,
}
#[derive(Debug, Serialize)]
pub struct ModeReport {
    pub version: Version,
    pub fixture: bool,
    pub family: Family,
    pub input_digest: Digest,
    pub qualification: &'static str,
    pub admission: bool,
    pub search_complete: bool,
    pub single_node_fits: bool,
    pub states: u32,
    pub cost_rows: u32,
    pub reason: &'static str,
    pub refusals: BTreeMap<String, u32>,
    pub winner: Option<Choice>,
    pub runner_up: Option<Choice>,
}
fn zero(digest: &Digest) -> bool {
    digest.as_str().bytes().all(|b| b == b'0')
}
fn plus(a: u64, b: u64) -> Result<u64, &'static str> {
    a.checked_add(b).ok_or("MODE_COST_OVERFLOW")
}

fn validate(input: &ModeInput) -> Result<(), CoreError> {
    if input.variants.is_empty()
        || input.variants.len() > 16
        || input.link_limits.p50_us.0 == 0
        || input.link_limits.p50_us > input.link_limits.p99_us
    {
        return Err(CoreError::Invalid);
    }
    let first = &input.variants[0].placement;
    let key = &first
        .nodes
        .first()
        .ok_or(CoreError::Invalid)?
        .expected_profile_key;
    let mut snapshots = BTreeMap::new();
    let mut variants = BTreeSet::new();
    for variant in &input.variants {
        let p = &variant.placement;
        if p.fixture != input.fixture
            || p.manifest != first.manifest
            || p.now_ms != first.now_ms
            || p.now_wall_ms != first.now_wall_ms
            || p.nodes.is_empty()
            || p.nodes.len() > 64
            || p.manifest.layers.is_empty()
            || p.manifest.layers.len() > 4096
            || p.links.len() > 128
            || variant.costs.tiers.len() > 512
            || variant.costs.storage.len() > 64
            || variant.costs.links.len() > 128
            || !variants.insert(content_hash(variant).map_err(|_| CoreError::Invalid)?)
        {
            return Err(CoreError::Invalid);
        }
        let mut profile_digests = BTreeMap::new();
        for node in &p.nodes {
            let cap = &node.capabilities;
            if node.family != input.family
                || node.expected_profile_key.workload != key.workload
                || cap.cluster_id != first.nodes[0].capabilities.cluster_id
            {
                return Err(CoreError::Invalid);
            }
            let captured =
                content_hash(&(cap, node.captured_at_ms)).map_err(|_| CoreError::Invalid)?;
            if snapshots
                .insert(cap.node_id, captured.clone())
                .is_some_and(|old| old != captured)
            {
                return Err(CoreError::Invalid);
            }
            if snapshots.len() > 64
                || profile_digests
                    .insert(
                        cap.node_id,
                        content_hash(&node.profile).map_err(|_| CoreError::Invalid)?,
                    )
                    .is_some()
            {
                return Err(CoreError::Invalid);
            }
        }
        let mut tiers = BTreeSet::new();
        for tier in &variant.costs.tiers {
            let node = p
                .nodes
                .iter()
                .find(|n| n.capabilities.node_id == tier.node)
                .ok_or(CoreError::Invalid)?;
            if tier.bytes_per_second.0 == 0
                || !tiers.insert((tier.node, &tier.pool_id))
                || Some(&tier.profile_digest) != profile_digests.get(&tier.node)
                || !node
                    .capabilities
                    .memory_pools
                    .iter()
                    .any(|m| m.pool_id == tier.pool_id)
            {
                return Err(CoreError::Invalid);
            }
        }
        let mut storage = BTreeSet::new();
        for read in &variant.costs.storage {
            if !storage.insert(read.node)
                || Some(&read.profile_digest) != profile_digests.get(&read.node)
            {
                return Err(CoreError::Invalid);
            }
            if let StorageTraffic::Available {
                bytes_per_second,
                layer_bytes_per_token,
                ..
            } = &read.traffic
                && (bytes_per_second.0 == 0
                    || layer_bytes_per_token.len() != p.manifest.layers.len())
            {
                return Err(CoreError::Invalid);
            }
        }
        let mut links = BTreeSet::new();
        for latency in &variant.costs.links {
            if !links.insert((latency.source, latency.destination))
                || !p.links.iter().any(|l| {
                    l.source == latency.source
                        && l.destination == latency.destination
                        && l.evidence_digest == latency.link_evidence_digest
                })
                || latency.p50_us.is_some_and(|x| x.0 == 0)
                || latency.p99_us.is_some_and(|x| x.0 == 0)
                || matches!((latency.p50_us, latency.p99_us), (Some(a),Some(b)) if a > b)
            {
                return Err(CoreError::Invalid);
            }
        }
    }
    Ok(())
}

struct Costs<'a> {
    nodes: BTreeMap<NodeId, &'a planner::NodeInput>,
    tiers: BTreeMap<(NodeId, &'a str), &'a TierBandwidth>,
    links: BTreeMap<(NodeId, NodeId), &'a LinkLatency>,
    storage: BTreeMap<NodeId, (&'a StorageTraffic, Vec<u128>)>,
}
impl<'a> Costs<'a> {
    fn new(variant: &'a Variant) -> Self {
        Self {
            nodes: variant
                .placement
                .nodes
                .iter()
                .map(|n| (n.capabilities.node_id, n))
                .collect(),
            tiers: variant
                .costs
                .tiers
                .iter()
                .map(|t| ((t.node, t.pool_id.as_str()), t))
                .collect(),
            links: variant
                .costs
                .links
                .iter()
                .map(|l| ((l.source, l.destination), l))
                .collect(),
            storage: variant
                .costs
                .storage
                .iter()
                .map(|s| {
                    let mut prefix = vec![0];
                    if let StorageTraffic::Available {
                        layer_bytes_per_token,
                        ..
                    } = &s.traffic
                    {
                        for bytes in layer_bytes_per_token {
                            prefix.push(prefix.last().copied().unwrap() + u128::from(bytes.0));
                        }
                    }
                    (s.node, (&s.traffic, prefix))
                })
                .collect(),
        }
    }
}
fn decode_cost(
    input: &ModeInput,
    variant: &Variant,
    costs: &Costs<'_>,
    candidate: &Candidate,
) -> Result<DecodeCost, &'static str> {
    let now = variant.placement.now_ms.0;
    let mut memory = 0;
    let mut storage = 0;
    let mut copy = 0;
    let mut link = 0;
    let mut evidence = BTreeSet::new();
    for stage in &candidate.stages {
        let node = costs.nodes.get(&stage.node).ok_or("MODE_INPUT_MISMATCH")?;
        copy = plus(copy, node.copy.decode_us_per_token.0)?;
        if stage
            .pools
            .iter()
            .all(|p| p.active_weight_bytes_per_token.0 == 0)
        {
            return Err("MODE_ACTIVE_BYTES_MISSING");
        }
        for pool in &stage.pools {
            if pool.active_weight_bytes_per_token.0 == 0 {
                continue;
            }
            let Some(tier) = costs.tiers.get(&(stage.node, pool.pool_id.as_str())) else {
                return Err("MODE_TIER_MISSING");
            };
            if !tier.observation.fresh(now, input.fixture) {
                return Err("MODE_TIER_UNQUALIFIED_OR_STALE");
            }
            let micros = (u128::from(pool.active_weight_bytes_per_token.0) * 1_000_000)
                .div_ceil(u128::from(tier.bytes_per_second.0));
            memory = plus(
                memory,
                u64::try_from(micros).map_err(|_| "MODE_COST_OVERFLOW")?,
            )?;
            evidence.insert(tier.observation.evidence_digest.clone());
        }
        let (traffic, prefix) = costs
            .storage
            .get(&stage.node)
            .ok_or("MODE_STORAGE_MISSING")?;
        match traffic {
            StorageTraffic::NotUsed => (),
            StorageTraffic::Unavailable => return Err("MODE_STORAGE_UNAVAILABLE"),
            StorageTraffic::Available {
                bytes_per_second,
                first_stage_bytes_per_token,
                last_stage_bytes_per_token,
                per_stage_bytes_per_token,
                observation,
                ..
            } => {
                if !observation.fresh(now, input.fixture) {
                    return Err("MODE_STORAGE_UNQUALIFIED_OR_STALE");
                }
                let mut bytes = prefix[stage.end] - prefix[stage.start]
                    + u128::from(per_stage_bytes_per_token.0);
                if stage.start == 0 {
                    bytes += u128::from(first_stage_bytes_per_token.0);
                }
                if stage.end == variant.placement.manifest.layers.len() {
                    bytes += u128::from(last_stage_bytes_per_token.0);
                }
                let micros = (bytes * 1_000_000).div_ceil(u128::from(bytes_per_second.0));
                storage = plus(
                    storage,
                    u64::try_from(micros).map_err(|_| "MODE_COST_OVERFLOW")?,
                )?;
                evidence.insert(observation.evidence_digest.clone());
            }
        }
    }
    for pair in candidate.stages.windows(2) {
        let Some(latency) = costs.links.get(&(pair[0].node, pair[1].node)) else {
            return Err("MODE_LINK_MISSING");
        };
        let (Some(p50), Some(p99), Some(idle)) =
            (latency.p50_us, latency.p99_us, latency.idle_wake_us)
        else {
            return Err("MODE_LINK_TAIL_UNKNOWN");
        };
        if !latency.observation.fresh(now, input.fixture) {
            return Err("MODE_LINK_UNQUALIFIED_OR_STALE");
        }
        if p50 > input.link_limits.p50_us
            || p99 > input.link_limits.p99_us
            || idle > input.link_limits.idle_wake_us
        {
            return Err("MODE_LINK_LIMIT");
        }
        link = plus(link, p50.0)?;
        evidence.insert(latency.observation.evidence_digest.clone());
    }
    let total = plus(plus(plus(memory, storage)?, copy)?, link)?;
    if total == 0 {
        return Err("MODE_COST_ZERO");
    }
    Ok(DecodeCost {
        memory_read_us: Decimal(memory),
        storage_read_us: Decimal(storage),
        local_copy_us: Decimal(copy),
        link_rtt_us: Decimal(link),
        total_us: Decimal(total),
        predicted_tokens_per_second: 1_000_000.0 / total as f64,
        evidence: evidence.into_iter().collect(),
    })
}

pub fn choose(input: &ModeInput, mode: FixtureMode) -> Result<ModeReport, CoreError> {
    validate(input)?;
    if input.fixture && mode != FixtureMode::Simulation {
        return Err(CoreError::Invalid);
    }
    let mut report = ModeReport {
        version: Version,
        fixture: input.fixture,
        family: input.family,
        input_digest: content_hash(input).map_err(|_| CoreError::Invalid)?,
        qualification: "not_assessed",
        admission: false,
        search_complete: true,
        single_node_fits: false,
        states: 0,
        cost_rows: 0,
        reason: "NO_USABLE_MODE_COST",
        refusals: BTreeMap::new(),
        winner: None,
        runner_up: None,
    };
    let mut choices = Vec::new();
    // All configurations must finish single-device assessment before any split search.
    for pooling in [false, true] {
        if pooling && report.single_node_fits {
            break;
        }
        for variant in &input.variants {
            if pooling
                && (variant.configuration == ConfigurationMode::ExpertOffload
                    || variant.placement.max_stages.get() == 1)
            {
                continue;
            }
            let costs_digest = content_hash(&(variant.configuration, &variant.costs))
                .map_err(|_| CoreError::Invalid)?;
            let approved =
                input.fixture || variant.approved_costs_digest.as_ref() == Some(&costs_digest);
            let costs = Costs::new(variant);
            let score = |candidate: &Candidate| -> Result<u64, &'static str> {
                if !approved {
                    return Err("UNAPPROVED_MODE_COSTS");
                }
                decode_cost(input, variant, &costs, candidate).map(|c| c.total_us.0)
            };
            if report.states >= 100000 || report.cost_rows >= 1000000 {
                report.search_complete = false;
                report.reason = "SEARCH_INCOMPLETE";
                return Ok(report);
            }
            let mut placement = variant.placement.clone();
            placement.max_states =
                BoundedU32::new(placement.max_states.get().min(100000 - report.states))
                    .map_err(|_| CoreError::Invalid)?;
            placement.max_cost_rows = BoundedU32::new(
                placement
                    .max_cost_rows
                    .get()
                    .min(1000000 - report.cost_rows),
            )
            .map_err(|_| CoreError::Invalid)?;
            placement.top_k = BoundedU32::new(2).map_err(|_| CoreError::Invalid)?;
            if !pooling {
                placement.max_stages = BoundedU32::new(1).map_err(|_| CoreError::Invalid)?;
            }
            let preview = planner::preview_scored(&placement, mode, Some(&score))?;
            report.single_node_fits |= preview.single_node_fits;
            report.search_complete &= preview.search_complete;
            report.states += preview.states;
            report.cost_rows += preview.cost_rows;
            for (code, count) in preview.refusals {
                *report.refusals.entry(code.into()).or_default() += count;
            }
            if !report.search_complete {
                report.reason = "SEARCH_INCOMPLETE";
                return Ok(report);
            }
            for candidate in preview.candidates {
                let profiles: Vec<ProfileKey> = candidate
                    .stages
                    .iter()
                    .map(|s| {
                        variant
                            .placement
                            .nodes
                            .iter()
                            .find(|n| n.capabilities.node_id == s.node)
                            .map(|n| n.expected_profile_key.clone())
                            .ok_or(CoreError::Invalid)
                    })
                    .collect::<Result<_, _>>()?;
                let decode = decode_cost(input, variant, &costs, &candidate)
                    .map_err(|_| CoreError::Invalid)?;
                let placement_digest = content_hash(&(
                    variant.configuration,
                    &profiles,
                    candidate
                        .stages
                        .iter()
                        .map(|s| (s.node, s.start, s.end))
                        .collect::<Vec<_>>(),
                ))
                .map_err(|_| CoreError::Invalid)?;
                choices.push(Choice {
                    mode: if candidate.stages.len() > 1 {
                        PlacementMode::LayerSplit
                    } else if variant.configuration == ConfigurationMode::ExpertOffload {
                        PlacementMode::ExpertOffload
                    } else {
                        PlacementMode::WholeRequest
                    },
                    variant_digest: content_hash(variant).map_err(|_| CoreError::Invalid)?,
                    costs_digest: costs_digest.clone(),
                    placement_digest,
                    profiles,
                    stages: candidate.stages,
                    decode,
                });
            }
        }
    }
    if report.single_node_fits {
        choices.retain(|c| c.mode != PlacementMode::LayerSplit);
    }
    choices.sort_by(|a, b| {
        a.decode
            .total_us
            .cmp(&b.decode.total_us)
            .then_with(|| a.variant_digest.cmp(&b.variant_digest))
            .then_with(|| {
                a.stages
                    .iter()
                    .map(|s| (s.node, s.start, s.end))
                    .cmp(b.stages.iter().map(|s| (s.node, s.start, s.end)))
            })
    });
    let mut seen = BTreeSet::new();
    let mut ranked = choices
        .into_iter()
        .filter(|c| seen.insert(c.placement_digest.clone()));
    report.winner = ranked.next();
    report.runner_up = ranked.next();
    if report.winner.is_some() {
        report.reason = if report.single_node_fits {
            "SINGLE_DEVICE_FITS"
        } else {
            "POOLING_FOR_CAPACITY"
        };
    }
    Ok(report)
}
