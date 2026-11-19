//! Advisory bounded cold-stage preview. No reservation, launch or native qualification.
use crate::{
    CoreError,
    profiles::{self, PoolBudget, ProfileAge, ProfileStatus},
};
use serde::{Deserialize, Serialize};
use starlings_protocol::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Nvidia,
    Apple,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    Latency,
    Throughput,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeInput {
    pub capabilities: NodeCapabilities,
    pub captured_at_ms: Decimal,
    pub family: Family,
    pub backend_id: String,
    pub expected_profile_key: ProfileKey,
    pub profile: ModelCostProfile,
    pub profile_clocks: ProfileAge,
    pub approved_profile_digest: Option<Digest>,
    /// Declared conservative cold-load upper bound for any selected range on this node.
    pub load_us: Decimal,
    /// Declared per-stage copy costs at this exact workload, separate from link transfer.
    pub copy: PhaseCost,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkCost {
    pub source: NodeId,
    pub destination: NodeId,
    pub fixture: bool,
    pub boundary_profile: String,
    pub workload_digest: Digest,
    pub evidence_digest: Digest,
    pub transfer: PhaseCost,
    pub age_at_capture_ms: Decimal,
    pub captured_at_ms: Decimal,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewInput {
    pub version: Version,
    pub fixture: bool,
    pub manifest: ModelManifest,
    pub now_ms: Decimal,
    pub now_wall_ms: i64,
    pub nodes: Vec<NodeInput>,
    pub links: Vec<LinkCost>,
    pub objective: Objective,
    pub max_stages: BoundedU32<1, 8>,
    pub top_k: BoundedU32<1, 16>,
    pub max_states: BoundedU32<1, 100000>,
    pub max_cost_rows: BoundedU32<1, 1000000>,
}
#[derive(Clone, Debug, Serialize)]
pub struct PreviewStage {
    pub node: NodeId,
    pub snapshot: SnapshotId,
    pub profile_digest: Digest,
    pub start: usize,
    pub end: usize,
    pub pools: Vec<PoolBudget>,
    pub load_us: Decimal,
    pub prefill_us: Decimal,
    pub decode_us: Decimal,
}
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub family: Family,
    pub stages: Vec<PreviewStage>,
    pub link_evidence: Vec<Digest>,
    pub load_us: Decimal,
    pub prefill_path_us: Decimal,
    pub decode_path_us: Decimal,
    pub bottleneck_us: Decimal,
    pub score_us: Decimal,
}
#[derive(Debug, Serialize)]
pub struct Preview {
    pub version: Version,
    pub fixture: bool,
    pub objective: Objective,
    pub input_digest: Digest,
    pub qualification: &'static str,
    pub admission: bool,
    pub search_complete: bool,
    pub single_node_fits: bool,
    pub states: u32,
    pub cost_rows: u32,
    pub refusals: BTreeMap<&'static str, u32>,
    /// Attempt counts, not a verdict that a node cannot participate in any candidate.
    pub node_refusals: BTreeMap<NodeId, BTreeMap<&'static str, u32>>,
    pub candidates: Vec<Candidate>,
}
struct Prepared {
    index: usize,
    digest: Digest,
    sharding: bool,
}
struct Search<'a> {
    input: &'a PreviewInput,
    nodes: Vec<Prepared>,
    links: BTreeMap<(NodeId, NodeId), &'a LinkCost>,
    report: Preview,
    score: Option<Score<'a>>,
}
type Score<'a> = &'a dyn Fn(&Candidate) -> Result<u64, &'static str>;
fn plus(a: u64, b: u64) -> Result<u64, CoreError> {
    a.checked_add(b).ok_or(CoreError::Limit)
}
fn age(initial: u64, captured: u64, now: u64) -> Result<u64, CoreError> {
    plus(initial, now.checked_sub(captured).ok_or(CoreError::Clock)?)
}

pub fn preview(input: &PreviewInput, mode: FixtureMode) -> Result<Preview, CoreError> {
    preview_scored(input, mode, None)
}
pub(crate) fn preview_scored(
    input: &PreviewInput,
    mode: FixtureMode,
    score: Option<Score<'_>>,
) -> Result<Preview, CoreError> {
    if input.nodes.len() > 64
        || input.links.len() > 128
        || input.fixture != input.manifest.fixture
        || (input.fixture && mode == FixtureMode::Reject)
    {
        return Err(CoreError::Invalid);
    }
    encode(&input.manifest, mode).map_err(|_| CoreError::Invalid)?;
    let workload = input
        .nodes
        .first()
        .map(|n| &n.expected_profile_key.workload);
    let workload_digest = workload
        .map(content_hash)
        .transpose()
        .map_err(|_| CoreError::Invalid)?;
    let mut ids = BTreeSet::new();
    let mut search = Search {
        input,
        nodes: vec![],
        links: BTreeMap::new(),
        score,
        report: Preview {
            version: Version,
            fixture: input.fixture,
            objective: input.objective,
            input_digest: content_hash(input).map_err(|_| CoreError::Invalid)?,
            qualification: "not_assessed",
            admission: false,
            search_complete: true,
            single_node_fits: false,
            states: 0,
            cost_rows: 0,
            refusals: BTreeMap::new(),
            node_refusals: BTreeMap::new(),
            candidates: vec![],
        },
    };
    let mut cluster = None;
    for (index, node) in input.nodes.iter().enumerate() {
        let cap = &node.capabilities;
        if !ids.insert(cap.node_id)
            || cap.fixture != input.fixture
            || node.profile.fixture != input.fixture
            || Some(&node.expected_profile_key.workload) != workload
            || node.profile_clocks.now_ms != input.now_ms.0
            || node.profile_clocks.now_wall_ms != input.now_wall_ms
        {
            return Err(CoreError::Invalid);
        }
        if cluster
            .replace(cap.cluster_id)
            .is_some_and(|old| old != cap.cluster_id)
        {
            return Err(CoreError::Invalid);
        }
        encode(cap, mode).map_err(|_| CoreError::Invalid)?;
        let assessment = profiles::assess(
            &node.profile,
            &input.manifest,
            &node.expected_profile_key,
            0..input.manifest.layers.len(),
            node.profile_clocks,
            node.approved_profile_digest.as_ref(),
            mode,
        );
        let assessment = match assessment {
            Ok(a) => a,
            Err(_) => {
                search.refuse_node(cap.node_id, "INVALID_PROFILE");
                continue;
            }
        };
        if assessment.status != ProfileStatus::Qualified
            && !(input.fixture && assessment.status == ProfileStatus::FixtureOnly)
        {
            search.refuse_node(cap.node_id, "UNQUALIFIED_OR_STALE_PROFILE");
            continue;
        }
        if cap.state != NodeCapabilitiesState::Ready
            || age(
                cap.sample_age_ms.get().into(),
                node.captured_at_ms.0,
                input.now_ms.0,
            )? >= 5000
        {
            search.refuse_node(cap.node_id, "STALE_OR_UNAVAILABLE_CAPABILITY");
            continue;
        }
        let family_matches = match node.family {
            Family::Apple => {
                cap.os == NodeCapabilitiesOs::Macos
                    && cap.architecture == NodeCapabilitiesArchitecture::Aarch64
            }
            Family::Nvidia => matches!(
                cap.os,
                NodeCapabilitiesOs::Linux | NodeCapabilitiesOs::Windows
            ),
        };
        if !family_matches {
            search.refuse_node(cap.node_id, "FAMILY_MISMATCH");
            continue;
        }
        let key = &node.expected_profile_key;
        let backend = cap.backends.iter().find(|b| {
            b.backend_id == node.backend_id
                && b.build_digest == key.backend_build_digest
                && b.architectures.contains(&input.manifest.architecture)
                && b.execution_profiles.contains(&key.execution_profile)
                && b.boundary_profiles.contains(&key.boundary_profile)
                && b.task_types.contains(&BackendTaskTypes::Generate)
                && b.max_context_tokens.get() >= key.workload.context_tokens.get()
                && b.max_sessions.get() >= key.workload.sessions.get()
        });
        let Some(backend) = backend else {
            search.refuse_node(cap.node_id, "INCOMPATIBLE_BACKEND");
            continue;
        };
        if assessment.pools.iter().any(|p| {
            !backend.memory_pool_ids.contains(&p.pool_id)
                || !cap
                    .memory_pools
                    .iter()
                    .any(|m| m.pool_id == p.pool_id && m.device_ids.contains(&backend.device_id))
        }) {
            search.refuse_node(cap.node_id, "POOL_BINDING_MISMATCH");
            continue;
        }
        search.nodes.push(Prepared {
            index,
            digest: assessment.profile_digest,
            sharding: backend.layer_sharding,
        });
    }
    search
        .nodes
        .sort_by_key(|p| input.nodes[p.index].capabilities.node_id);
    let mut edges = BTreeSet::new();
    for link in &input.links {
        if !ids.contains(&link.source)
            || !ids.contains(&link.destination)
            || link.source == link.destination
            || link.fixture != input.fixture
            || Some(&link.workload_digest) != workload_digest.as_ref()
            || link.boundary_profile != input.manifest.required_boundary_profile
            || link.transfer.prefill_us.0 == 0
            || link.transfer.decode_us_per_token.0 == 0
            || link.evidence_digest.as_str().bytes().all(|b| b == b'0')
            || !edges.insert((link.source, link.destination))
        {
            return Err(CoreError::Invalid);
        }
        if age(
            link.age_at_capture_ms.0,
            link.captured_at_ms.0,
            input.now_ms.0,
        )? >= crate::links::LINK_TTL_MS
        {
            search.refuse("STALE_LINK");
            continue;
        }
        search.links.insert((link.source, link.destination), link);
    }
    if search.nodes.is_empty() {
        search.refuse("NO_ELIGIBLE_NODE");
        return Ok(search.report);
    }
    // Always assess every single-node fit before considering pooling for capacity.
    for prepared in 0..search.nodes.len() {
        if let Some(stage) = search.stage(prepared, 0, input.manifest.layers.len(), false)? {
            search.complete(vec![stage])?;
        }
        if !search.report.search_complete {
            return Ok(search.report);
        }
    }
    if search.report.single_node_fits || input.max_stages.get() == 1 {
        if search.report.single_node_fits {
            search.refuse("POOLING_NOT_NEEDED");
        }
        return Ok(search.report);
    }
    search.walk(0, 0, &mut Vec::new())?;
    Ok(search.report)
}
impl Search<'_> {
    fn refuse(&mut self, code: &'static str) {
        *self.report.refusals.entry(code).or_default() += 1;
    }
    fn refuse_node(&mut self, node: NodeId, code: &'static str) {
        self.refuse(code);
        *self
            .report
            .node_refusals
            .entry(node)
            .or_default()
            .entry(code)
            .or_default() += 1;
    }
    fn stage(
        &mut self,
        prepared: usize,
        start: usize,
        end: usize,
        pipeline: bool,
    ) -> Result<Option<PreviewStage>, CoreError> {
        if self.report.states == self.input.max_states.get() {
            self.report.search_complete = false;
            return Ok(None);
        }
        self.report.states += 1;
        let p = &self.nodes[prepared];
        let n = &self.input.nodes[p.index];
        if pipeline && !p.sharding {
            self.refuse_node(n.capabilities.node_id, "LAYER_SHARDING_UNSUPPORTED");
            return Ok(None);
        }
        let ModelCosts::Available {
            pool_ids,
            components,
            ..
        } = &n.profile.costs
        else {
            return Err(CoreError::Invalid);
        };
        let rows = (end - start + components.len() + 1)
            .checked_mul(pool_ids.len())
            .ok_or(CoreError::Limit)?;
        let cost = self
            .report
            .cost_rows
            .checked_add(rows.try_into().map_err(|_| CoreError::Limit)?)
            .ok_or(CoreError::Limit)?;
        if cost > self.input.max_cost_rows.get() {
            self.report.search_complete = false;
            return Ok(None);
        }
        self.report.cost_rows = cost;
        let (pools, timing) = profiles::stage_costs(&n.profile, start..end)?;
        if pools.iter().any(|p| {
            n.capabilities
                .memory_pools
                .iter()
                .find(|m| m.pool_id == p.pool_id)
                .is_none_or(|m| {
                    m.available_bytes
                        .0
                        .checked_sub(m.reserved_bytes.0)
                        .is_none_or(|free| p.total_bytes.0 > free)
                })
        }) {
            self.refuse_node(n.capabilities.node_id, "INSUFFICIENT_POOL_MEMORY");
            return Ok(None);
        }
        if !pipeline {
            self.report.single_node_fits = true;
        }
        let Some(timing) = timing else {
            self.refuse_node(n.capabilities.node_id, "MISSING_TIMING");
            return Ok(None);
        };
        Ok(Some(PreviewStage {
            node: n.capabilities.node_id,
            snapshot: n.capabilities.snapshot_id,
            profile_digest: p.digest.clone(),
            start,
            end,
            pools,
            load_us: n.load_us,
            prefill_us: Decimal(plus(timing.prefill_us.0, n.copy.prefill_us.0)?),
            decode_us: Decimal(plus(
                timing.decode_us_per_token.0,
                n.copy.decode_us_per_token.0,
            )?),
        }))
    }
    fn walk(
        &mut self,
        start: usize,
        used: u64,
        stages: &mut Vec<PreviewStage>,
    ) -> Result<(), CoreError> {
        if stages.len() >= self.input.max_stages.get() as usize {
            self.refuse("STAGE_LIMIT");
            return Ok(());
        }
        for prepared in 0..self.nodes.len() {
            let node = &self.input.nodes[self.nodes[prepared].index];
            let id = node.capabilities.node_id;
            let bit = 1u64 << prepared;
            if used & bit != 0 {
                continue;
            }
            if let Some(first) = stages.first() {
                let family = self
                    .input
                    .nodes
                    .iter()
                    .find(|n| n.capabilities.node_id == first.node)
                    .ok_or(CoreError::Invalid)?
                    .family;
                if node.family != family {
                    self.refuse_node(id, "FAMILY_MISMATCH");
                    continue;
                }
            }
            if let Some(previous) = stages.last()
                && !self.links.contains_key(&(previous.node, id))
            {
                self.refuse_node(id, "MISSING_DIRECTED_LINK");
                continue;
            }
            for end in start + 1..=self.input.manifest.layers.len() {
                let Some(stage) = self.stage(prepared, start, end, true)? else {
                    if !self.report.search_complete {
                        return Ok(());
                    }
                    continue;
                };
                stages.push(stage);
                if end == self.input.manifest.layers.len() {
                    self.complete(stages.clone())?;
                } else {
                    self.walk(end, used | bit, stages)?;
                }
                stages.pop();
                if !self.report.search_complete {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
    fn complete(&mut self, stages: Vec<PreviewStage>) -> Result<(), CoreError> {
        let mut links = Vec::new();
        let mut load = 0;
        let mut prefill = 0;
        let mut decode = 0;
        let mut bottleneck = 0;
        for (i, stage) in stages.iter().enumerate() {
            load = plus(load, stage.load_us.0)?;
            prefill = plus(prefill, stage.prefill_us.0)?;
            decode = plus(decode, stage.decode_us.0)?;
            let mut cycle = stage.decode_us.0;
            if let Some(next) = stages.get(i + 1) {
                let edge = self
                    .links
                    .get(&(stage.node, next.node))
                    .ok_or(CoreError::Invalid)?;
                prefill = plus(prefill, edge.transfer.prefill_us.0)?;
                decode = plus(decode, edge.transfer.decode_us_per_token.0)?;
                cycle = plus(cycle, edge.transfer.decode_us_per_token.0)?;
                links.push(edge.evidence_digest.clone());
            }
            bottleneck = bottleneck.max(cycle);
        }
        let output = self.input.nodes[self.nodes[0].index]
            .profile
            .key
            .workload
            .output_tokens
            .get();
        let score = match self.input.objective {
            Objective::Latency => plus(
                plus(load, prefill)?,
                decode.checked_mul(output.into()).ok_or(CoreError::Limit)?,
            )?,
            Objective::Throughput => bottleneck,
        };
        let family = self
            .input
            .nodes
            .iter()
            .find(|n| n.capabilities.node_id == stages[0].node)
            .ok_or(CoreError::Invalid)?
            .family;
        let mut candidate = Candidate {
            family,
            stages,
            link_evidence: links,
            load_us: Decimal(load),
            prefill_path_us: Decimal(prefill),
            decode_path_us: Decimal(decode),
            bottleneck_us: Decimal(bottleneck),
            score_us: Decimal(score),
        };
        if let Some(score) = self.score {
            match score(&candidate) {
                Ok(value) => candidate.score_us = Decimal(value),
                Err(code) => {
                    self.refuse(code);
                    return Ok(());
                }
            }
        }
        self.report.candidates.push(candidate);
        self.report.candidates.sort_by(|a, b| {
            a.score_us.cmp(&b.score_us).then_with(|| {
                a.stages
                    .iter()
                    .map(|s| (s.node, s.start, s.end))
                    .cmp(b.stages.iter().map(|s| (s.node, s.start, s.end)))
            })
        });
        self.report
            .candidates
            .truncate(self.input.top_k.get() as usize);
        Ok(())
    }
}
