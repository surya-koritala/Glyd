//! Coordinator-owned selection. No I/O under the router lock or untrusted qualification.
use starlings_cli::{
    inference::Dispatch,
    routes::{CoordinatorReplica, CoordinatorRoute},
};
use starlings_core::routing::{Task, TaskRouter};
use starlings_protocol::{FixtureMode, NodeId, RequestId};
use starlings_security::{AuthorityEpoch, Error, ResourceView};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Instant,
};
type Result<T> = std::result::Result<T, Error>;
type Owner = Arc<Mutex<Option<(AuthorityEpoch, Dispatch)>>>;
const MAX_RETAINED_BYTES: usize = 32 * 1024 * 1024;

struct State {
    epoch: Option<AuthorityEpoch>,
    router: TaskRouter,
    samples: BTreeMap<NodeId, String>,
    owners: BTreeMap<NodeId, Owner>,
    requests: BTreeMap<RequestId, usize>,
    excluded: BTreeMap<RequestId, BTreeSet<NodeId>>,
    claimed: BTreeSet<RequestId>,
}

#[cfg(all(test, feature = "test-backend"))]
mod tests {
    use super::*;
    use starlings_cli::routes::Qualification;
    use starlings_core::routing::WarmReplica;
    use starlings_protocol::*;
    use starlings_security::{Error, ResourceSample};
    use std::{thread, time::Duration};

    fn replica(n: u32) -> (CoordinatorReplica, ResourceView) {
        let id: NodeId = format!("00000000-0000-4000-8000-{n:012x}").parse().unwrap();
        let digest = byte_digest(b"synthetic router identity; not qualification evidence");
        let mut plan: PlacementPlan = decode(
            include_bytes!("../../../examples/placement-plan.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        plan.fixture = false;
        plan.mode = PlacementPlanMode::WholeRequest;
        plan.manifest_digest = digest.clone();
        plan.stages.truncate(1);
        let stage = &mut plan.stages[0];
        stage.node_id = id;
        stage.device_id = "cpu".into();
        stage.backend_build_digest = digest.clone();
        stage.memory_pool_id = "host".into();
        stage.layer_end_exclusive = plan.model_layer_count;
        stage.owns_final_norm_and_head = true;
        let request_bytes = stage.reserved_bytes().unwrap();
        plan.plan_digest =
            canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
        let qualification = Qualification {
            version: Version,
            fixture: true,
            os: "linux".into(),
            architecture: "x86_64".into(),
            device_kind: "cpu".into(),
            target: digest.clone(),
            loading_bytes: Decimal(request_bytes * 2),
            model_layer_count: plan.model_layer_count.get(),
            boundary_profile: plan.boundary_profile.clone(),
            request_drain_profile: "slots-idle-v1".into(),
            configuration: digest.clone(),
            max_output_tokens: 8,
            memory_qualified: true,
            exact_count_qualified: true,
            request_drain_qualified: true,
            cpu_threads: None,
            replica: WarmReplica {
                node: id,
                manifest: digest.clone(),
                profile: plan.execution_profile.clone(),
                backend: plan.stages[0].backend_id.clone(),
                build: digest.clone(),
                pool: "host".into(),
                evidence: digest.clone(),
                request_bytes,
                measured_request_ms: n.into(),
                context_limit: plan.context_limit.get(),
                max_sessions: 1,
                accepting: true,
            },
        };
        let endpoint = format!("127.0.0.1:{}", 19000 + n).parse().unwrap();
        let mut node: NodeCapabilities = decode(
            include_bytes!("../../../examples/node-capabilities.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        // Exercise production record validation using explicitly synthetic test data.
        node.fixture = false;
        node.node_id = id;
        node.control_endpoint = format!("tls://{endpoint}");
        node.memory_pools[0].pool_id = "host".into();
        node.memory_pools[0].kind = MemoryPoolKind::System;
        node.memory_pools[0].device_ids = vec!["cpu".into()];
        node.backends[0].build_digest = digest;
        node.backends[0].device_id = "cpu".into();
        node.backends[0].memory_pool_ids = vec!["host".into()];
        node.backends[0].layer_sharding = false;
        (
            CoordinatorReplica {
                worker_endpoint: endpoint,
                plan,
                qualification,
            },
            ResourceView {
                sample: ResourceSample {
                    sequence: Decimal(1),
                    capabilities: node,
                    links: vec![],
                },
                age_ms: 0,
            },
        )
    }
    #[test]
    fn selection_is_atomic_bounded_retains_uncertain_work_and_preserves_owner() {
        let (a, view_a) = replica(1);
        let (b, view_b) = replica(2);
        let mut config = CoordinatorRoute {
            version: Version,
            listen: "127.0.0.1:0".parse().unwrap(),
            worker_endpoint: a.worker_endpoint,
            plan: a.plan.clone(),
            qualification: a.qualification.clone(),
            replicas: vec![],
        };
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("replicas").is_none());
        config.validate(&content_hash(&config).unwrap()).unwrap();
        config.replicas.push(b.clone());
        config.validate(&content_hash(&config).unwrap()).unwrap();
        let mut invalid = config.clone();
        invalid.replicas.push(b);
        assert!(invalid.validate(&content_hash(&invalid).unwrap()).is_err());
        let epoch = AuthorityEpoch {
            number: Decimal(1),
            incarnation: uuid::Uuid::new_v4(),
        };
        let client = "00000000-0000-4000-8000-000000000050".parse().unwrap();
        let mut pool = Pool::new(config);
        let views = [
            (Some(view_a), Instant::now()),
            (Some(view_b), Instant::now()),
        ];
        pool.refresh(epoch, &views).unwrap();
        let first = pool
            .select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
            .unwrap();
        let second = pool
            .select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
            .unwrap();
        assert_ne!(
            first.config.qualification.replica.node,
            second.config.qualification.replica.node
        );
        assert!(
            pool.select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
                .is_err()
        );
        pool.failed(first.id, first.config.qualification.replica.node);
        let mut invalid_views = views.clone();
        let invalid = &mut invalid_views[0].0.as_mut().unwrap().sample.capabilities;
        invalid.snapshot_id = uuid::Uuid::new_v4().to_string().parse().unwrap();
        invalid.backends[0].build_digest = "0".repeat(64).try_into().unwrap();
        pool.refresh(epoch, &invalid_views).unwrap();
        assert!(
            !pool.cancelled(second.id),
            "one invalid worker must not withdraw healthy active work"
        );
        pool.refresh(epoch, &views).unwrap();
        assert!(
            pool.select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
                .is_err(),
            "fresh status cannot free ambiguous work"
        );
        assert!(pool.cancelled(first.id));
        pool.drained(first.id).unwrap();
        pool.refresh(epoch, &views).unwrap();
        let next = pool
            .select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
            .unwrap();
        assert!(Arc::ptr_eq(&first.owner, &next.owner));
        pool.origin -= Duration::from_millis(5010);
        pool.refresh(epoch, &views).unwrap();
        assert!(
            pool.cancelled(next.id),
            "repeated snapshot cannot refresh its original age"
        );
        let newer = AuthorityEpoch {
            number: Decimal(2),
            incarnation: uuid::Uuid::new_v4(),
        };
        assert!(
            pool.refresh(newer, &views).is_err(),
            "epoch change cannot drop outstanding reservations"
        );
        assert!(pool.cancelled(second.id));
    }
    #[test]
    fn batch_collection_time_cannot_expire_a_later_fresh_worker() {
        for missing in [false, true] {
            let (a, va) = replica(1);
            let (b, vb) = replica(2);
            let node_b = b.qualification.replica.node;
            let pool = Pool::new(CoordinatorRoute {
                version: Version,
                listen: "127.0.0.1:0".parse().unwrap(),
                worker_endpoint: a.worker_endpoint,
                plan: a.plan.clone(),
                qualification: a.qualification.clone(),
                replicas: vec![b],
            });
            let epoch = AuthorityEpoch {
                number: Decimal(1),
                incarnation: uuid::Uuid::new_v4(),
            };
            let client = "00000000-0000-4000-8000-000000000050".parse().unwrap();
            pool.refresh(
                epoch,
                &[
                    (Some(va.clone()), Instant::now()),
                    (Some(vb.clone()), Instant::now()),
                ],
            )
            .unwrap();
            let first = pool
                .select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
                .unwrap();
            let second = pool
                .select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
                .unwrap();
            assert_eq!(second.config.qualification.replica.node, node_b);
            let mut fresh = vb;
            fresh.age_ms = 2500;
            fresh.sample.capabilities.observed_at = "2026-10-12T00:01:00Z".into();
            fresh.sample.capabilities.snapshot_id =
                uuid::Uuid::new_v4().to_string().parse().unwrap();
            let captured = [Instant::now() - Duration::from_secs(3), Instant::now()];
            let mut earlier = va;
            earlier.age_ms = 2500;
            pool.refresh(
                epoch,
                &[
                    (if missing { None } else { Some(earlier) }, captured[0]),
                    (Some(fresh.clone()), captured[1]),
                ],
            )
            .unwrap();
            assert!(
                pool.cancelled(first.id),
                "missing or elapsed-stale worker must withdraw"
            );
            assert!(
                !pool.cancelled(second.id),
                "earlier collection time aged the later fresh view twice"
            );
            assert_eq!(pool.status()["retained_requests"], 2);
            fresh.age_ms = 5000;
            pool.refresh(
                epoch,
                &[(None, Instant::now()), (Some(fresh), Instant::now())],
            )
            .unwrap();
            assert!(
                pool.cancelled(second.id),
                "truly stale view must still withdraw"
            );
            assert_eq!(pool.status()["retained_requests"], 2);
            pool.drained(first.id).unwrap();
            pool.drained(second.id).unwrap();
            assert_eq!(pool.status()["retained_requests"], 0);
        }
    }
    #[test]
    fn queues_enforce_bytes_fairness_abandonment_expiry_and_retry_exclusions() {
        let (a, va) = replica(1);
        let (b, vb) = replica(2);
        let (c, vc) = replica(3);
        let pool = Pool::new(CoordinatorRoute {
            version: Version,
            listen: "127.0.0.1:0".parse().unwrap(),
            worker_endpoint: a.worker_endpoint,
            plan: a.plan.clone(),
            qualification: a.qualification.clone(),
            replicas: vec![b, c],
        });
        let epoch = AuthorityEpoch {
            number: Decimal(1),
            incarnation: uuid::Uuid::new_v4(),
        };
        let client = "00000000-0000-4000-8000-000000000050".parse().unwrap();
        let other = "00000000-0000-4000-8000-000000000051".parse().unwrap();
        pool.refresh(
            epoch,
            &[
                (Some(va), Instant::now()),
                (Some(vb), Instant::now()),
                (Some(vc), Instant::now()),
            ],
        )
        .unwrap();
        let first = pool
            .select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
            .unwrap();
        let second = pool
            .select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
            .unwrap();
        let mut blocked = pool
            .submit(epoch, client, &BTreeSet::new(), pool.deadline().unwrap(), 1)
            .unwrap();
        assert!(blocked.poll(epoch).unwrap().is_none());
        let healthy = pool
            .select(epoch, other, &BTreeSet::new(), pool.deadline().unwrap())
            .unwrap();
        assert_ne!(
            healthy.config.qualification.replica.node,
            first.config.qualification.replica.node
        );
        let abandoned = pool
            .submit(epoch, other, &BTreeSet::new(), pool.deadline().unwrap(), 1)
            .unwrap();
        let abandoned_id = abandoned.id;
        drop(abandoned);
        assert!(
            !pool
                .state
                .lock()
                .unwrap()
                .requests
                .contains_key(&abandoned_id)
        );
        pool.drained(first.id).unwrap();
        let resumed = blocked.poll(epoch).unwrap().unwrap();
        assert_eq!(
            resumed.config.qualification.replica.node,
            first.config.qualification.replica.node
        );
        pool.drained(resumed.id).unwrap();
        // A failure before connection may avoid that node without withdrawing it for others.
        let excluded = BTreeSet::from([first.config.qualification.replica.node]);
        let mut retry = pool
            .submit(epoch, other, &excluded, pool.deadline().unwrap(), 1)
            .unwrap();
        assert!(retry.poll(epoch).unwrap().is_none());
        let unexcluded = pool
            .select(epoch, client, &BTreeSet::new(), pool.deadline().unwrap())
            .unwrap();
        assert_eq!(
            unexcluded.config.qualification.replica.node,
            first.config.qualification.replica.node
        );
        pool.drained(unexcluded.id).unwrap();
        let mut expiring = pool
            .submit(
                epoch,
                other,
                &BTreeSet::new(),
                pool.deadline().unwrap() - 59_000,
                1,
            )
            .unwrap();
        thread::sleep(Duration::from_millis(1100));
        assert!(matches!(expiring.poll(epoch), Err(Error::Expired)));
        drop(retry);
        pool.drained(second.id).unwrap();
        pool.drained(healthy.id).unwrap();
        let mut filled = Vec::new();
        for n in 0..32 {
            let distinct: NodeId = format!("00000000-0000-4000-8000-{:012x}", 100 + n)
                .parse()
                .unwrap();
            filled.push(
                pool.submit(
                    epoch,
                    distinct,
                    &BTreeSet::new(),
                    pool.deadline().unwrap(),
                    starlings_protocol::MAX_JSON_BYTES,
                )
                .unwrap(),
            );
        }
        assert!(matches!(
            pool.submit(epoch, client, &BTreeSet::new(), pool.deadline().unwrap(), 1),
            Err(Error::Capacity)
        ));
        drop(filled);
        assert_eq!(pool.state.lock().unwrap().requests.len(), 0);
    }
}
pub struct Pool {
    config: CoordinatorRoute,
    workers: Vec<CoordinatorReplica>,
    state: Mutex<State>,
    origin: Instant,
}
pub struct Selected {
    pub id: RequestId,
    pub config: CoordinatorReplica,
    pub owner: Owner,
}
pub struct Pending<'a> {
    pool: &'a Pool,
    id: RequestId,
    claimed: bool,
}
impl Pending<'_> {
    pub fn poll(&mut self, epoch: AuthorityEpoch) -> Result<Option<Selected>> {
        let selected = self.pool.poll(epoch, self.id)?;
        self.claimed |= selected.is_some();
        Ok(selected)
    }
}
impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if !self.claimed {
            self.pool.abort_unclaimed(self.id);
        }
    }
}
impl Pool {
    pub fn new(config: CoordinatorRoute) -> Self {
        Self {
            workers: config.workers(),
            state: Mutex::new(State {
                epoch: None,
                router: TaskRouter::new(config.plan.cluster_id, FixtureMode::Reject),
                samples: BTreeMap::new(),
                owners: BTreeMap::new(),
                requests: BTreeMap::new(),
                excluded: BTreeMap::new(),
                claimed: BTreeSet::new(),
            }),
            config,
            origin: Instant::now(),
        }
    }
    pub fn workers(&self) -> &[CoordinatorReplica] {
        &self.workers
    }
    pub fn status(&self) -> serde_json::Value {
        let Ok(state) = self.state.lock() else {
            return serde_json::json!({"state":"unavailable"});
        };
        serde_json::json!({"queued":state.router.queued(), "active":state.router.assignments().count(), "retained_requests":state.requests.len(), "retained_bytes":state.requests.values().sum::<usize>().to_string()})
    }
    pub fn refresh(
        &self,
        epoch: AuthorityEpoch,
        views: &[(Option<ResourceView>, Instant)],
    ) -> Result<()> {
        if views.len() != self.workers.len() {
            return Err(Error::Invalid);
        }
        let mut state = self.state.lock().map_err(|_| Error::Denied)?;
        let now: u64 = self
            .origin
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| Error::Expired)?;
        if state.epoch != Some(epoch) {
            if state.router.assignments().next().is_some() {
                for config in &self.workers {
                    state.router.withdraw(config.qualification.replica.node);
                }
                state.epoch = None;
                return Err(Error::Expired);
            }
            state.router = TaskRouter::new(self.config.plan.cluster_id, FixtureMode::Reject);
            state.samples.clear();
            state.owners.clear();
            state.requests.clear();
            state.excluded.clear();
            state.claimed.clear();
            state.epoch = Some(epoch);
        }
        for (config, (view, captured)) in self.workers.iter().zip(views) {
            let q = &config.qualification.replica;
            let age = view.as_ref().map(|v| {
                u64::from(v.age_ms).saturating_add(
                    captured
                        .elapsed()
                        .as_millis()
                        .try_into()
                        .unwrap_or(u64::MAX),
                )
            });
            let view = view.as_ref().filter(|v| {
                age.is_some_and(|n| n < 5000)
                    && v.sample.capabilities.node_id == q.node
                    && v.sample.capabilities.control_endpoint
                        == format!("tls://{}", config.worker_endpoint)
            });
            let Some(view) = view else {
                state.router.withdraw(q.node);
                continue;
            };
            let mut node = view.sample.capabilities.clone();
            let sample = node.snapshot_id.to_string();
            if state.samples.get(&q.node) != Some(&sample) {
                node.sample_age_ms =
                    starlings_protocol::BoundedU32::new(age.ok_or(Error::Expired)? as u32)
                        .map_err(|_| Error::Invalid)?;
                if state.router.observe(node, now).is_err() {
                    state.router.withdraw(q.node);
                    continue;
                }
                state.samples.insert(q.node, sample);
            }
            // Outstanding/uncertain work owns its qualification and reservation until drain.
            if !state.router.assignments().any(|a| a.node == q.node) {
                state.router.qualify(q.clone()).map_err(|_| Error::Denied)?;
            }
        }
        state.router.advance(now).map_err(|_| Error::Expired)?;
        Ok(())
    }
    pub fn deadline(&self) -> Result<u64> {
        let now: u64 = self
            .origin
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| Error::Expired)?;
        now.checked_add(60_000).ok_or(Error::Expired)
    }
    pub fn submit(
        &self,
        epoch: AuthorityEpoch,
        client: NodeId,
        excluded: &BTreeSet<NodeId>,
        deadline: u64,
        bytes: usize,
    ) -> Result<Pending<'_>> {
        let mut state = self.state.lock().map_err(|_| Error::Denied)?;
        let now: u64 = self
            .origin
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| Error::Expired)?;
        if state.epoch != Some(epoch) {
            return Err(Error::Expired);
        }
        if bytes > starlings_protocol::MAX_JSON_BYTES
            || state.requests.len() >= 128
            || state
                .requests
                .values()
                .try_fold(bytes, |sum, n| sum.checked_add(*n))
                .is_none_or(|n| n > MAX_RETAINED_BYTES)
        {
            return Err(Error::Capacity);
        }
        let id: RequestId = uuid::Uuid::new_v4()
            .to_string()
            .parse()
            .map_err(|_| Error::Invalid)?;
        let q = &self.config.qualification.replica;
        state
            .router
            .enqueue(
                Task {
                    id,
                    client,
                    manifest: q.manifest.clone(),
                    profile: q.profile.clone(),
                    context_tokens: q.context_limit,
                    deadline_ms: deadline,
                },
                now,
            )
            .map_err(|_| Error::Capacity)?;
        state.requests.insert(id, bytes);
        state.excluded.insert(id, excluded.clone());
        Ok(Pending {
            pool: self,
            id,
            claimed: false,
        })
    }
    fn poll(&self, epoch: AuthorityEpoch, id: RequestId) -> Result<Option<Selected>> {
        let mut state = self.state.lock().map_err(|_| Error::Denied)?;
        let now = self
            .origin
            .elapsed()
            .as_millis()
            .try_into()
            .map_err(|_| Error::Expired)?;
        if state.epoch != Some(epoch) {
            return Err(Error::Expired);
        }
        state.router.advance(now).map_err(|_| Error::Expired)?;
        for expired in state.router.take_expired() {
            state.requests.remove(&expired.id);
            state.excluded.remove(&expired.id);
        }
        if !state.requests.contains_key(&id) {
            return Err(Error::Expired);
        }
        for _ in 0..self.workers.len() {
            let State {
                router, excluded, ..
            } = &mut *state;
            if router
                .next_excluding(now, excluded)
                .map_err(|_| Error::Capacity)?
                .is_none()
            {
                break;
            }
        }
        let Some(assignment) = state
            .router
            .assignments()
            .find(|a| a.task.id == id)
            .cloned()
        else {
            return Ok(None);
        };
        if assignment.cancelled {
            return Err(Error::Expired);
        }
        let config = self
            .workers
            .iter()
            .find(|w| w.qualification.replica.node == assignment.node)
            .ok_or(Error::Denied)?
            .clone();
        let owner = state
            .owners
            .entry(assignment.node)
            .or_insert_with(|| Arc::new(Mutex::new(None)))
            .clone();
        state.claimed.insert(id);
        Ok(Some(Selected { id, config, owner }))
    }
    #[cfg(all(test, feature = "test-backend"))]
    fn select(
        &self,
        epoch: AuthorityEpoch,
        client: NodeId,
        excluded: &BTreeSet<NodeId>,
        deadline: u64,
    ) -> Result<Selected> {
        self.submit(epoch, client, excluded, deadline, 1)?
            .poll(epoch)?
            .ok_or(Error::Capacity)
    }
    fn abort_unclaimed(&self, id: RequestId) {
        if let Ok(mut state) = self.state.lock()
            && !state.claimed.contains(&id)
        {
            state.router.cancel(id);
            if state.router.assignments().any(|a| a.task.id == id) {
                let _ = state.router.drained(id);
            }
            state.requests.remove(&id);
            state.excluded.remove(&id);
        }
    }
    pub fn cancelled(&self, id: RequestId) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return true;
        };
        let Ok(now) = self.origin.elapsed().as_millis().try_into() else {
            return true;
        };
        state.router.advance(now).is_err()
            || state
                .router
                .assignments()
                .find(|a| a.task.id == id)
                .is_none_or(|a| a.cancelled)
    }
    pub fn invalidate(&self) {
        if let Ok(mut state) = self.state.lock() {
            for config in &self.workers {
                state.router.withdraw(config.qualification.replica.node);
            }
        }
    }
    /// Only before sending any worker command, or after the validated request-drain reply.
    pub fn drained(&self, id: RequestId) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| Error::Denied)?;
        state.router.drained(id).map_err(|_| Error::Denied)?;
        state.requests.remove(&id);
        state.excluded.remove(&id);
        state.claimed.remove(&id);
        Ok(())
    }
    pub fn failed(&self, id: RequestId, node: NodeId) {
        if let Ok(mut state) = self.state.lock() {
            state.router.cancel(id);
            state.router.withdraw(node);
        }
    }
}
