//! One serial owner per worker/epoch; native placement qualification is supplied locally.
use super::{Result, count_prompt, relay};
use starlings_core::routing::WarmReplica;
use starlings_protocol::*;
use starlings_security::{ClientResponse, Error, GenerationBinding, GenerationStream};
use std::str::FromStr;

/// Keep this owner for the entire worker/epoch lifetime. Reconstruct only after a
/// new durable coordinator epoch; reconnects must not reset control sequences.
pub struct Dispatch {
    binding: GenerationBinding,
    plan: PlacementPlan,
    lifetime_ms: u32,
    sequence: u64,
    mutation_possible: bool,
}
impl Dispatch {
    pub fn new(
        binding: GenerationBinding,
        plan: PlacementPlan,
        profile: &WarmReplica,
        lifetime_ms: u32,
    ) -> Result<Self> {
        starlings_protocol::encode(&plan, FixtureMode::Reject)?;
        profile.validate().map_err(|_| Error::Invalid)?;
        let [stage] = plan.stages.as_slice() else {
            return Err(Error::Denied);
        };
        if plan.mode != PlacementPlanMode::WholeRequest
            || plan.authority_epoch != binding.epoch.number
            || binding.epoch.incarnation.is_nil()
            || plan.manifest_digest != binding.model
            || plan.execution_profile != binding.profile
            || profile.manifest != binding.model
            || profile.profile != binding.profile
            || plan.max_sessions.get() != 1
            || profile.max_sessions != 1
            || !profile.accepting
            || plan.context_limit.get() > profile.context_limit
            || stage.node_id != profile.node
            || stage.backend_id != profile.backend
            || stage.backend_build_digest != profile.build
            || stage.memory_pool_id != profile.pool
            || stage.reserved_bytes()? < profile.request_bytes
            || !(1000..=60_000).contains(&lifetime_ms)
        {
            return Err(Error::Denied);
        }
        Ok(Self {
            binding,
            plan,
            lifetime_ms,
            sequence: 0,
            mutation_possible: false,
        })
    }

    /// Exact counting and controls share one authenticated connection and mutable
    /// owner. No retry, migration, native qualification or resource release is inferred.
    pub fn run(
        &mut self,
        response: ClientResponse,
        chat: &ChatRequest,
        mut worker: GenerationStream,
    ) -> Result<()> {
        self.mutation_possible = false;
        let prepared = (|| {
            if worker.binding() != &self.binding {
                return Err(Error::Denied);
            }
            let count = count_prompt(&response, chat, &mut worker)?;
            self.controls(response.client(), chat, count)
        })();
        let controls = match prepared {
            Ok(controls) => controls,
            Err(error) => {
                worker.watch().cancel();
                let _ = response.fail(&super::failure(error, id()?));
                return Err(error);
            }
        };
        self.mutation_possible = true;
        relay(response, chat, controls, worker)
    }
    /// Inspect under this serial owner; false proves no mutating control was attempted.
    pub fn mutation_possible(&self) -> bool {
        self.mutation_possible
    }

    fn controls(
        &mut self,
        client: NodeId,
        chat: &ChatRequest,
        count: u32,
    ) -> Result<[ControlRequest; 3]> {
        // Consume IDs even on failure; gaps are legal, reuse is not.
        let first = self.sequence.checked_add(1).ok_or(Error::Capacity)?;
        self.sequence = self.sequence.checked_add(3).ok_or(Error::Capacity)?;
        let now = time::OffsetDateTime::now_utc();
        let format = time::format_description::well_known::Rfc3339;
        let mut plan = self.plan.clone();
        plan.plan_id = id()?;
        plan.created_at = now.format(&format).map_err(|_| Error::Invalid)?;
        plan.expires_at = (now + time::Duration::milliseconds(self.lifetime_ms.into()))
            .format(&format)
            .map_err(|_| Error::Invalid)?;
        plan.plan_digest = canonical_digest(&serde_json::to_value(&plan)?, "plan_digest")?;
        let lease = Lease {
            kind: "Lease".into(),
            protocol_version: Version,
            lease_id: id()?,
            cluster_id: plan.cluster_id,
            plan_digest: plan.plan_digest.clone(),
            authority_epoch: plan.authority_epoch,
            owner_node_id: plan.stages[0].node_id,
            issued_at: plan.created_at.clone(),
            lifetime_ms: BoundedU32::new(self.lifetime_ms).map_err(|_| Error::Invalid)?,
            max_sessions: plan.max_sessions,
            reserved_bytes: Decimal(plan.stages[0].reserved_bytes()?),
        };
        let session = Session {
            kind: "Session".into(),
            protocol_version: Version,
            session_id: id()?,
            generation_id: id()?,
            client_id: client.to_string().parse().map_err(|_| Error::Invalid)?,
            lease_id: lease.lease_id,
            plan_digest: plan.plan_digest.clone(),
            authority_epoch: plan.authority_epoch,
            prompt_token_count: BoundedU32::new(count).map_err(|_| Error::Invalid)?,
            max_output_tokens: chat.max_tokens,
            context_limit: plan.context_limit,
            sampling: Sampling {
                profile: "sample-v1".into(),
                temperature: chat.temperature.ok_or(Error::Invalid)?,
                top_p: chat.top_p.ok_or(Error::Invalid)?,
                seed: chat.seed.ok_or(Error::Invalid)?,
            },
            state: SessionState::Queued,
        };
        if chat.model != self.binding.model.as_str() {
            return Err(Error::Denied);
        }
        let operations = [
            Operation::Prepare {
                plan: Box::new(plan),
                lease: Box::new(lease.clone()),
            },
            Operation::Commit {
                lease_id: lease.lease_id,
            },
            Operation::StartRequest {
                session: Box::new(session),
                invocation_id: id()?,
                request: Box::new(chat.clone()),
            },
        ];
        let mut sequence = first;
        let [a, b, c]: [Result<ControlRequest>; 3] = operations.map(|operation| {
            let request =
                ControlRequest::new(id()?, self.binding.epoch.number.0, sequence, operation)?;
            sequence = sequence.saturating_add(1);
            request.validate(FixtureMode::Reject)?;
            Ok(request)
        });
        Ok([a?, b?, c?])
    }
}
fn id<T: FromStr>() -> Result<T> {
    uuid::Uuid::new_v4()
        .to_string()
        .parse()
        .map_err(|_| Error::Invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_bind_client_count_and_unique_sequences_even_after_failure() {
        let mut plan: PlacementPlan = decode(
            include_bytes!("../../../../examples/placement-plan.json"),
            FixtureMode::Simulation,
        )
        .unwrap();
        plan.fixture = false;
        plan.mode = PlacementPlanMode::WholeRequest;
        plan.manifest_digest = byte_digest(b"synthetic model");
        plan.stages.truncate(1);
        plan.stages[0].layer_end_exclusive = plan.model_layer_count;
        plan.stages[0].owns_final_norm_and_head = true;
        plan.stages[0].backend_build_digest = byte_digest(b"synthetic backend");
        plan.plan_digest =
            canonical_digest(&serde_json::to_value(&plan).unwrap(), "plan_digest").unwrap();
        let stage = &plan.stages[0];
        let profile = WarmReplica {
            node: stage.node_id,
            manifest: plan.manifest_digest.clone(),
            profile: plan.execution_profile.clone(),
            backend: stage.backend_id.clone(),
            build: stage.backend_build_digest.clone(),
            pool: stage.memory_pool_id.clone(),
            evidence: byte_digest(b"synthetic qualification"),
            request_bytes: stage.reserved_bytes().unwrap(),
            measured_request_ms: 1,
            context_limit: plan.context_limit.get(),
            max_sessions: 1,
            accepting: true,
        };
        let binding = GenerationBinding {
            version: Version,
            epoch: starlings_security::AuthorityEpoch {
                number: plan.authority_epoch,
                incarnation: uuid::Uuid::new_v4(),
            },
            model: plan.manifest_digest.clone(),
            profile: plan.execution_profile.clone(),
        };
        let mut chat: ChatRequest = decode(
            include_bytes!("../../../../examples/chat-request.json"),
            FixtureMode::Reject,
        )
        .unwrap();
        chat.model = plan.manifest_digest.to_string();
        let mut dispatch = Dispatch::new(binding.clone(), plan.clone(), &profile, 5000).unwrap();
        assert!(Dispatch::new(binding, plan, &profile, 60001).is_err());
        let client = id().unwrap();
        let first = dispatch.controls(client, &chat, 17).unwrap();
        assert!(dispatch.controls(client, &chat, 2048).is_err());
        let next = dispatch.controls(client, &chat, 17).unwrap();
        assert_eq!(first.each_ref().map(|r| r.request_sequence.0), [1, 2, 3]);
        assert_eq!(next.each_ref().map(|r| r.request_sequence.0), [7, 8, 9]);
        let Operation::StartRequest { session, .. } = &next[2].operation else {
            panic!("start")
        };
        assert_eq!(session.client_id.to_string(), client.to_string());
        assert_eq!(session.prompt_token_count.get(), 17);
        assert_ne!(first[0].request_id, next[0].request_id);
        dispatch.sequence = u64::MAX - 2;
        assert!(dispatch.controls(client, &chat, 17).is_err());
    }
}
