//! One admitted client, one enrolled worker, one backend attempt. Never retry output.
use serde::{Deserialize, Serialize};
use starlings_adapters::service::Service;
use starlings_protocol::{
    ChatRequest, ControlRequest, FixtureMode, GenerationId, InvocationId, Operation, RequestId,
};
use starlings_security::{
    Action, ClientAdmission, ClientResponse, EnrolledWorker, Error, GenerationBinding,
    GenerationStream, IncomingClient, Scope,
};
use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

mod dispatch;
pub use dispatch::Dispatch;

type Result<T> = std::result::Result<T, Error>;
const FRAGMENT: usize = 16 * 1024;
const POLL: Duration = Duration::from_millis(20);

#[cfg(all(test, feature = "test-backend"))]
mod tests;

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Count { request: Box<ChatRequest> },
    Apply { request: Box<ControlRequest> },
    Poll { generation: GenerationId },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Reply {
    Counted {
        request: starlings_protocol::Digest,
        tokens: u32,
    },
    Applied {
        request: RequestId,
    },
    Pending,
    Fragment {
        text: String,
        end: bool,
    },
    Finished,
    Rejected,
}

/// Fixed immutable model route. Authentication precedes parsing the chat body.
pub fn admit(
    incoming: IncomingClient,
    admission: &ClientAdmission,
    binding: &GenerationBinding,
) -> Result<(ChatRequest, ClientResponse)> {
    let mut chat = None;
    let (_, response) = incoming.prepare_with_errors(
        admission,
        &Scope {
            model: binding.model.clone(),
            action: Action::Infer,
        },
        |request| {
            if request.method != "POST" || request.path != "/v1/chat/completions" {
                return Err(Error::Invalid);
            }
            let validated: ChatRequest = starlings_protocol::decode(
                &serde_json::to_vec(&request.body)?,
                FixtureMode::Reject,
            )?;
            if validated.model != binding.model.as_str()
                || validated.temperature.is_none()
                || validated.top_p.is_none()
                || validated.seed.is_none()
            {
                return Err(Error::Denied);
            }
            chat = Some(validated);
            Ok(())
        },
    )?;
    Ok((chat.ok_or(Error::Invalid)?, response))
}

/// Count on the selected worker before building its lease/session controls. Keep
/// this connection for relay; no reservation or backend generation starts yet.
pub fn count_prompt(
    response: &ClientResponse,
    chat: &ChatRequest,
    worker: &mut GenerationStream,
) -> Result<u32> {
    let result = (|| {
        let watch = response.watch()?;
        watch.check()?;
        if chat.model != worker.binding().model.as_str() {
            return Err(Error::Denied);
        }
        check_response_scope(response, chat)?;
        starlings_protocol::encode(chat, FixtureMode::Reject)?;
        worker.send_record(&Command::Count {
            request: Box::new(chat.clone()),
        })?;
        let (request, tokens) = match worker.receive_record()? {
            Reply::Counted { request, tokens } => (request, tokens),
            Reply::Rejected => return Err(Error::Invalid),
            _ => return Err(Error::Invalid),
        };
        watch.check()?;
        if request != starlings_protocol::content_hash(chat)? || !(1..=1_048_576).contains(&tokens)
        {
            return Err(Error::Invalid);
        }
        Ok(tokens)
    })();
    if result.is_err() {
        worker.watch().cancel();
    }
    result
}

/// The caller builds these controls from qualified placement and count_prompt on
/// this same connection. The worker checks the count and exact request again.
/// This relay cannot invent token counts, qualification, lease times or sequence IDs.
pub fn relay(
    response: ClientResponse,
    chat: &ChatRequest,
    controls: [ControlRequest; 3],
    mut worker: GenerationStream,
) -> Result<()> {
    check_response_scope(&response, chat)?;
    let [prepare, commit, start] = &controls;
    if !matches!(prepare.operation, Operation::Prepare { .. })
        || !matches!(commit.operation, Operation::Commit { .. })
    {
        return Err(Error::Invalid);
    }
    let Operation::StartRequest {
        session, request, ..
    } = &start.operation
    else {
        return Err(Error::Invalid);
    };
    if request.as_ref() != chat
        || chat.model != worker.binding().model.as_str()
        || session.client_id.to_string() != response.client().to_string()
    {
        return Err(Error::Denied);
    }
    let generation = session.generation_id;
    let session_id = session.session_id;
    let request_id = start.request_id;
    let watch = response.watch()?;
    let admitted = (|| {
        for request in controls {
            request.validate(FixtureMode::Reject)?;
            watch.check()?;
            let id = request.request_id;
            worker.send_record(&Command::Apply {
                request: Box::new(request),
            })?;
            if !matches!(worker.receive_record()?, Reply::Applied { request } if request == id) {
                return Err(Error::Invalid);
            }
            watch.check()?;
        }
        Ok(())
    })();
    if let Err(error) = admitted {
        worker.watch().cancel();
        let _ = response.fail(&failure(error, request_id));
        return Err(error);
    }
    let streaming = chat.stream == Some(true);
    let mut response = Some(response);
    let mut stream = if streaming {
        Some(
            response
                .take()
                .ok_or(Error::Invalid)?
                .stream_for(session_id)?,
        )
    } else {
        None
    };
    let mut event = Vec::new();
    let mut json = None;
    let mut events = 0;
    let result = (|| loop {
        watch.check()?;
        worker.send_record(&Command::Poll { generation })?;
        let reply = worker.receive_record()?;
        watch.check()?;
        match reply {
            Reply::Pending => thread::sleep(POLL),
            Reply::Fragment { text, end } => {
                let bytes = text.as_bytes();
                let limit = if streaming {
                    64 * 1024
                } else {
                    starlings_protocol::MAX_JSON_BYTES
                };
                if json.is_some()
                    || bytes.is_empty()
                    || bytes.len() > FRAGMENT
                    || event.len() + bytes.len() > limit
                {
                    return Err(Error::Capacity);
                }
                event.extend_from_slice(bytes);
                if end {
                    starlings_protocol::parse_json(&event)?;
                    events += 1;
                    if let Some(stream) = &mut stream {
                        stream.event(&event)?;
                        event.clear();
                    } else {
                        json = Some(std::mem::take(&mut event));
                    }
                }
            }
            Reply::Finished => {
                if !event.is_empty() || events == 0 {
                    return Err(Error::Invalid);
                }
                return if let Some(stream) = &mut stream {
                    stream.finish()
                } else {
                    response
                        .take()
                        .ok_or(Error::Invalid)?
                        .json(&json.ok_or(Error::Invalid)?)
                };
            }
            Reply::Applied { .. } | Reply::Counted { .. } | Reply::Rejected => {
                return Err(Error::Invalid);
            }
        }
    })();
    if result.is_err() {
        worker.watch().cancel();
        if let Some(stream) = &mut stream {
            let _ = stream.fail(&failure(Error::Transport, request_id));
        } else if let Some(response) = response {
            let _ = response.fail(&failure(Error::Transport, request_id));
        }
    }
    result
}

pub fn failure(error: Error, request_id: RequestId) -> starlings_protocol::WireError {
    use starlings_protocol::ErrorDetailCode as Code;
    let code = match error {
        Error::Invalid => Code::InvalidRequest,
        Error::Denied => Code::Forbidden,
        Error::Expired => Code::DeadlineExceeded,
        Error::Replay | Error::ReplayExpired => Code::StaleEpoch,
        Error::Capacity | Error::Busy => Code::CapacityUnavailable,
        Error::Transport => Code::WorkerLost,
        _ => Code::Internal,
    };
    starlings_protocol::WireError {
        error: starlings_protocol::ErrorDetail {
            code,
            message: "Generation stopped before completion".into(),
            request_id,
            retryable: false,
        },
    }
}

fn check_response_scope(response: &ClientResponse, chat: &ChatRequest) -> Result<()> {
    if response.scope().action != Action::Infer || response.scope().model.as_str() != chat.model {
        return Err(Error::Denied);
    }
    Ok(())
}

/// Backend completion is not device/process-tree drain. Keep the reservation until
/// a qualified local supervisor confirms cleanup with EnrolledWorker::drained.
pub struct WorkerCompletion {
    pub generation: GenerationId,
    pub invocation: InvocationId,
}

/// Service and profile must already be locally approved and loaded. The worker
/// samples its own monotonic clock under its state lock. No model starts here.
/// None means no Count was accepted and the runtime has no reservations. Close
/// only that connection. Other outcomes retain their cancellation/drain policy;
/// work is never replayed.
pub fn serve_worker(
    service: &mut Service,
    worker: &EnrolledWorker,
    channel: GenerationStream,
) -> Result<Option<WorkerCompletion>> {
    serve_worker_with(service, worker, channel, None, None)
}

/// Apply the approved envelope, complete plan identity and live CPU slot drain.
pub fn serve_worker_qualified(
    service: &mut Service,
    worker: &EnrolledWorker,
    channel: GenerationStream,
    qualification: &crate::routes::Qualification,
) -> Result<Option<WorkerCompletion>> {
    qualification.validate()?;
    serve_worker_with(service, worker, channel, Some(qualification), None)
}
pub fn serve_worker_qualified_guarded(
    service: &mut Service,
    worker: &EnrolledWorker,
    channel: GenerationStream,
    qualification: &crate::routes::Qualification,
    resources: Option<&crate::contribution::ContributionGate>,
) -> Result<Option<WorkerCompletion>> {
    qualification.validate()?;
    serve_worker_with(service, worker, channel, Some(qualification), resources)
}
fn serve_worker_with(
    service: &mut Service,
    worker: &EnrolledWorker,
    mut channel: GenerationStream,
    qualification: Option<&crate::routes::Qualification>,
    resources: Option<&crate::contribution::ContributionGate>,
) -> Result<Option<WorkerCompletion>> {
    let cancel = service.cancellation();
    let mut rejected = false;
    let mut count_accepted = false;
    let mut request_cancel = None;
    let result = (|| {
        let config = service.configuration();
        let profile = worker.profile();
        if config.manifest_digest != profile.manifest
            || config.execution_profile != profile.profile
            || config.executable_sha256 != profile.build
            || config.request_timeout_ms > 60_000
        {
            return Err(Error::Denied);
        }
        let Command::Count { request } = channel.receive_record()? else {
            return Err(Error::Invalid);
        };
        count_accepted = true;
        if qualification.is_some_and(|q| q.admit(request.max_tokens.get()).is_err())
            || resources.is_some_and(|g| g.check().is_err())
        {
            channel.send_record(&Reply::Rejected)?;
            rejected = true;
            return Err(Error::Capacity);
        }
        let request_hash = starlings_protocol::content_hash(request.as_ref())?;
        let prompt_tokens = service.count_prompt(&request).map_err(|_| Error::Invalid)?;
        request_cancel = service.request_cancellation();
        if prompt_tokens
            .checked_add(request.max_tokens.get())
            .is_none_or(|n| n > profile.context_limit)
        {
            channel.send_record(&Reply::Rejected)?;
            rejected = true;
            return Err(Error::Capacity);
        }
        channel.send_record(&Reply::Counted {
            request: request_hash.clone(),
            tokens: prompt_tokens,
        })?;
        drop(request);
        let mut started = None;
        for phase in 0..3 {
            let Command::Apply { request } = channel.receive_record()? else {
                return Err(Error::Invalid);
            };
            let expected = match phase {
                0 => matches!(request.operation, Operation::Prepare { .. }),
                1 => matches!(request.operation, Operation::Commit { .. }),
                _ => matches!(request.operation, Operation::StartRequest { .. }),
            };
            if let (Some(q), Operation::Prepare { plan, .. }) = (qualification, &request.operation)
            {
                q.check_plan(plan)?;
            }
            if let Operation::StartRequest {
                session, request, ..
            } = &request.operation
                && (session.prompt_token_count.get() != prompt_tokens
                    || starlings_protocol::content_hash(request.as_ref())? != request_hash)
            {
                return Err(Error::Denied);
            }
            if !expected {
                return Err(Error::Invalid);
            }
            let applied = || worker.apply(&channel, &request);
            let replay = if let Some(gate) = resources {
                gate.with_eligible(applied)?
            } else {
                applied()?
            };
            if replay {
                return Err(Error::Replay);
            }
            channel.send_record(&Reply::Applied {
                request: request.request_id,
            })?;
            if let Operation::StartRequest {
                session,
                invocation_id,
                request,
            } = request.operation
            {
                started = Some((session.generation_id, invocation_id, request));
            }
        }
        let (generation, invocation, request) = started.ok_or(Error::Invalid)?;
        thread::scope(|threads| {
            let (sender, receiver) = mpsc::sync_channel(1);
            let watch = channel.watch();
            let producer_watch = watch.clone();
            let producer = threads.spawn(move || {
                service.generate(&request, |bytes| {
                    let text = std::str::from_utf8(bytes)
                        .map_err(|_| starlings_adapters::service::Error::Invalid)?;
                    let mut offset = 0;
                    while offset < text.len() {
                        let end = text.floor_char_boundary((offset + FRAGMENT).min(text.len()));
                        let mut reply = Reply::Fragment {
                            text: text[offset..end].into(),
                            end: end == text.len(),
                        };
                        let deadline = Instant::now() + Duration::from_secs(2);
                        loop {
                            if producer_watch.check().is_err() || Instant::now() >= deadline {
                                return Err(starlings_adapters::service::Error::Cancelled);
                            }
                            match sender.try_send(reply) {
                                Ok(()) => break,
                                Err(mpsc::TrySendError::Full(value)) => reply = value,
                                Err(_) => {
                                    return Err(starlings_adapters::service::Error::Cancelled);
                                }
                            }
                            thread::sleep(Duration::from_millis(1));
                        }
                        offset = end;
                    }
                    Ok(())
                })?;
                if qualification.is_some() {
                    service.confirm_request_idle()?;
                }
                Ok::<(), starlings_adapters::service::Error>(())
            });
            let mut producer = Some(producer);
            let result = (|| {
                loop {
                    let Command::Poll {
                        generation: requested,
                    } = channel.receive_record()?
                    else {
                        return Err(Error::Invalid);
                    };
                    if requested != generation {
                        return Err(Error::Denied);
                    }
                    if let Some(gate) = resources {
                        gate.maintain()?;
                    }
                    // Take only the short local queue operation under the output fence.
                    let event = worker.with_output(&channel, generation, || receiver.try_recv())?;
                    let reply = match event {
                        Ok(reply) => reply,
                        Err(mpsc::TryRecvError::Empty) => Reply::Pending,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            producer
                                .take()
                                .ok_or(Error::Invalid)?
                                .join()
                                .map_err(|_| Error::Transport)?
                                .map_err(|_| Error::Transport)?;
                            channel.send_record(&Reply::Finished)?;
                            return Ok(WorkerCompletion {
                                generation,
                                invocation,
                            });
                        }
                    };
                    channel.send_record(&reply)?;
                }
            })();
            if result.is_err() {
                watch.cancel();
                if let Some(control) = &request_cancel {
                    let _ = control.cancel();
                } else {
                    let _ = cancel.cancel();
                }
            }
            drop(receiver);
            // Scoped join observes the backend result before the runtime can be reused.
            if let Some(producer) = producer {
                let _ = producer.join();
            }
            result
        })
    })();
    if !count_accepted
        && matches!(
            result,
            Err(Error::Transport | Error::Invalid | Error::Capacity)
        )
        && worker.snapshot().is_ok_and(|s| s.reserved_bytes.0 == 0)
    {
        channel.watch().cancel();
        return Ok(None);
    }
    if result.is_err() && !rejected {
        if let Some(control) = &request_cancel {
            let _ = control.cancel();
        } else {
            let _ = cancel.cancel();
        }
    }
    channel.watch().cancel();
    let withdrawn = worker.withdraw(&channel);
    match result {
        Ok(done) => {
            withdrawn?;
            Ok(Some(done))
        }
        Err(error) => Err(error),
    }
}
