//! Bounded client measurements against explicitly selected, already-running loopback services.
use crate::scoreboard::{self, Artifact, Matrix, Metric, Observation, Outcome, System};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use starlings_protocol::{
    ChatRequest, Digest, FixtureMode, Version, byte_digest, content_hash, decode, parse_json,
};
use std::{
    collections::BTreeSet,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    net::SocketAddr,
    path::{Component, Path},
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, &'static str>;
const MAX_RESPONSE: usize = 16 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    #[serde(rename = "version")]
    _version: Version,
    fixture: bool,
    repeats: u8,
    timeout_ms: u32,
    budget_ms: u32,
    exo: Endpoint,
    glyd: Endpoint,
    cells: Vec<Selection>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Endpoint {
    address: SocketAddr,
    model: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    id: String,
    request_path: String,
    cancel_after_content: bool,
}
/// Redacted workload receipt, referenced by the existing matrix's workload digest.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Workload {
    #[serde(rename = "version")]
    _version: Version,
    request_sha256: Digest,
    expected_prompt_tokens: u32,
    concurrency: u8,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {
    #[serde(rename = "version")]
    _version: Version,
    metric: Metric,
    boundary: String,
}
#[derive(Serialize)]
struct RequestRecord {
    request_sha256: Digest,
    attempted: bool,
    http_status: Option<u16>,
    code: &'static str,
    prompt_tokens: Option<u64>,
    output_tokens: Option<u64>,
    first_content_ms: Option<f64>,
    duration_ms: f64,
}
#[derive(Serialize)]
struct Trial {
    version: Version,
    fixture: bool,
    plan_sha256: Digest,
    collector_sha256: Digest,
    started_at: String,
    match_digest: Digest,
    program_digest: Digest,
    system: System,
    repeat: u8,
    order: u8,
    elapsed_ms: f64,
    requests: Vec<RequestRecord>,
}
struct Prepared {
    index: usize,
    request: ChatRequest,
    workload: Workload,
    cancel: bool,
}
fn json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    serde_json::from_value(parse_json(bytes).map_err(|_| "COLLECT_INVALID_JSON")?)
        .map_err(|_| "COLLECT_INVALID_RECORD")
}
fn file(path: &Path) -> Result<Vec<u8>> {
    scoreboard::read(
        File::open(path).map_err(|_| "COLLECT_READ_FAILED")?,
        starlings_protocol::MAX_JSON_BYTES,
    )
}
fn local(root: &Path, name: &str) -> Result<std::path::PathBuf> {
    if name.is_empty()
        || name.len() > 256
        || Path::new(name)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err("COLLECT_RELATIVE_PATH_REQUIRED");
    }
    let path = root
        .join(name)
        .canonicalize()
        .map_err(|_| "COLLECT_READ_FAILED")?;
    if !path.starts_with(root) || !path.is_file() {
        return Err("COLLECT_LOCAL_FILE_REQUIRED");
    }
    Ok(path)
}
fn receipt<T: serde::de::DeserializeOwned>(
    matrix: &Matrix,
    root: &Path,
    digest: &Digest,
) -> Result<T> {
    let a = matrix
        .artifacts
        .iter()
        .find(|a| &a.sha256 == digest)
        .ok_or("COLLECT_RECEIPT_REQUIRED")?;
    let bytes = file(&local(root, &a.path)?)?;
    if &byte_digest(&bytes) != digest {
        return Err("COLLECT_RECEIPT_CHANGED");
    }
    json(&bytes)
}

/// Output is a NEW matrix beside the input; its NEW .raw.jsonl receipt retains every attempted trial.
/// No service is started, retried, reset or qualified. Dropping HTTP is not proof of backend drain.
pub fn collect_file(
    matrix_path: &Path,
    plan_path: &Path,
    output: &Path,
    mode: FixtureMode,
) -> Result<scoreboard::Report> {
    let matrix_path = matrix_path
        .canonicalize()
        .map_err(|_| "COLLECT_READ_FAILED")?;
    let root = matrix_path.parent().ok_or("COLLECT_READ_FAILED")?;
    let mut matrix: Matrix = json(&file(&matrix_path)?)?;
    scoreboard::compare(&matrix, root, mode)?;
    let plan_bytes = file(plan_path)?;
    let plan: Plan = json(&plan_bytes)?;
    let plan_digest = byte_digest(&plan_bytes);
    if plan.fixture != matrix.fixture || (plan.fixture && mode == FixtureMode::Reject) {
        return Err("COLLECT_FIXTURE_REFUSED");
    }
    if !(5..=8).contains(&plan.repeats)
        || !(1..=30_000).contains(&plan.timeout_ms)
        || !(1..=300_000).contains(&plan.budget_ms)
        || plan.cells.is_empty()
        || plan.cells.len() > 16
    {
        return Err("COLLECT_BUDGET_REFUSED");
    }
    for endpoint in [&plan.exo, &plan.glyd] {
        if !endpoint.address.ip().is_loopback()
            || endpoint.address.port() == 0
            || endpoint.model.is_empty()
            || endpoint.model.len() > 256
            || endpoint.model.chars().any(char::is_control)
        {
            return Err("COLLECT_LOOPBACK_ENDPOINT_REQUIRED");
        }
    }
    let programs = [matrix.programs.exo.as_ref(), matrix.programs.glyd.as_ref()];
    if programs.iter().any(|p| p.is_none()) {
        return Err("COLLECT_PROGRAM_REQUIRED");
    }
    let mut seen = BTreeSet::new();
    let mut prepared = Vec::new();
    let mut count = 0usize;
    for selection in &plan.cells {
        if !seen.insert(&selection.id) {
            return Err("COLLECT_DUPLICATE_CELL");
        }
        let index = matrix
            .cells
            .iter()
            .position(|c| c.id == selection.id)
            .ok_or("COLLECT_UNKNOWN_CELL")?;
        let cell = &matrix.cells[index];
        if !cell.observations.is_empty() {
            return Err("COLLECT_EXISTING_OBSERVATIONS");
        }
        let matched = cell.matched.as_ref().ok_or("COLLECT_MATCH_REQUIRED")?;
        let workload: Workload = receipt(&matrix, root, &matched.workload)?;
        let definition: Definition = receipt(&matrix, root, &matched.definition)?;
        let boundary = match cell.metric {
            Metric::TtftMilliseconds => "client_first_nonempty_content_or_reasoning_delta",
            Metric::ConcurrentTokensPerSecond => "client_batch_usage_tokens_over_wall_seconds",
            _ => return Err("COLLECT_METRIC_UNSUPPORTED"),
        };
        if definition.metric != cell.metric || definition.boundary != boundary {
            return Err("COLLECT_DEFINITION_MISMATCH");
        }
        let bytes = file(&local(root, &selection.request_path)?)?;
        let request: ChatRequest =
            decode(&bytes, FixtureMode::Reject).map_err(|_| "COLLECT_REQUEST_REFUSED")?;
        if byte_digest(&bytes) != workload.request_sha256
            || !(1..=16).contains(&workload.concurrency)
            || !(1..=1_048_576).contains(&workload.expected_prompt_tokens)
            || request.temperature.is_none()
            || request.top_p.is_none()
            || request.seed.is_none()
            || request.stream.is_none()
            || (selection.cancel_after_content && request.stream != Some(true))
            || (cell.metric == Metric::TtftMilliseconds
                && (request.stream != Some(true) || workload.concurrency != 1))
        {
            return Err("COLLECT_WORKLOAD_MISMATCH");
        }
        count += usize::from(workload.concurrency) * usize::from(plan.repeats) * 2;
        prepared.push(Prepared {
            index,
            request,
            workload,
            cancel: selection.cancel_after_content,
        });
    }
    if count > 256
        || matrix
            .cells
            .iter()
            .map(|c| c.observations.len())
            .sum::<usize>()
            + prepared.len() * usize::from(plan.repeats) * 2
            > 512
        || matrix.artifacts.len() >= 512
    {
        return Err("COLLECT_REQUEST_LIMIT");
    }
    // The output shares the matrix evidence root, so existing receipts need no copying or path rewriting.
    if output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()
        .map_err(|_| "COLLECT_OUTPUT_REFUSED")?
        != root
        || output.extension().is_none_or(|e| e != "json")
    {
        return Err("COLLECT_OUTPUT_REFUSED");
    }
    let raw_path = output.with_extension("raw.jsonl");
    let raw_name = raw_path
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| n.len() <= 256)
        .ok_or("COLLECT_OUTPUT_REFUSED")?;
    if raw_name.chars().any(char::is_control) {
        return Err("COLLECT_OUTPUT_REFUSED");
    }
    // Load credentials before creating output; fixed environment names, never part of records/errors.
    let tokens = [
        token("STARLINGS_SCOREBOARD_EXO_TOKEN")?,
        token("STARLINGS_SCOREBOARD_GLYD_TOKEN")?,
    ];
    let collector_digest = starlings_adapters::service::executable_digest(
        &std::env::current_exe().map_err(|_| "COLLECT_EXECUTABLE_UNAVAILABLE")?,
    )
    .map_err(|_| "COLLECT_EXECUTABLE_UNAVAILABLE")?;
    let mut raw = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&raw_path)
        .map_err(|_| "COLLECT_OUTPUT_EXISTS_OR_REFUSED")?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|_| "COLLECT_OUTPUT_EXISTS_OR_REFUSED")?;
    let deadline = Instant::now() + Duration::from_millis(plan.budget_ms.into());
    let mut stopped = false;
    for p in prepared {
        let cell = &mut matrix.cells[p.index];
        cell.unavailable = None;
        for repeat in 1..=plan.repeats {
            let systems = if repeat % 2 == 1 {
                [System::Exo, System::Glyd]
            } else {
                [System::Glyd, System::Exo]
            };
            for (order, system) in systems.into_iter().enumerate() {
                let (endpoint, program, key) = match system {
                    System::Exo => (&plan.exo, matrix.programs.exo.as_ref().unwrap(), &tokens[0]),
                    System::Glyd => (
                        &plan.glyd,
                        matrix.programs.glyd.as_ref().unwrap(),
                        &tokens[1],
                    ),
                };
                let mut body =
                    serde_json::to_value(&p.request).map_err(|_| "COLLECT_REQUEST_REFUSED")?;
                body["model"] = endpoint.model.clone().into();
                if p.request.stream == Some(true) {
                    body["stream_options"] = serde_json::json!({"include_usage":true});
                }
                let body = serde_json::to_vec(&body).map_err(|_| "COLLECT_REQUEST_REFUSED")?;
                let started_at = time::OffsetDateTime::now_utc()
                    .format(&time::format_description::well_known::Rfc3339)
                    .map_err(|_| "COLLECT_CLOCK_UNAVAILABLE")?;
                let began = Instant::now();
                let attempt = Attempt {
                    stopped,
                    stream_object: stream_object(system, &program.source_commit),
                    endpoint,
                    key: key.as_deref(),
                    body: &body,
                    chat: &p.request,
                    workload: &p.workload,
                    cancel: p.cancel,
                    deadline,
                    timeout_ms: plan.timeout_ms,
                };
                let requests = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..p.workload.concurrency)
                        .map(|_| scope.spawn(|| attempt.run()))
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().expect("collector thread panicked"))
                        .collect::<Vec<_>>()
                });
                let elapsed = began.elapsed();
                let failure = requests
                    .iter()
                    .find(|r| r.code != "SUCCESS")
                    .map(|r| r.code);
                stopped |= failure.is_some();
                let outcome = if let Some(code) = failure {
                    Outcome::Failed { code: code.into() }
                } else {
                    let value = match cell.metric {
                        Metric::TtftMilliseconds => requests[0].first_content_ms.unwrap(),
                        _ => {
                            requests
                                .iter()
                                .map(|r| r.output_tokens.unwrap())
                                .sum::<u64>() as f64
                                / elapsed.as_secs_f64()
                        }
                    };
                    Outcome::Measured { value }
                };
                let binding = cell.binding()?;
                let program_digest =
                    content_hash(&(system, program)).map_err(|_| "COLLECT_BINDING_FAILED")?;
                let trial = Trial {
                    version: Version,
                    fixture: plan.fixture,
                    plan_sha256: plan_digest.clone(),
                    collector_sha256: collector_digest.clone(),
                    started_at,
                    match_digest: binding.clone(),
                    program_digest: program_digest.clone(),
                    system,
                    repeat,
                    order: order as u8,
                    elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                    requests,
                };
                serde_json::to_writer(&mut raw, &trial).map_err(|_| "COLLECT_WRITE_FAILED")?;
                raw.write_all(b"\n")
                    .and_then(|_| raw.sync_data())
                    .map_err(|_| "COLLECT_WRITE_FAILED")?;
                cell.observations.push(Observation {
                    fixture: plan.fixture,
                    id: uuid::Uuid::new_v4().to_string().parse().unwrap(),
                    system,
                    repeat,
                    program_digest,
                    match_digest: binding,
                    unit: cell.metric.unit().into(),
                    raw: plan_digest.clone(),
                    outcome,
                });
            }
        }
    }
    raw.sync_all().map_err(|_| "COLLECT_WRITE_FAILED")?;
    let raw_digest = byte_digest(&scoreboard::read(
        File::open(&raw_path).map_err(|_| "COLLECT_READ_FAILED")?,
        MAX_RESPONSE,
    )?);
    for cell in &mut matrix.cells {
        if seen.contains(&cell.id) {
            for observation in &mut cell.observations {
                observation.raw = raw_digest.clone();
            }
        }
    }
    matrix.artifacts.push(Artifact {
        path: raw_name.into(),
        sha256: raw_digest,
    });
    let report = scoreboard::compare(&matrix, root, mode)?;
    serde_json::to_writer_pretty(&mut out, &matrix).map_err(|_| "COLLECT_WRITE_FAILED")?;
    out.sync_all().map_err(|_| "COLLECT_WRITE_FAILED")?;
    Ok(report)
}
fn token(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(s) if !s.is_empty() && s.len() <= 4096 && s.bytes().all(|b| b.is_ascii_graphic()) => {
            Ok(Some(s))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        _ => Err("COLLECT_CREDENTIAL_REFUSED"),
    }
}

fn stream_object(system: System, source_commit: &str) -> &'static str {
    if system == System::Exo && source_commit == "bc412e2230f395e652c23426a5eb88dfa515769b" {
        "chat.completion"
    } else {
        "chat.completion.chunk"
    }
}

struct Attempt<'a> {
    stopped: bool,
    stream_object: &'static str,
    endpoint: &'a Endpoint,
    key: Option<&'a str>,
    body: &'a [u8],
    chat: &'a ChatRequest,
    workload: &'a Workload,
    cancel: bool,
    deadline: Instant,
    timeout_ms: u32,
}
impl Attempt<'_> {
    fn run(&self) -> RequestRecord {
        let began = Instant::now();
        let mut record = RequestRecord {
            request_sha256: byte_digest(self.body),
            attempted: false,
            http_status: None,
            code: "BUDGET_EXHAUSTED",
            prompt_tokens: None,
            output_tokens: None,
            first_content_ms: None,
            duration_ms: 0.0,
        };
        if self.stopped {
            record.code = "NOT_ATTEMPTED_AFTER_FAILURE";
        } else if let Some(remaining) = self.deadline.checked_duration_since(began) {
            record.attempted = true;
            let result = self.receive(
                began,
                remaining.min(Duration::from_millis(self.timeout_ms.into())),
                &mut record,
            );
            record.code = result.err().unwrap_or("SUCCESS");
        }
        record.duration_ms = began.elapsed().as_secs_f64() * 1000.0;
        record
    }
    fn receive(&self, began: Instant, timeout: Duration, record: &mut RequestRecord) -> Result<()> {
        let Self {
            endpoint,
            key,
            body,
            chat,
            cancel,
            ..
        } = self;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .proxy(None)
            .max_redirects(0)
            .http_status_as_error(false)
            .max_response_header_size(16 * 1024)
            .timeout_global(Some(timeout))
            .build()
            .into();
        let mut req = agent
            .post(format!("http://{}/v1/chat/completions", endpoint.address))
            .header("Content-Type", "application/json")
            .header("Accept-Encoding", "identity");
        if let Some(key) = key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }
        let mut response = req.send(*body).map_err(|e| match e {
            ureq::Error::Timeout(_) => "TIMEOUT",
            _ => "TRANSPORT",
        })?;
        let status = response.status().as_u16();
        record.http_status = Some(status);
        if status != 200 {
            return Err("HTTP_STATUS");
        }
        let stream = chat.stream == Some(true);
        let expected = if stream {
            "text/event-stream"
        } else {
            "application/json"
        };
        if response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(';').next())
            .map(str::trim)
            != Some(expected)
            || response
                .headers()
                .get("content-encoding")
                .is_some_and(|v| v != "identity")
        {
            return Err("CONTENT_TYPE");
        }
        let mut output = Output::default();
        if !stream {
            let mut bytes = Vec::new();
            response
                .body_mut()
                .as_reader()
                .take(starlings_protocol::MAX_JSON_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|_| "BODY_READ")?;
            if bytes.len() > starlings_protocol::MAX_JSON_BYTES {
                return Err("BYTE_LIMIT");
            }
            output.event(
                &parse_json(&bytes).map_err(|_| "INVALID_JSON")?,
                self,
                false,
                began,
                record,
            )?;
            return if output.finished && record.output_tokens.is_some() {
                Ok(())
            } else {
                Err("INCOMPLETE")
            };
        }
        let mut reader = BufReader::new(response.body_mut().as_reader());
        let mut total = 0;
        let mut data = Vec::new();
        loop {
            let mut line = Vec::new();
            let n = reader
                .by_ref()
                .take(65537)
                .read_until(b'\n', &mut line)
                .map_err(|_| "BODY_READ")?;
            total += n;
            if n > 65536 || total > MAX_RESPONSE {
                return Err("BYTE_LIMIT");
            }
            if n == 0 || line.pop() != Some(b'\n') {
                return Err("INCOMPLETE");
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                if data.is_empty() {
                    continue;
                }
                data.pop();
                if data == b"[DONE]" {
                    return if output.finished
                        && record.output_tokens.is_some()
                        && record.first_content_ms.is_some()
                    {
                        Ok(())
                    } else {
                        Err("INCOMPLETE")
                    };
                }
                output.event(
                    &parse_json(&data).map_err(|_| "INVALID_JSON")?,
                    self,
                    true,
                    began,
                    record,
                )?;
                if *cancel && record.first_content_ms.is_some() {
                    return Err("CANCELLED");
                }
                data.clear();
            } else if let Some(part) = line.strip_prefix(b"data:") {
                data.extend_from_slice(part.strip_prefix(b" ").unwrap_or(part));
                data.push(b'\n');
                if data.len() > 65536 {
                    return Err("BYTE_LIMIT");
                }
            } else if !line.starts_with(b":") {
                return Err("UNSUPPORTED_EVENT");
            }
        }
    }
}
#[derive(Default)]
struct Output {
    id: Option<String>,
    finished: bool,
}
impl Output {
    fn event(
        &mut self,
        value: &Value,
        attempt: &Attempt<'_>,
        stream: bool,
        began: Instant,
        record: &mut RequestRecord,
    ) -> Result<()> {
        let Attempt {
            endpoint,
            chat,
            workload,
            ..
        } = attempt;
        let id = value["id"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 256)
            .ok_or("INVALID_RESPONSE")?;
        if self.id.as_ref().is_some_and(|old| old != id)
            || record.output_tokens.is_some()
            || value["model"] != endpoint.model
            || value["object"]
                != if stream {
                    attempt.stream_object
                } else {
                    "chat.completion"
                }
        {
            return Err("INVALID_RESPONSE");
        }
        self.id = Some(id.into());
        let choices = value["choices"].as_array().ok_or("INVALID_RESPONSE")?;
        if choices.len() > 1 || (!stream && choices.len() != 1) {
            return Err("INVALID_RESPONSE");
        }
        if let Some(choice) = choices.first() {
            if self.finished || choice["index"] != 0 {
                return Err("INVALID_RESPONSE");
            }
            let message = choice[if stream { "delta" } else { "message" }]
                .as_object()
                .ok_or("INVALID_RESPONSE")?;
            if message
                .keys()
                .any(|k| !matches!(k.as_str(), "role" | "content" | "reasoning_content"))
                || message.get("role").is_some_and(|v| v != "assistant")
                || ["content", "reasoning_content"].iter().any(|k| {
                    message
                        .get(*k)
                        .is_some_and(|v| !v.is_null() && !v.is_string())
                })
            {
                return Err("INVALID_RESPONSE");
            }
            if stream
                && record.first_content_ms.is_none()
                && ["content", "reasoning_content"].iter().any(|k| {
                    message
                        .get(*k)
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                })
            {
                record.first_content_ms = Some(began.elapsed().as_secs_f64() * 1000.0);
            }
            match choice["finish_reason"].as_str() {
                Some("stop" | "length") => self.finished = true,
                None if stream && choice["finish_reason"].is_null() => (),
                _ => return Err("INVALID_RESPONSE"),
            }
        }
        if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
            let prompt = usage["prompt_tokens"].as_u64().ok_or("INVALID_USAGE")?;
            let completion = usage["completion_tokens"].as_u64().ok_or("INVALID_USAGE")?;
            if !self.finished
                || prompt != u64::from(workload.expected_prompt_tokens)
                || completion == 0
                || completion > u64::from(chat.max_tokens.get())
                || prompt.checked_add(completion) != usage["total_tokens"].as_u64()
            {
                return Err("INVALID_USAGE");
            }
            record.prompt_tokens = Some(prompt);
            record.output_tokens = Some(completion);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
