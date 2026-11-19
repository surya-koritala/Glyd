//! Approved service receipts and request ownership; neither is native drain evidence.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlydIdentity {
    pub build: GlydBuild,
    pub model: GlydModel,
    pub level: GlydLevel,
    pub precision: GlydPrecision,
    pub context_tokens: u32,
    pub concurrency: u32,
    pub request_fields: Digest,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlydBuild {
    pub version: String,
    pub commit: Option<String>,
    pub development: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlydModel {
    pub id: String,
    pub repo: Option<String>,
    pub revision: Option<String>,
    pub source: Option<GlydSource>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlydSource {
    pub repo: Option<String>,
    pub revision: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlydLevel {
    pub requested: Option<String>,
    pub effective: String,
    pub lossless: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlydPrecision {
    pub weights: String,
    pub activations: String,
    pub sums: String,
    pub kv: Option<String>,
    pub state: Option<String>,
}
impl GlydIdentity {
    pub(super) fn valid_for(&self, config: &ServiceConfig) -> bool {
        fn text(s: &str) -> bool {
            !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control)
        }
        fn revision(s: &str) -> bool {
            s.len() == 40
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        self.build.version == config.backend_version
            && self.model.id == config.backend_model
            && self.concurrency == 1
            && (1..=2_000_000).contains(&self.context_tokens)
            && self.build.commit.as_deref().is_none_or(revision)
            && self.model.revision.as_deref().is_none_or(revision)
            && self.model.source.as_ref().is_none_or(|s| {
                s.repo.as_deref().is_none_or(text) && s.revision.as_deref().is_none_or(revision)
            })
            && self.model.repo.as_deref().is_none_or(text)
            && self.level.requested.as_deref().is_none_or(text)
            && self.precision.kv.as_deref().is_none_or(text)
            && self.precision.state.as_deref().is_none_or(text)
            && [
                &self.build.version,
                &self.model.id,
                &self.level.effective,
                &self.precision.weights,
                &self.precision.activations,
                &self.precision.sums,
            ]
            .iter()
            .all(|s| text(s))
    }
    pub(super) fn from_info(value: &serde_json::Value) -> Result<Self> {
        if value["object"] != "glyd.info"
            || value["api"] != "glyd-serve/1"
            || value["state"] != "ready"
            || value.get("error") != Some(&serde_json::Value::Null)
            || value["requests"]["running"].as_u64() != Some(0)
            || value["requests"]["cancelling"].as_u64() != Some(0)
        {
            return Err(Error::Unapproved);
        }
        // Missing nullable identity is not an explicitly reported null.
        for path in [
            "/build/commit",
            "/model/repo",
            "/model/revision",
            "/model/source",
            "/level/requested",
            "/precision/kv",
            "/precision/state",
        ] {
            value.pointer(path).ok_or(Error::Invalid)?;
        }
        if !value["model"]["source"].is_null() {
            for field in ["repo", "revision"] {
                value["model"]["source"].get(field).ok_or(Error::Invalid)?;
            }
        }
        let fields = value["fields"].as_object().ok_or(Error::Invalid)?;
        if fields.len() != 3 {
            return Err(Error::Invalid);
        }
        for name in ["read", "no_effect", "refused"] {
            let names = fields
                .get(name)
                .and_then(serde_json::Value::as_array)
                .ok_or(Error::Invalid)?;
            if names.len() > 128
                || names.iter().any(|v| {
                    v.as_str().is_none_or(|s| {
                        s.is_empty() || s.len() > 128 || !s.bytes().all(|b| b.is_ascii_graphic())
                    })
                })
            {
                return Err(Error::Invalid);
            }
        }
        serde_json::from_value(serde_json::json!({
            "build": value["build"],
            "model": {"id":value["model"]["id"], "repo":value["model"]["repo"],
                "revision":value["model"]["revision"], "source":value["model"]["source"]},
            "level": value["level"], "precision":value["precision"],
            "context_tokens":value["limits"]["context_tokens"],
            "concurrency":value["limits"]["concurrency"],
            "request_fields":content_hash(&value["fields"]).map_err(|_| Error::Invalid)?,
        }))
        .map_err(|_| Error::Invalid)
    }
}

/// One adapter-generated ID in one owned process. A retained handle cannot target its successor.
#[derive(Clone)]
pub struct RequestCancellation {
    id: String,
    agent: ureq::Agent,
    endpoint: SocketAddr,
    key: String,
    process: Cancellation,
    state: Arc<Mutex<RequestState>>,
}
#[derive(Default)]
struct RequestState {
    cancelled: bool,
    dispatched: bool,
    ended: Option<bool>,
    cancel_pending: bool,
}
impl RequestState {
    fn settled(&self) -> bool {
        self.ended.is_some() && !self.cancel_pending
    }
}
impl RequestCancellation {
    pub(super) fn new(service: &Service) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            agent: service.agent.clone(),
            endpoint: service.config.endpoint,
            key: service.key.clone(),
            process: service.control.clone(),
            state: Arc::new(Mutex::new(RequestState::default())),
        }
    }
    pub(super) fn body_id(&self) -> &str {
        self.id
            .strip_prefix("chatcmpl-")
            .expect("locally constructed request ID")
    }
    pub(super) fn id(&self) -> &str {
        &self.id
    }
    pub(super) fn dispatch(&self) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        if state.cancelled {
            return Err(Error::Cancelled);
        }
        state.dispatched = true;
        Ok(())
    }
    pub(super) fn cancelled_and_ended(&self) -> bool {
        self.state.lock().is_ok_and(|s| s.cancelled && s.settled())
    }
    pub(super) fn check(&self) -> Result<()> {
        if self.state.lock().map_err(|_| Error::Unavailable)?.cancelled {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
    pub(super) fn is_cancelled(&self) -> bool {
        self.state.lock().is_ok_and(|s| s.cancelled)
    }
    /// Withdraw this request, then ask only its owned ID to end. Unconfirmed cancellation stops the service.
    /// An `ended` reply is a service receipt, not permission to free GPU reservations.
    pub fn cancel(&self) -> Result<()> {
        // Serialize cancellation intent against dispatch; never hold this lock across I/O.
        let first = {
            let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
            let first = !state.cancelled;
            state.cancelled = true;
            if !state.dispatched {
                state.ended = Some(false);
            }
            if state.settled() {
                return Ok(());
            }
            if first {
                state.cancel_pending = true;
            }
            first
        };
        let result = if first {
            self.wait_ended(true).map(|_| ())
        } else {
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                if self.state.lock().map_err(|_| Error::Unavailable)?.settled() {
                    break Ok(());
                }
                if self.process.cancelled.load(Ordering::Acquire) || Instant::now() >= deadline {
                    break Err(Error::StopUnconfirmed);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        if result.is_err() {
            let _ = self.process.cancel();
        }
        if first {
            // A normal status reply can arrive while cancellation I/O is still pending.
            // Its late failure must withdraw this service before anyone can reuse it.
            let settled = self
                .state
                .lock()
                .map(|mut state| state.cancel_pending = false)
                .map_err(|_| Error::Unavailable);
            if settled.is_err() {
                let _ = self.process.cancel();
            }
            settled?;
        }
        result
    }
    pub(super) fn finish(&self) -> Result<()> {
        let ended = self.state.lock().map_err(|_| Error::Unavailable)?.ended;
        let finished = match ended {
            Some(finished) => finished,
            None => self.wait_ended(false)?,
        };
        self.check()?;
        if finished {
            Ok(())
        } else {
            Err(Error::Cancelled)
        }
    }
    fn wait_ended(&self, cancel: bool) -> Result<bool> {
        let deadline = Instant::now() + Duration::from_secs(2);
        if cancel {
            let mut response = self
                .agent
                .post(format!(
                    "http://{}/v1/glyd/requests/{}/cancel",
                    self.endpoint, self.id
                ))
                .header("Authorization", format!("Bearer {}", self.key))
                .config()
                .timeout_global(Some(Duration::from_secs(2)))
                .build()
                .send_empty()
                .map_err(|_| Error::Transport)?;
            check_response(&response, "application/json")?;
            let value = read_json(response.body_mut().as_reader(), 4096)?;
            let _ = request_status(value, &self.id)?;
        }
        loop {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero())
                .ok_or(Error::StopUnconfirmed)?;
            let mut response = self
                .agent
                .get(format!(
                    "http://{}/v1/glyd/requests/{}",
                    self.endpoint, self.id
                ))
                .header("Authorization", format!("Bearer {}", self.key))
                .config()
                .timeout_global(Some(remaining))
                .build()
                .call()
                .map_err(|_| Error::Transport)?;
            check_response(&response, "application/json")?;
            if let Some(finished) =
                request_status(read_json(response.body_mut().as_reader(), 4096)?, &self.id)?
            {
                self.state.lock().map_err(|_| Error::Unavailable)?.ended = Some(finished);
                return Ok(finished);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

fn request_status(value: serde_json::Value, id: &str) -> Result<Option<bool>> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Status {
        id: String,
        state: String,
        cancel: Option<String>,
        began: f64,
        ended: Option<f64>,
        how: Option<String>,
    }
    for field in ["cancel", "ended", "how"] {
        value.get(field).ok_or(Error::Invalid)?;
    }
    let status: Status = serde_json::from_value(value).map_err(|_| Error::Invalid)?;
    if status.id != id
        || status.began < 0.0
        || status
            .cancel
            .as_deref()
            .is_some_and(|s| !matches!(s, "asked" | "client gone"))
    {
        return Err(Error::Invalid);
    }
    match (status.state.as_str(), status.ended, status.how.as_deref()) {
        ("running", None, None) if status.cancel.is_none() => Ok(None),
        ("cancelling", None, None) if status.cancel.is_some() => Ok(None),
        ("ended", Some(end), Some(how)) if end >= status.began => match how {
            "finished" => Ok(Some(status.cancel.is_none())),
            "cancelled" => Ok(Some(false)),
            _ => Err(Error::Unavailable),
        },
        _ => Err(Error::Invalid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_end_does_not_settle_an_outstanding_cancel() {
        let mut state = RequestState::default();
        assert!(!state.settled());
        state.dispatched = true;
        state.cancelled = true;
        state.cancel_pending = true;
        // Either normal finish/status or a second cancel may observe this first.
        for finished in [true, false] {
            state.ended = Some(finished);
            assert!(!state.settled());
        }
        // Only the cancellation owner, after any failure withdrawal, can settle it.
        state.cancel_pending = false;
        assert!(state.settled());
        // Pre-dispatch and already-completed old IDs still require no cancellation I/O.
        state.dispatched = false;
        assert!(state.settled());
        state.cancelled = false;
        state.dispatched = true;
        assert!(state.settled());
    }
    #[test]
    fn approved_identity_and_owned_status_fail_closed() {
        let value = super::super::fixture::glyd_info();
        let expected = GlydIdentity::from_info(&value).unwrap();
        let original = serde_json::json!({"executable":"/not-opened","executable_sha256":starlings_protocol::byte_digest(b"fixture"),
            "arguments":[],"directory":"/not-opened","environment":{},"key_environment":"KEY",
            "endpoint":"127.0.0.1:1","backend_version":"synthetic-v1","api":"glyd-serve/1",
            "glyd_identity":expected,"manifest_digest":starlings_protocol::byte_digest(b"manifest"),
            "execution_profile":"synthetic-f32","backend_model":"synthetic-backend","reasoning_content":true,"request_timeout_ms":1000});
        let config: ServiceConfig = serde_json::from_value(original).unwrap();
        assert!(expected.valid_for(&config));
        assert!(
            !serde_json::to_string(&expected)
                .unwrap()
                .contains("not-retained")
        );
        for (path, changed) in [
            ("/build/version", serde_json::json!("other")),
            ("/model/id", serde_json::json!("other")),
            ("/level/effective", serde_json::json!("other")),
            ("/level/lossless", serde_json::json!(false)),
            ("/precision/weights", serde_json::json!("other")),
            ("/limits/context_tokens", serde_json::json!(513)),
            ("/fields/read", serde_json::json!(["different"])),
        ] {
            let mut changed_value = value.clone();
            *changed_value.pointer_mut(path).unwrap() = changed;
            assert_ne!(GlydIdentity::from_info(&changed_value).unwrap(), expected);
        }
        assert!(GlydIdentity::from_info(&serde_json::Value::Null).is_err());
        for (path, replacement) in [
            ("/state", serde_json::json!("failed")),
            ("/requests/running", serde_json::json!(1)),
            ("/fields/read", serde_json::json!([null])),
            ("/fields/read", serde_json::json!({})),
            ("/precision/weights", serde_json::json!(null)),
        ] {
            let mut invalid = value.clone();
            *invalid.pointer_mut(path).unwrap() = replacement;
            assert!(GlydIdentity::from_info(&invalid).is_err());
        }
        let mut missing = value.clone();
        missing["build"].as_object_mut().unwrap().remove("commit");
        assert!(GlydIdentity::from_info(&missing).is_err());
        let mut missing = value.clone();
        missing["model"]["source"] = serde_json::json!({});
        assert!(GlydIdentity::from_info(&missing).is_err());
        let mut bad = expected.clone();
        bad.concurrency = 2;
        assert!(!bad.valid_for(&config));
        let mut bad = expected.clone();
        bad.precision.weights.clear();
        assert!(!bad.valid_for(&config));
        let mut bad = expected.clone();
        bad.build.commit = Some("not-a-commit".into());
        assert!(!bad.valid_for(&config));
        let status = serde_json::json!({"id":"chatcmpl-owned","state":"ended","cancel":null,"began":1.0,"ended":2.0,"how":"finished"});
        assert_eq!(
            request_status(status.clone(), "chatcmpl-owned"),
            Ok(Some(true))
        );
        assert!(request_status(status.clone(), "chatcmpl-other").is_err());
        let mut cancelled = status.clone();
        cancelled["how"] = "cancelled".into();
        cancelled["cancel"] = "asked".into();
        assert_eq!(request_status(cancelled, "chatcmpl-owned"), Ok(Some(false)));
        for (key, value) in [
            ("state", serde_json::json!("running")),
            ("ended", serde_json::json!(0.0)),
            ("how", serde_json::json!("failed: private details")),
            ("cancel", serde_json::json!("other")),
            ("extra", serde_json::json!(true)),
            ("id", serde_json::Value::Null),
        ] {
            let mut invalid = status.clone();
            invalid[key] = value;
            assert!(request_status(invalid, "chatcmpl-owned").is_err());
        }
        let mut missing = status;
        missing.as_object_mut().unwrap().remove("cancel");
        assert!(request_status(missing, "chatcmpl-owned").is_err());
    }
}
