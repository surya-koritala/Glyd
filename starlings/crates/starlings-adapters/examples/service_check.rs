//! Explicit native API smoke check; never run by routine CI. No downloads or result content logging.
use serde_json::{Value, json};
use starlings_adapters::service::{Service, ServiceConfig};
use starlings_protocol::{
    BoundedU32, ChatMessage, ChatMessageRole, ChatRequest, Digest, parse_json,
};
use std::{
    io::{Read, Write},
    time::{Duration, Instant},
};

fn bounded(reader: impl Read, limit: u64) -> Result<Vec<u8>, &'static str> {
    let mut bytes = Vec::new();
    reader
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "READ_FAILED")?;
    if bytes.len() as u64 > limit {
        return Err("RESPONSE_LIMIT");
    }
    Ok(bytes)
}
fn semantic(value: &Value) -> Value {
    json!({"content":value["choices"][0]["message"]["content"].as_str().unwrap_or(""),
        "reasoning_content":value["choices"][0]["message"]["reasoning_content"].as_str().unwrap_or(""),
        "finish_reason":value["choices"][0]["finish_reason"],"usage":value["usage"]})
}
fn run() -> Result<(), &'static str> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let cancel = args.len() == 3 && args[2] == "--cancel-after-text";
    if args.len() != 2 && !cancel {
        return Err(
            "USAGE: service_check APPROVED_CONFIG_JSON CONFIG_DIGEST [--cancel-after-text]",
        );
    }
    let bytes = bounded(
        std::fs::File::open(&args[0]).map_err(|_| "CONFIG_UNAVAILABLE")?,
        64 * 1024,
    )?;
    let config: ServiceConfig =
        serde_json::from_value(parse_json(&bytes).map_err(|_| "INVALID_CONFIG")?)
            .map_err(|_| "INVALID_CONFIG")?;
    let approved = Digest::try_from(args[1].to_str().ok_or("INVALID_DIGEST")?.to_owned())
        .map_err(|_| "INVALID_DIGEST")?;
    let key = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let mut service = Service::spawn(config.clone(), &approved, key.clone())
        .map_err(|_| "BACKEND_START_REFUSED")?;
    let ready_until = Instant::now() + Duration::from_secs(60);
    while service.probe().is_err() {
        if Instant::now() >= ready_until {
            return Err("BACKEND_NOT_READY");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let mut request = ChatRequest {
        model: config.manifest_digest.to_string(),
        messages: vec![ChatMessage {
            role: ChatMessageRole::User,
            content: "Reply with the single word ready.".into(),
        }],
        max_tokens: BoundedU32::new(8).map_err(|_| "INVALID_LIMIT")?,
        temperature: Some(0.0),
        top_p: Some(1.0),
        seed: Some(BoundedU32::new(1).map_err(|_| "INVALID_SEED")?),
        stream: Some(false),
    };
    if cancel {
        request.stream = Some(true);
        request.messages[0].content = "Count upward from one, one number per line.".into();
        let control = service.cancellation();
        let mut cancelled_at = None;
        let result = service.generate(&request, |bytes| {
            let value =
                parse_json(bytes).map_err(|_| starlings_adapters::service::Error::Invalid)?;
            if unfinished_text(&value) {
                cancelled_at = Some(Instant::now());
                control.cancel()?;
                return Err(starlings_adapters::service::Error::Cancelled);
            }
            Ok(())
        });
        service.stop().map_err(|_| "STOP_UNCONFIRMED")?;
        let elapsed = cancelled_at.ok_or("NO_ACTIVE_TEXT_TO_CANCEL")?.elapsed();
        if result != Err(starlings_adapters::service::Error::Cancelled) {
            return Err("CANCELLATION_NOT_OBSERVED");
        }
        if elapsed > Duration::from_secs(3) {
            return Err("CANCELLATION_DEADLINE");
        }
        let record = json!({"kind":"NativeCancellationSmokeOnly","configuration_digest":approved,
            "executable_sha256":config.executable_sha256,"declared_backend_version":config.backend_version,
            "declared_manifest_digest":config.manifest_digest,"declared_execution_profile":config.execution_profile,
            "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,
            "unfinished_text_observed":true,"request_cancelled":true,"parent_exit_observed":true,
            "cancel_to_parent_exit_ms":elapsed.as_millis(),"device_drain_qualified":false,"memory_qualified":false});
        serde_json::to_writer(std::io::stdout().lock(), &record).map_err(|_| "OUTPUT_FAILED")?;
        writeln!(std::io::stdout()).map_err(|_| "OUTPUT_FAILED")?;
        return Ok(());
    }
    let mut direct = request.clone();
    direct.model.clone_from(&config.backend_model);
    let direct_body = serde_json::to_vec(&direct).map_err(|_| "INVALID_REQUEST")?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .max_response_header_size(16 * 1024)
        .timeout_global(Some(Duration::from_millis(u64::from(
            config.request_timeout_ms,
        ))))
        .build()
        .into();
    let start = Instant::now();
    let mut response = agent
        .post(format!("http://{}/v1/chat/completions", config.endpoint))
        .header("Authorization", format!("Bearer {key}"))
        .header("Content-Type", "application/json")
        .send(&direct_body)
        .map_err(|_| "DIRECT_REQUEST_FAILED")?;
    if response.status().as_u16() != 200 {
        return Err("DIRECT_REQUEST_FAILED");
    }
    let value = parse_json(&bounded(response.body_mut().as_reader(), 1024 * 1024)?)
        .map_err(|_| "INVALID_DIRECT_RESPONSE")?;
    if value["choices"].as_array().is_none_or(|v| v.len() != 1) || value["usage"].is_null() {
        return Err("INVALID_DIRECT_RESPONSE");
    }
    let reference = semantic(&value);
    let direct_ms = start.elapsed().as_millis();
    let mut proxied = None;
    let start = Instant::now();
    service
        .generate(&request, |bytes| {
            proxied = Some(semantic(
                &parse_json(bytes).map_err(|_| starlings_adapters::service::Error::Invalid)?,
            ));
            Ok(())
        })
        .map_err(|_| "PROXY_REQUEST_FAILED")?;
    let proxy_ms = start.elapsed().as_millis();
    if proxied.as_ref() != Some(&reference) {
        return Err("DIRECT_PROXY_MISMATCH");
    }
    let mut stream = request;
    stream.stream = Some(true);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage = Value::Null;
    let mut finish = Value::Null;
    let start = Instant::now();
    service
        .generate(&stream, |bytes| {
            let value =
                parse_json(bytes).map_err(|_| starlings_adapters::service::Error::Invalid)?;
            content.push_str(
                value["choices"][0]["delta"]["content"]
                    .as_str()
                    .unwrap_or(""),
            );
            reasoning.push_str(
                value["choices"][0]["delta"]["reasoning_content"]
                    .as_str()
                    .unwrap_or(""),
            );
            if !value["usage"].is_null() {
                usage = value["usage"].clone();
            }
            if !value["choices"][0]["finish_reason"].is_null() {
                finish = value["choices"][0]["finish_reason"].clone();
            }
            Ok(())
        })
        .map_err(|_| "STREAM_REQUEST_FAILED")?;
    let stream_ms = start.elapsed().as_millis();
    if json!({"content":content,"reasoning_content":reasoning,"usage":usage,"finish_reason":finish})
        != reference
    {
        return Err("STREAM_SEMANTICS_MISMATCH");
    }
    service.stop().map_err(|_| "STOP_UNCONFIRMED")?;
    let record = json!({"kind":"NativeApiSmokeOnly","configuration_digest":approved,"executable_sha256":config.executable_sha256,
        "declared_backend_version":config.backend_version,"declared_manifest_digest":config.manifest_digest,"declared_execution_profile":config.execution_profile,
        "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,
        "direct_ms":direct_ms,"proxy_ms":proxy_ms,"stream_ms":stream_ms,"usage":reference["usage"],
        "exact_semantics":true,"parent_exit_observed":true,"device_drain_qualified":false,"memory_qualified":false});
    serde_json::to_writer(std::io::stdout().lock(), &record).map_err(|_| "OUTPUT_FAILED")?;
    writeln!(std::io::stdout()).map_err(|_| "OUTPUT_FAILED")?;
    Ok(())
}
fn unfinished_text(value: &Value) -> bool {
    let choice = &value["choices"][0];
    choice["finish_reason"].is_null()
        && ["content", "reasoning_content"].iter().any(|field| {
            choice["delta"][field]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        })
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_requires_active_text_not_role_usage_or_completed_output() {
        assert!(!unfinished_text(&json!({"choices":[]})));
        assert!(!unfinished_text(
            &json!({"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]})
        ));
        for field in ["content", "reasoning_content"] {
            let mut event = json!({"choices":[{"delta":{},"finish_reason":null}]});
            event["choices"][0]["delta"][field] = "synthetic text".into();
            assert!(unfinished_text(&event));
            event["choices"][0]["finish_reason"] = "stop".into();
            assert!(!unfinished_text(&event));
        }
    }
}
