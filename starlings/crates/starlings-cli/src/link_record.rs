//! Typed private profile snapshots; no probe activation, retry or placement authority.
use crate::AdminReply;
use serde::{Deserialize, Serialize};
use starlings_protocol::{Decimal, Version, parse_json};
use starlings_security::{
    AuthorityEpoch,
    probes::{DirectedMeasurement, ProbeLimits, TokenProfileReport},
};
use std::net::SocketAddr;
use uuid::Uuid;

const INVALID: &str = "TOKEN_PROFILE_SNAPSHOT_REFUSED";
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum State {
    Pending,
    Failed,
    Unavailable,
    ProfileComplete,
}
// Projection intentionally discards root errors, interface names and future unrelated fields.
#[derive(Deserialize, Serialize)]
pub struct Snapshot {
    state: State,
    setup_error: Option<starlings_security::Error>,
    profile_limits: ProbeLimits,
    profile_attempt: Option<Attempt>,
}
#[derive(Deserialize, Serialize)]
struct Attempt {
    report: Report,
    age_ms: Option<Decimal>,
    admission: bool,
}
#[derive(Deserialize, Serialize)]
struct Report {
    version: Version,
    id: Uuid,
    epoch: AuthorityEpoch,
    local_address: SocketAddr,
    peer_address: SocketAddr,
    rtt_mean_us: Option<u32>,
    rtt_variance_us2: Option<Decimal>,
    inter_ping_gap_ms: u32,
    transfers: [DirectedMeasurement; 2],
    total_application_bytes: Decimal,
    token_profile: TokenProfileReport,
}
/// Project one bounded owner-IPC reply; success means a snapshot, not a successful experiment.
pub fn project(reply: &AdminReply) -> Result<Snapshot, &'static str> {
    if !reply.ok || reply.code != "LINKS" || reply.message.len() > 16 * 1024 {
        return Err(INVALID);
    }
    let result: Snapshot =
        serde_json::from_value(parse_json(reply.message.as_bytes()).map_err(|_| INVALID)?)
            .map_err(|_| INVALID)?;
    result.profile_limits.validate().map_err(|_| INVALID)?;
    let limits = &result.profile_limits;
    let config = limits.token_profile.as_ref().ok_or(INVALID)?;
    if let Some(attempt) = &result.profile_attempt {
        let report = &attempt.report;
        let p = &report.token_profile;
        if attempt.admission
            || p.native_qualified
            || !p.local_tcp_nodelay
            || p.requested_samples != limits.ping_samples
            || p.completed_samples > p.requested_samples
            || p.samples.len() > p.requested_samples as usize
            || p.keepalive_frames > 2000
            || p.gap_basis != config.gap_basis
            || p.keepalive_ms != config.keepalive_ms
            || report.inter_ping_gap_ms != limits.inter_ping_gap_ms
            || report.total_application_bytes.0 > 2 * 1024 * 1024
            || (p.rtt_p99_us.is_some() && p.completed_samples < 100)
            || (p.error.is_none() && p.completed_samples != p.requested_samples)
            || report.transfers.iter().any(|m| {
                !matches!(
                    m.timing_basis.as_str(),
                    "token-profile-client-window" | "token-profile-server-window"
                )
            })
        {
            return Err(INVALID);
        }
        let client = report.transfers[0].timing_basis == "token-profile-client-window";
        if report.transfers[1].timing_basis != report.transfers[0].timing_basis
            || p.error.is_some() != p.failure_phase.is_some()
            || report.transfers[0].payload_bytes != p.completed_samples * 16384
            || report.transfers[1].payload_bytes != p.completed_samples * 64
            || (client
                && p.samples.iter().filter(|s| s.rtt_us.is_some()).count()
                    != p.completed_samples as usize)
            || (!client && !p.samples.is_empty())
        {
            return Err(INVALID);
        }
        for (i, sample) in p.samples.iter().enumerate() {
            if sample.sequence as usize != i
                || sample.rtt_us == Some(0)
                || sample.keepalive_frames > 2000
                || (sample.rtt_us.is_some()
                    && (sample.started_us.is_none() || sample.error.is_some()))
                || (sample.error.is_some() && (p.error.is_none() || i + 1 != p.samples.len()))
            {
                return Err(INVALID);
            }
        }
        if matches!(result.state, State::ProfileComplete) && p.error.is_some() {
            return Err(INVALID);
        }
    } else if matches!(result.state, State::ProfileComplete) {
        return Err(INVALID);
    }
    if matches!(result.state, State::Failed)
        && result.setup_error.is_none()
        && result
            .profile_attempt
            .as_ref()
            .is_none_or(|a| a.report.token_profile.error.is_none())
    {
        return Err(INVALID);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn completed() -> serde_json::Value {
        let id = "00000000-0000-4000-8000-000000000001";
        let samples: Vec<_> = (0..5).map(|n| serde_json::json!({"sequence":n,"started_us":(n*20000).to_string(),"rtt_us":1000,"keepalive_frames":0,"error":null})).collect();
        let transfer = |bytes| serde_json::json!({"source_node":id,"destination_node":id,"payload_bytes":bytes,"elapsed_us":"81000","timing_basis":"token-profile-client-window","acknowledged_bytes_per_second":"1000"});
        serde_json::json!({"state":"profile_complete","setup_error":null,"profile_limits":{"version":1,"payload_bytes":16384,"ping_samples":5,"inter_ping_gap_ms":20,"minimum_interval_ms":30000,"token_profile":{"gap_basis":"start_to_start","keepalive_ms":null}},"profile_attempt":{"age_ms":"1","admission":false,"report":{"version":1,"id":id,"epoch":{"number":"1","incarnation":id},"local_address":"127.0.0.1:10001","peer_address":"127.0.0.1:10002","rtt_mean_us":1000,"rtt_variance_us2":"0","inter_ping_gap_ms":20,"transfers":[transfer(81920),transfer(320)],"total_application_bytes":"83000","token_profile":{"version":1,"producer_sha256":"0000000000000000000000000000000000000000000000000000000000000000","requested_samples":5,"completed_samples":5,"samples":samples,"gap_basis":"start_to_start","keepalive_ms":null,"keepalive_frames":0,"elapsed_us":"81000","started_at_unix_ms":"1","rtt_p50_us":1000,"rtt_p99_us":null,"error":null,"failure_phase":null,"local_tcp_nodelay":true,"native_qualified":false}}}})
    }
    #[test]
    fn complete_failed_and_stale_snapshots_preserve_diagnostics_without_authority() {
        let mut value = completed();
        assert!(project(&AdminReply::ok("LINKS", value.to_string())).is_ok());
        value["state"] = "unavailable".into();
        value["profile_attempt"]["age_ms"] = serde_json::Value::Null;
        assert!(project(&AdminReply::ok("LINKS", value.to_string())).is_ok());
        value["state"] = "failed".into();
        let p = &mut value["profile_attempt"]["report"]["token_profile"];
        p["error"] = "Transport".into();
        p["failure_phase"] = "final_receipt".into();
        let result =
            serde_json::to_value(project(&AdminReply::ok("LINKS", value.to_string())).unwrap())
                .unwrap();
        assert_eq!(
            result["profile_attempt"]["report"]["token_profile"]["samples"]
                .as_array()
                .unwrap()
                .len(),
            5
        );
        assert_eq!(result["state"], "failed");
        assert_eq!(result["profile_attempt"]["admission"], false);
        assert_eq!(
            result["profile_attempt"]["report"]["token_profile"]["native_qualified"],
            false
        );
    }
    #[test]
    fn impossible_or_private_profile_claims_are_refused() {
        for (pointer, replacement) in [
            ("/profile_attempt/admission", serde_json::json!(true)),
            (
                "/profile_attempt/report/token_profile/native_qualified",
                serde_json::json!(true),
            ),
            (
                "/profile_attempt/report/token_profile/local_tcp_nodelay",
                serde_json::json!(false),
            ),
            (
                "/profile_attempt/report/token_profile/completed_samples",
                serde_json::json!(4),
            ),
            (
                "/profile_attempt/report/token_profile/rtt_p99_us",
                serde_json::json!(1000),
            ),
            (
                "/profile_attempt/report/token_profile/samples/0/rtt_us",
                serde_json::json!(0),
            ),
            (
                "/profile_attempt/report/token_profile/samples/0/error",
                serde_json::json!("private credential"),
            ),
            (
                "/profile_attempt/report/total_application_bytes",
                serde_json::json!("2097153"),
            ),
        ] {
            let mut value = completed();
            *value.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                project(&AdminReply::ok("LINKS", value.to_string())).is_err(),
                "{pointer}"
            );
        }
        let mut reply = AdminReply::ok("LINKS", completed().to_string());
        reply.message = reply
            .message
            .replacen("\"state\":", "\"state\":\"failed\",\"state\":", 1);
        assert!(project(&reply).is_err());
    }
    #[test]
    fn pending_and_initial_failure_snapshots_keep_status_and_drop_unrelated_private_text() {
        for state in ["pending", "failed", "unavailable"] {
            let message = serde_json::json!({"state":state,"profile_limits":{"version":1,"payload_bytes":16384,"ping_samples":5,"inter_ping_gap_ms":20,"minimum_interval_ms":30000,"token_profile":{"gap_basis":"start_to_start","keepalive_ms":null}},"profile_attempt":null,"setup_error":if state=="failed" {serde_json::json!("Transport")} else {serde_json::Value::Null},"error":"private raw error","prompt":"private prompt","token":"private token"});
            let mut reply = AdminReply::ok("LINKS", message.to_string());
            let value = serde_json::to_value(project(&reply).unwrap()).unwrap();
            assert_eq!(value["state"], state);
            assert!(value["profile_attempt"].is_null());
            assert!(!value.to_string().contains("private"));
            reply.message = reply.message.replace(state, "profile_complete");
            assert!(project(&reply).is_err());
        }
    }
}
