//! Opt-in projection of existing local reports; never retain raw reports or discover hosts.
use crate::AdminReply;
use serde::{Deserialize, Serialize};
use starlings_protocol::{Decimal, Version, parse_json};
use starlings_security::{NodeStatus, NodeSummary, Role};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::Path,
};

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Section<T> {
    Available { data: T },
    Unavailable { action: &'static str },
    InvalidReport { action: &'static str },
}
#[derive(Serialize)]
pub struct Bundle {
    version: Version,
    program_version: &'static str,
    os: &'static str,
    architecture: &'static str,
    qualification: &'static str,
    routing: &'static str,
    layer_sharding: &'static str,
    daemon: Section<Daemon>,
    members: Section<Members>,
    service: Section<Option<Service>>,
}
#[derive(Serialize)]
struct Daemon {
    role: &'static str,
    draining: bool,
    registry: &'static str,
    connections: u32,
    handshakes: u32,
    pending_admissions: u32,
    pending_approvals: u32,
}
#[derive(Default, Serialize)]
struct Members {
    workers: u32,
    clients: u32,
    online: u32,
    offline: u32,
    draining: u32,
    revoked: u32,
    schedulable_workers: u32,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ServiceState {
    WaitingForRegistry,
    VerifyingCatalog,
    Loading,
    ReadyUnqualified,
    Ready,
    Serving,
    Stopping,
    Stopped,
    Recovered,
    StoppedPinsRetained,
    FailedPinsRetained,
    StoppedCleanupProven,
    FailedCleanupProven,
}
// Projection only: all other service fields, including raw errors and paths, are discarded.
#[derive(Deserialize, Serialize)]
struct Service {
    state: ServiceState,
    fixture: bool,
    inference_qualified: bool,
    process_group_absent: bool,
    cleanup_proven: bool,
    reserved_bytes: Option<Decimal>,
}

fn section<T>(
    reply: Option<AdminReply>,
    codes: &[&str],
    project: impl FnOnce(&AdminReply) -> Option<T>,
) -> Section<T> {
    let Some(reply) = reply.filter(|r| r.ok) else {
        return Section::Unavailable {
            action: "Check the selected local daemon, role and private endpoint permissions; do not retry mutations.",
        };
    };
    if reply.message.len() <= 16 * 1024
        && codes.contains(&reply.code.as_str())
        && let Some(data) = project(&reply)
    {
        Section::Available { data }
    } else {
        Section::InvalidReport {
            action: "Check that CLI and daemon use the same supported build; no raw report is included.",
        }
    }
}
fn daemon(reply: &AdminReply) -> Option<Daemon> {
    let mut fields = BTreeMap::new();
    for part in reply.message.split_whitespace() {
        if let Some((key, value)) = part.split_once('=')
            && fields.insert(key, value).is_some()
        {
            return None;
        }
    }
    Some(Daemon {
        role: match *fields.get("role")? {
            "coordinator" => "coordinator",
            "worker" => "worker",
            "client" => "client",
            _ => return None,
        },
        draining: reply.code == "DRAINING",
        registry: match *fields.get("registry")? {
            "active" => "active",
            "online" => "online",
            "offline" => "offline",
            "draining" => "draining",
            "revoked" => "revoked",
            "unavailable" => "unavailable",
            _ => return None,
        },
        connections: fields.get("connections")?.parse().ok()?,
        handshakes: fields.get("handshakes")?.parse().ok()?,
        pending_admissions: fields.get("pending_admissions")?.parse().ok()?,
        pending_approvals: fields.get("pending_approvals")?.parse().ok()?,
    })
}
fn members(reply: &AdminReply) -> Option<Members> {
    let nodes: Vec<NodeSummary> =
        serde_json::from_value(parse_json(reply.message.as_bytes()).ok()?).ok()?;
    if nodes.len() > 64 {
        return None;
    }
    let mut result = Members::default();
    let mut ids = BTreeSet::new();
    for node in nodes {
        if !ids.insert(node.node) {
            return None;
        }
        match node.role {
            Role::Worker => result.workers += 1,
            Role::Controller => result.clients += 1,
            Role::Coordinator => return None,
        }
        match node.status {
            NodeStatus::Online => result.online += 1,
            NodeStatus::Offline => result.offline += 1,
            NodeStatus::Draining => result.draining += 1,
            NodeStatus::Revoked => result.revoked += 1,
        }
        if node.schedulable {
            if node.role != Role::Worker || node.status != NodeStatus::Online {
                return None;
            }
            result.schedulable_workers += 1;
        }
    }
    Some(result)
}

/// Exactly three existing read-only local calls; no retries or per-node fan-out.
pub fn collect(state: &Path) -> Bundle {
    collect_with(|command| {
        starlings_security::admin_call(state, &serde_json::json!({"command":command})).ok()
    })
}
fn collect_with(mut query: impl FnMut(&str) -> Option<AdminReply>) -> Bundle {
    Bundle {
        version: Version,
        program_version: env!("CARGO_PKG_VERSION"),
        os: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        qualification: "not_assessed",
        routing: "whole_request",
        layer_sharding: "not_qualified",
        daemon: section(query("status"), &["ENROLLED_IDLE", "DRAINING"], daemon),
        members: section(query("nodes"), &["NODES"], members),
        service: section(query("service"), &["SERVICE"], |reply| {
            serde_json::from_value(parse_json(reply.message.as_bytes()).ok()?).ok()
        }),
    }
}
pub fn write(bundle: &Bundle, mut output: impl Write) -> Result<(), &'static str> {
    serde_json::to_writer_pretty(&mut output, bundle).map_err(|_| "DIAGNOSTIC_WRITE_FAILED")?;
    output
        .write_all(b"\n")
        .map_err(|_| "DIAGNOSTIC_WRITE_FAILED")
}

#[cfg(test)]
mod tests;
