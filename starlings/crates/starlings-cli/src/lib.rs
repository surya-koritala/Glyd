//! Application integration shared by the daemon and bounded acceptance harnesses.
pub mod benchmark;
pub mod calibration;
pub mod collector;
pub mod contribution;
pub mod diagnostics;
pub mod glyd_costs;
pub mod inference;
pub mod link_record;
pub mod pipeline;
pub mod planning;
pub mod routes;
pub mod scoreboard;
pub mod split;
pub mod ui;

/// Strict local administrative response shared by the CLI and acceptance harness.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdminReply {
    pub ok: bool,
    pub code: String,
    pub message: String,
}
impl AdminReply {
    pub fn ok(code: &str, message: String) -> Self {
        Self {
            ok: true,
            code: code.into(),
            message,
        }
    }
    pub fn error(error: String) -> Self {
        Self {
            ok: false,
            code: "REFUSED".into(),
            message: error,
        }
    }
}
