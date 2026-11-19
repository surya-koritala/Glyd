//! Bounded v1 records and validation. No network authentication or inference.
mod control;
mod costs;
mod data;
mod json;
mod records;
mod scalars;
mod validation;

pub use control::*;
pub use costs::*;
pub use data::*;
pub use json::{MAX_JSON_BYTES, parse_json};
pub use records::*;
pub use scalars::*;
pub use validation::{Contract, FixtureMode, canonical_digest, decode, encode};

/// Rows one decode step may carry: its token and the drafts after it (GLY-360).
pub const MAX_DECODE_ROWS: u32 = 8;
pub const PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Error(pub &'static str);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}
