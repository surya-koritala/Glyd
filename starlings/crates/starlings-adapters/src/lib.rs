//! Host probes and external adapters. Native inference requires a separately qualified backend.
#[cfg(feature = "test-backend")]
pub mod fake;
pub mod resources;
pub mod service;

pub mod partial;
