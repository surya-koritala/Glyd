//! glyd-engine: Glyd's own inference engine. Qwen3 dense on CUDA from Glyd-packed weights with every matrix
//! compressed, the lossless KV cache, a chat and an OpenAI-compatible API. See README.md.

/// A tuning variable of a development build (the `dev` feature, on by default): its value from the process's environment. A release build
/// (`--no-default-features`) reads none: this is `None` there, and its binary does not name the variable.
#[macro_export]
macro_rules! tune {
    ($name:literal) => {
        if cfg!(feature = "dev") {
            ::std::env::var($name).ok()
        } else {
            None
        }
    };
}

/// A marker on the engine's stream for the timeline of a traced step (src/trace.rs), in a development build; nothing in a release build.
#[macro_export]
macro_rules! tmark {
    ($s:expr, $label:literal) => {
        #[cfg(feature = "dev")]
        $crate::trace::mark($s, $label);
    };
}

pub mod args;
pub mod trace;
pub mod calib;
#[cfg(feature = "dev")]
pub mod ceiling;
pub mod chat;
pub mod choice;
pub mod cli;
pub mod conv;
pub mod costs;
#[cfg(feature = "dev")]
pub mod cpu_bench;
pub mod cpu_cb;
pub mod cpu_experts;
pub mod cpu_fp4;
pub mod cpu_hb;
mod cpu_simd;
pub mod deepseek_v4;
#[cfg(feature = "dev")]
pub mod dsv4_ref;
pub mod dev;
pub mod dynamic;
pub mod f8;
pub mod disk;
pub mod ffi;
pub mod os;
pub mod fit;
pub mod levels;
pub mod glm;
pub mod kimi;
#[cfg(feature = "dev")]
pub mod kimi_ref;
pub mod glm_cb;
pub mod hub;
#[cfg(feature = "dev")]
pub mod hbpack;
#[cfg(feature = "dev")]
pub mod kcheck;
#[cfg(feature = "dev")]
pub mod kcheck35;
#[cfg(feature = "dev")]
pub mod kcheck_glm;
pub mod kcheck_hb;
#[cfg(feature = "dev")]
pub mod kcheck_moe;
#[cfg(feature = "dev")]
pub mod kcheck_nh;
#[cfg(feature = "dev")]
pub mod kcheck_qwen4x;
pub mod kvcal;
pub mod kern;
pub mod kv;
#[cfg(feature = "dev")]
pub mod kvtest;
pub mod machine;
pub mod mapped;
pub mod model;
pub mod monitor;
pub mod muse;
#[cfg(feature = "dev")]
pub mod muse_ref;
pub mod model35;
pub mod model_gemma4;
pub mod mla_glm;
pub mod mtp_glm;
pub mod model_glm;
pub mod model_qwen4x;
pub mod moe_qwen4x;
#[cfg(feature = "dev")]
pub mod qx_trellis;
pub mod qwen4x;
pub mod moe;
pub mod moe_glm;
pub mod moe_glm_cb;
pub mod cb2_prefill;
pub mod skinny;
#[cfg(feature = "dev")]
pub mod glm_serve;
pub mod qwen4x_serve;
pub mod vision;
pub mod pixels;
pub mod qwen_tools;
#[cfg(feature = "dev")]
pub mod ship;
pub mod draft_loop;
pub mod moe_hb;
pub mod moe_nh;
pub mod moe_rows;
#[cfg(feature = "dev")]
pub mod moe_ref;
pub mod model_nh;
pub mod nemotron;
pub mod offload;
pub mod pack;
pub mod qwen35;
#[cfg(feature = "dev")]
pub mod qwen35_ref;
#[cfg(feature = "dev")]
pub mod report;
pub mod sampler;
pub mod sched;
pub mod server;
pub mod status;
pub mod store;
pub mod tools;
#[cfg(feature = "dev")]
pub mod stress;
pub mod tok;
pub mod xtml;
pub mod ui;
pub mod weights;
pub mod wire;
