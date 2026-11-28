//! glyd-engine: Glyd's own inference engine. The families of `support.rs` as descriptions of layer kinds run by one executor (`dynamic/`) through one scheduler
//! (`sched.rs`), on NVIDIA GPUs from Glyd-packed weights or the release's own, with a chat and the OpenAI, Anthropic and Ollama APIs. See REFERENCE.md.

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
pub mod bench;
pub mod calib;
pub mod cb2_prefill;
pub mod cb_pack;
#[cfg(feature = "dev")]
pub mod cb_random;
#[cfg(feature = "dev")]
pub mod ceiling;
pub mod chat;
pub mod choice;
pub mod choose;
pub mod cli;
/// The commands and their flags as one table, the file of the root crate (src/cmdtable.rs): the help, the man page and the flags a command refuses are made from it.
#[path = "../../src/cmdtable.rs"]
pub mod cmdtable;
pub mod conn;
pub mod connect;
pub mod constrain;
pub mod conv;
pub mod costs;
#[cfg(feature = "dev")]
pub mod cpu_bench;
pub mod cpu_cb;
pub mod cpu_experts;
pub mod cpu_fp4;
pub mod cpu_hb;
pub mod cpu_hbg;
mod cpu_simd;
pub mod deepseek_v4;
pub mod dense_ram;
pub mod dev;
pub mod devplan;
/// A type for each of the engine's kernels, with its arguments' types (R15; made by build.rs from kernels/*.cu: build/kernel_types.rs).
#[allow(dead_code)]
pub(crate) mod kt {
    include!(concat!(env!("OUT_DIR"), "/kernel_types.rs"));
}
pub mod device;
pub mod disk;
pub mod doctor;
#[cfg(feature = "dev")]
pub mod dsv4_ref;
pub mod dynamic;
pub mod engine_api;
pub mod entry;
pub mod f8;
#[cfg(test)]
mod families;
pub mod ffi;
pub mod fold;
pub mod formats;
pub mod gemma4;
pub mod gguf;
pub mod gguf_pre;
pub mod gguf_tok;
pub mod glm;
pub mod gpu_ops;
pub mod hb_codec;
pub mod hb_pack;
pub mod hb_usage;
pub mod hbg_store;
#[cfg(feature = "dev")]
pub mod hbpack;
pub mod hub;
#[cfg(feature = "dev")]
pub mod intake;
#[cfg(feature = "dev")]
pub mod kcheck;
#[cfg(feature = "dev")]
pub mod kcheck35;
pub mod kcheck_hb;
#[cfg(feature = "dev")]
pub mod kcheck_moe;
#[cfg(feature = "dev")]
pub mod kcheck_nh;
pub mod kern;
pub mod kern35;
pub mod kern_glm;
pub mod kern_nh;
pub mod kimi;
#[cfg(feature = "dev")]
pub mod kimi_ref;
pub mod kv;
pub mod kv_events;
pub mod kvcal;
#[cfg(feature = "dev")]
pub mod kvshapes;
pub mod levels;
#[cfg(feature = "dev")]
pub mod logits;
pub mod machine;
pub mod mapped;
pub mod models;
pub mod moe;
pub mod moe_cb;
pub mod moe_hb;
pub mod moe_hbg;
#[cfg(feature = "dev")]
pub mod moe_ref;
pub mod moe_rows;
pub mod monitor;
pub mod muse;
#[cfg(feature = "dev")]
pub mod muse_ref;
pub mod nemotron;
pub mod offload;
pub mod ops;
pub mod os;
pub mod otel;
pub mod pack;
#[cfg(feature = "dev")]
pub mod packcheck;
pub mod pixels;
pub mod plan;
pub mod plancheck;
/// The quality bar of the lossy levels: the suite, the cards, publish or refuse (R14, docs/quality.md).
pub mod quality;
pub mod qwen35;
#[cfg(feature = "dev")]
pub mod qwen35_ref;
pub mod qwen4x;
pub mod qwen_tools;
#[cfg(feature = "dev")]
pub mod qx_trellis;
pub mod reader;
#[cfg(feature = "dev")]
pub mod report;
pub mod runtime;
pub mod sampler;
pub mod sched;
pub mod server;
pub mod service;
#[cfg(feature = "dev")]
pub mod ship;
pub mod sizes;
pub mod skinny;
#[cfg(feature = "dev")]
pub mod stacked;
pub mod status;
pub mod store;
pub mod support;
pub mod tiering;
pub mod tok;
pub mod tools;
pub mod trace;
pub mod ui;
pub mod vision;
#[cfg(feature = "dev")]
pub mod vision_check;
pub mod weights;
pub mod wire;
pub mod xtml;
