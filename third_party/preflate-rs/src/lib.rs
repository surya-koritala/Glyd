/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

// forbid lints that we already have eliminated from the codebase so they don't show up in the future
#![forbid(unsafe_code)]
#![forbid(trivial_casts)]
#![forbid(trivial_numeric_casts)]
#![forbid(non_ascii_idents)]
#![forbid(redundant_lifetimes)]
#![forbid(single_use_lifetimes)]
#![forbid(macro_use_extern_crate)]
#![forbid(missing_unsafe_on_extern)]
#![forbid(unused_import_braces)]
#![warn(unused_crate_dependencies)]
#![warn(unused_extern_crates)]
#![warn(unused_lifetimes)]
#![warn(unused_macro_rules)]

mod bit_helper;
mod cabac_codec;
pub mod chunked;
mod deflate;
mod estimator;
mod hash_algorithm;
mod hash_chain;
mod hash_chain_holder;

mod preflate_error;
mod preflate_input;

mod statistical_codec;
mod stream_processor;
mod token_predictor;
mod tree_predictor;
mod utils;

pub use estimator::preflate_parameter_estimator::TokenPredictorParameters;
pub use hash_algorithm::HashAlgorithm;
pub use preflate_error::ExitCode;
pub use preflate_error::{AddContext, PreflateError, Result, err_exit_code};
pub use preflate_input::{PlainText, PreflateInput};

pub use statistical_codec::{CodecCorrection, CountNonDefaultActions};
pub use stream_processor::{
    PreflateStreamChunkResult, PreflateStreamProcessor, RecreateStreamProcessor,
    preflate_whole_deflate_stream, recreate_whole_deflate_stream,
};

/// Configure the preflate stream processor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreflateConfig {
    /// Maximum number of lookups we will do in the hash chain. This will limit the
    /// amount of CPU time we spend on each chunk.
    pub max_chain_length: u32,

    /// The maximum size of the plain text that will be decompressed to memory before
    /// being recompressed. This limits memory usage and processing time, especially
    /// for files that are extremely compressed (eg zip bombs)
    pub plain_text_limit: usize,

    /// Decompress the data and verify that it can be recompressed to exactly the same bytes.
    /// This will slow down processing is isn't necessary if you are already doing end-to-end
    /// verification of the data.
    pub verify_compression: bool,
}

impl Default for PreflateConfig {
    fn default() -> Self {
        Self {
            max_chain_length: 4096,
            plain_text_limit: 128 * 1024 * 1024,
            verify_compression: true,
        }
    }
}

#[cfg(test)]
static INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the logger for tests. This is a no-op if the logger is already initialized.
#[cfg(test)]
pub fn init_logging() {
    INIT.call_once(|| {
        let _ = env_logger::builder().is_test(true).try_init();
    });
}
