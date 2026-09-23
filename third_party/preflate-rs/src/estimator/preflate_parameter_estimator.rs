/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use bitcode::{Decode, Encode};

use crate::{
    bit_helper::bit_length,
    deflate::{deflate_constants, deflate_reader::DeflateContents},
    estimator::{add_policy_estimator::DictionaryAddPolicy, preflate_parse_config::MatchingType},
    hash_algorithm::HashAlgorithm,
    preflate_error::Result,
    preflate_input::PlainText,
};

use super::{
    add_policy_estimator::estimate_add_policy,
    complevel_estimator::estimate_preflate_comp_level,
    preflate_stream_info::{PreflateStreamInfo, extract_preflate_info},
};

#[derive(Encode, Decode, Debug, Copy, Clone, Eq, PartialEq)]
pub struct TokenPredictorParameters {
    /// Zlib does not match to first byte of a file in order to reserve 0 for the end of chain
    pub matches_to_start_detected: bool,

    /// if there are matches that have a distance larger than window_size - MAX_MATCH.
    /// Zlib does not allow these.
    pub very_far_matches_detected: bool,
    pub window_bits: u32,

    pub strategy: PreflateStrategy,
    pub nice_length: u32,

    /// if something, then we use the "fast" compressor, which only adds smaller substrings
    /// to the dictionary
    pub add_policy: DictionaryAddPolicy,

    pub max_token_count: u16,

    pub zlib_compatible: bool,
    pub max_dist_3_matches: u32,
    pub matching_type: MatchingType,
    pub max_chain: u32,
    pub min_len: u32,

    pub hash_algorithm: HashAlgorithm,

    pub block_type_strategy: BlockTypeStrategy,
}

#[derive(Encode, Decode, Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum BlockTypeStrategy {
    Dynamic,
    Mixed,
    Static,
    Uncompressed,
}

/// From the plain text and the preflate blocks, estimate the preflate parameters
pub fn estimate_preflate_parameters(
    deflate_contents: &DeflateContents,
    plain_text: &PlainText,
) -> Result<TokenPredictorParameters> {
    let info = extract_preflate_info(&deflate_contents.blocks);

    let preflate_strategy = estimate_preflate_strategy(&info);
    let huff_strategy = estimate_preflate_huff_strategy(&info);

    if preflate_strategy == PreflateStrategy::Store
        || preflate_strategy == PreflateStrategy::HuffOnly
    {
        // No dictionary used
        return Ok(TokenPredictorParameters {
            window_bits: 0,
            very_far_matches_detected: false,
            matches_to_start_detected: false,
            strategy: preflate_strategy,
            nice_length: 0,
            add_policy: DictionaryAddPolicy::AddAll,
            max_token_count: 16386,
            zlib_compatible: true,
            max_dist_3_matches: 0,
            matching_type: MatchingType::Greedy,
            max_chain: 0,
            min_len: 0,
            hash_algorithm: HashAlgorithm::None,
            block_type_strategy: huff_strategy,
        });
    }

    let window_bits = estimate_preflate_window_bits(info.max_dist);
    let mem_level = estimate_preflate_mem_level(info.max_tokens_per_block);
    let add_policy = estimate_add_policy(&deflate_contents.blocks);

    //let hash_shift = 5;
    //let hash_mask = 32767;

    let max_token_count = (1 << (6 + mem_level)) - 1;

    let cl = estimate_preflate_comp_level(
        window_bits,
        mem_level,
        info.min_len,
        deflate_contents,
        plain_text,
        add_policy,
    )?;

    let zlib_compatible = !info.matches_to_start_detected
        && !cl.very_far_matches_detected
        && (info.max_dist_3_matches < 4096 || add_policy != DictionaryAddPolicy::AddAll);

    Ok(TokenPredictorParameters {
        window_bits,
        very_far_matches_detected: cl.very_far_matches_detected,
        matches_to_start_detected: info.matches_to_start_detected,
        strategy: estimate_preflate_strategy(&info),
        nice_length: cl.nice_length,
        add_policy: add_policy,
        max_token_count,
        zlib_compatible,
        max_dist_3_matches: info.max_dist_3_matches,
        matching_type: cl.match_type,
        max_chain: cl.max_chain,
        min_len: info.min_len,
        hash_algorithm: cl.hash_algorithm,
        block_type_strategy: estimate_preflate_huff_strategy(&info),
    })
}

#[derive(Encode, Decode, Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum PreflateStrategy {
    Default,
    RleOnly,
    HuffOnly,
    Store,
}

fn estimate_preflate_mem_level(max_block_size_: u32) -> u32 {
    let mut max_block_size = max_block_size_;
    let mut mbits = 0;
    while max_block_size > 0 {
        mbits += 1;
        max_block_size >>= 1;
    }
    mbits = std::cmp::min(std::cmp::max(mbits, 7), 15);
    mbits - 6
}

fn estimate_preflate_window_bits(max_dist_: u32) -> u32 {
    let mut max_dist = max_dist_;
    max_dist += deflate_constants::MIN_LOOKAHEAD;
    let wbits = bit_length(max_dist - 1);
    std::cmp::min(std::cmp::max(wbits, 9), 15)
}

fn estimate_preflate_strategy(info: &PreflateStreamInfo) -> PreflateStrategy {
    if info.count_stored_blocks == info.count_blocks {
        return PreflateStrategy::Store;
    }
    if info.count_huff_blocks == info.count_blocks {
        return PreflateStrategy::HuffOnly;
    }
    if info.count_rle_blocks == info.count_blocks {
        return PreflateStrategy::RleOnly;
    }
    PreflateStrategy::Default
}

fn estimate_preflate_huff_strategy(info: &PreflateStreamInfo) -> BlockTypeStrategy {
    if info.count_static_huff_tree_blocks == info.count_blocks {
        return BlockTypeStrategy::Static;
    }
    if info.count_stored_blocks == info.count_blocks {
        return BlockTypeStrategy::Uncompressed;
    }
    if info.count_static_huff_tree_blocks == 0 {
        return BlockTypeStrategy::Dynamic;
    }
    BlockTypeStrategy::Mixed
}

#[test]
#[ignore = "reads sample files the crate does not ship"]
fn verify_zlib_recognition() {
    use crate::{
        deflate::deflate_reader::parse_deflate_whole,
        estimator::preflate_parse_config::{
            SLOW_PREFLATE_PARSER_SETTINGS, ZLIB_PREFLATE_PARSER_SETTINGS,
        },
        utils::read_file,
    };

    for i in 0..=9 {
        let v = read_file(&format!("compressed_zlib_level{}.deflate", i));
        let (contents, plain_text) = parse_deflate_whole(&v).unwrap();

        let params = estimate_preflate_parameters(&contents, &plain_text).unwrap();

        assert_eq!(params.zlib_compatible, true);
        if i == 0 {
            assert_eq!(params.strategy, PreflateStrategy::Store);
        } else if i >= 1 && i < 4 {
            let config = &ZLIB_PREFLATE_PARSER_SETTINGS[i as usize - 1];
            assert!(
                params.max_chain <= config.max_chain,
                "max_chain mismatch {} should be <= {}",
                params.max_chain,
                config.max_chain
            );
            assert_eq!(params.matching_type, config.match_type);
            assert_eq!(params.add_policy, config.dictionary_add_policy);
            assert_eq!(params.nice_length, config.nice_length);
            assert_eq!(params.strategy, PreflateStrategy::Default);
        } else if i >= 4 {
            let config = &SLOW_PREFLATE_PARSER_SETTINGS[i as usize - 4];
            assert!(
                params.max_chain <= config.max_chain,
                "max_chain mismatch {} should be <= {}",
                params.max_chain,
                config.max_chain
            );
            assert_eq!(params.matching_type, config.match_type);
            assert_eq!(params.add_policy, config.dictionary_add_policy);
            assert_eq!(params.nice_length, config.nice_length);
            assert_eq!(params.strategy, PreflateStrategy::Default);
        }
    }
}

#[test]
#[ignore = "reads sample files the crate does not ship"]
fn verify_miniz_recognition() {
    use crate::deflate::deflate_reader::parse_deflate_whole;
    use crate::utils::read_file;

    for i in 0..=9 {
        let v = read_file(&format!("compressed_flate2_level{}.deflate", i));
        let (contents, plain_text) = parse_deflate_whole(&v).unwrap();

        let params = estimate_preflate_parameters(&contents, &plain_text).unwrap();

        if i == 0 {
            assert_eq!(params.strategy, PreflateStrategy::Store);
        } else if i == 1 {
            println!("{:?}", params);
        } else {
            println!("{:?}", params);
        }
    }
}

#[test]
#[ignore = "reads sample files the crate does not ship"]
fn verify_zlibng_recognition() {
    use crate::deflate::deflate_reader::parse_deflate_whole;
    use crate::utils::read_file;

    for i in 1..=2 {
        let v = read_file(&format!("compressed_zlibng_level{}.deflate", i));
        let (contents, plain_text) = parse_deflate_whole(&v).unwrap();

        let params = estimate_preflate_parameters(&contents, &plain_text).unwrap();

        if i == 0 {
            assert_eq!(params.strategy, PreflateStrategy::Store);
        } else if i == 1 {
            println!("{:?}", params);
        } else {
            println!("{:?}", params);
        }
    }
}
