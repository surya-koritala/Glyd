/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use cabac::{CabacReader, CabacWriter};

use crate::{
    bit_helper::bit_length,
    statistical_codec::{
        CodecCorrection, CountNonDefaultActions, PredictionDecoder, PredictionEncoder,
    },
};

/// calculates the difference between two values, keeping track
/// of the sign as the lowest bit so that the difference is never negative
pub fn encode_difference(pred_val: u32, act_val: u32) -> u32 {
    if pred_val >= act_val {
        (pred_val - act_val) << 1
    } else {
        ((act_val - pred_val) << 1) | 1
    }
}

/// decodes the result of the previous calculation
pub fn decode_difference(pred_val: u32, encoded_val: u32) -> u32 {
    if encoded_val & 1 == 0 {
        pred_val - (encoded_val >> 1)
    } else {
        pred_val + (encoded_val >> 1)
    }
}

#[test]
fn test_encode_decode_difference() {
    for i in 0..10 {
        assert_eq!(i, decode_difference(0, encode_difference(0, i)));
        assert_eq!(i, decode_difference(10, encode_difference(10, i)));
        assert_eq!(i, decode_difference(100, encode_difference(100, i)));
    }
}

#[derive(Default)]
struct PredictionCabacContext<CTX> {
    default_signal_ctx: [CTX; CodecCorrection::MAX as usize],

    default_ctx: [[CTX; 32]; CodecCorrection::MAX as usize],
    default_ctx_bits: [[CTX; 32]; CodecCorrection::MAX as usize],

    correction_ctx: [[CTX; 32]; CodecCorrection::MAX as usize],
    correction_ctx_bits: [[CTX; 32]; CodecCorrection::MAX as usize],
    //debug_ops: VecDeque<DebugOps>,
}

impl<CTX> PredictionCabacContext<CTX> {
    fn write_exp_encoded<const N: usize, W: CabacWriter<CTX>>(
        value: u32,
        context: &mut [CTX; N],
        context_bits: &mut [CTX; N],
        writer: &mut W,
    ) {
        let bl = bit_length(value) as usize;

        writer.put_unary_encoded(bl, context).unwrap();

        if bl > 1 {
            writer
                .put_n_bits((value & ((1 << bl) - 1)).into(), bl - 1, context_bits)
                .unwrap();
        }
    }

    fn read_exp_value<R: CabacReader<CTX>, const N: usize>(
        context: &mut [CTX; N],
        context_bits: &mut [CTX; N],
        reader: &mut R,
    ) -> u32 {
        let bits_found = reader.get_unary_encoded(context).unwrap();

        match bits_found {
            0 => 0,
            1 => 1,
            _ => {
                reader.get_n_bits(bits_found - 1, context_bits).unwrap() as u32
                    | (1 << (bits_found - 1))
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Actions {
    Default(CodecCorrection, u32),
    Correction(CodecCorrection, u32),
}

pub struct PredictionEncoderCabac<W, CTX> {
    context: PredictionCabacContext<CTX>,

    actions: Vec<Actions>,

    last_action: [Option<usize>; CodecCorrection::MAX as usize],

    action_seen: [bool; CodecCorrection::MAX as usize],

    count: CountNonDefaultActions,
    writer: W,
}

impl<W: CabacWriter<CTX>, CTX: Default> PredictionEncoderCabac<W, CTX> {
    pub fn new(writer: W) -> Self {
        Self {
            context: PredictionCabacContext::<CTX>::default(),
            writer,
            actions: Default::default(),
            last_action: Default::default(),
            count: CountNonDefaultActions::default(),
            action_seen: Default::default(),
        }
    }

    /// for debugging
    #[allow(dead_code)]
    pub fn print(&self) {
        self.count.print();
    }
}

impl<W: CabacWriter<CTX>, CTX: Default> PredictionEncoder for PredictionEncoderCabac<W, CTX> {
    fn encode_verify_state(&mut self, _message: &'static str, _checksum: u64) {}

    fn encode_correction(&mut self, c: CodecCorrection, value: u32) {
        let i = c as usize;

        // if it is a non-zero correction, then we just insert it
        if value != 0 {
            self.last_action[i] = Some(self.actions.len());
            self.actions.push(Actions::Correction(c, value));
            self.action_seen[i] = true;

            self.count.record_correction(c, value);
            return;
        }

        // otherwise we look to see if the last action was a default action
        // and if so, just increment it
        if let Some(l) = self.last_action[i] {
            let a = &mut self.actions[l];
            if let Actions::Default(_, x) = a {
                *x += 1;
                return;
            }
        }

        // otherwise we add a new default action
        self.last_action[c as usize] = Some(self.actions.len());
        self.actions.push(Actions::Default(c, 1));
    }

    fn finish(&mut self) {
        // Now that we've collected everything, we can write out the actions
        // in the correct order so that the callers can just read them in order
        //
        // This is necessary since the reads are done the first time we need
        // something, so we can't write them out in the same order that we get them.

        // first write the bit set of things we've seen so we can completely exclude
        let mut actions_seen_ctx = CTX::default();
        for &s in self.action_seen.iter() {
            self.writer.put(s, &mut actions_seen_ctx).unwrap();
        }

        // now write all the actions
        for &a in self.actions.iter() {
            match a {
                Actions::Default(c, value) => {
                    let i = c as usize;

                    if self.action_seen[i] {
                        self.writer
                            .put(true, &mut self.context.default_signal_ctx[i])
                            .unwrap();

                        PredictionCabacContext::write_exp_encoded(
                            value - 1,
                            &mut self.context.default_ctx[i],
                            &mut self.context.default_ctx_bits[i],
                            &mut self.writer,
                        );
                    }
                }
                Actions::Correction(c, value) => {
                    let i = c as usize;
                    self.writer
                        .put(false, &mut self.context.default_signal_ctx[i])
                        .unwrap();
                    PredictionCabacContext::write_exp_encoded(
                        value - 1,
                        &mut self.context.correction_ctx[i],
                        &mut self.context.correction_ctx_bits[i],
                        &mut self.writer,
                    );
                }
            }
        }

        self.writer.finish().unwrap();
    }
}

pub struct PredictionDecoderCabac<R, CTX> {
    context: PredictionCabacContext<CTX>,
    reader: R,

    actions_seen: [bool; CodecCorrection::MAX as usize],
    default_actions: [u32; CodecCorrection::MAX as usize],
}

impl<R: CabacReader<CTX>, CTX: Default> PredictionDecoderCabac<R, CTX> {
    pub fn new(mut reader: R) -> Self {
        let mut actions_seen = [false; CodecCorrection::MAX as usize];
        let mut actions_seen_ctx = CTX::default();
        for i in 0..actions_seen.len() {
            actions_seen[i] = reader.get(&mut actions_seen_ctx).unwrap();
        }

        Self {
            context: PredictionCabacContext::<CTX>::default(),
            default_actions: Default::default(),
            actions_seen,
            reader,
        }
    }

    #[cold]
    fn decode_correction_slow(&mut self, c: usize) -> Result<u32, u32> {
        if self
            .reader
            .get(&mut self.context.default_signal_ctx[c])
            .unwrap()
        {
            let value = PredictionCabacContext::read_exp_value(
                &mut self.context.default_ctx[c],
                &mut self.context.default_ctx_bits[c],
                &mut self.reader,
            ) + 1;

            self.default_actions[c] = value - 1;
            return Err(0);
        } else {
            let value = PredictionCabacContext::read_exp_value(
                &mut self.context.correction_ctx[c],
                &mut self.context.correction_ctx_bits[c],
                &mut self.reader,
            ) + 1;

            return Err(value);
        }
    }
}

impl<R: CabacReader<CTX>, CTX: Default> PredictionDecoder for PredictionDecoderCabac<R, CTX> {
    fn decode_verify_state(&mut self, _message: &'static str, _checksum: u64) {}

    #[inline]
    fn decode_correction(&mut self, correction: CodecCorrection) -> u32 {
        // if the action hasn't been seen at all, then always return 0
        if !self.actions_seen[correction as usize] {
            return 0;
        }

        // otherwise if we still have default actions left, use those
        let c = correction as usize;
        if self.default_actions[c] > 0 {
            self.default_actions[c] -= 1;
            return 0;
        }

        match self.decode_correction_slow(c) {
            Ok(value) => value,
            Err(value) => return value,
        }
    }
}

#[test]
fn roundtree_cabac_decoding() {
    use crate::statistical_codec::{CodecAction, drive_encoder, verify_decoder};
    use cabac::vp8::{VP8Reader, VP8Writer};
    use std::io::Cursor;

    let mut buffer = Vec::new();

    let test_codec_actions = [
        CodecAction::Correction(CodecCorrection::DistanceCountCorrection, 1),
        CodecAction::Correction(CodecCorrection::TokenCount, 100000),
        CodecAction::Correction(CodecCorrection::BlockTypeCorrection, 5),
        CodecAction::Correction(CodecCorrection::DistAfterLenCorrection, 0),
    ];

    let mut encoder = PredictionEncoderCabac::new(VP8Writer::new(&mut buffer).unwrap());

    drive_encoder(&mut encoder, &test_codec_actions);

    encoder.finish();

    let mut decoder = PredictionDecoderCabac::new(VP8Reader::new(Cursor::new(&buffer)).unwrap());

    verify_decoder(&mut decoder, &test_codec_actions);
}

#[cfg(test)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Operation {
    Correction(u32, CodecCorrection),
}

#[test]
fn roundtree_cabac_correction() {
    // use the debug version of the cabac writer/reader to make sure that the we don't mix up contexts anywhere
    use cabac::debug::{DebugReader, DebugWriter};
    use std::io::Cursor;

    // generate a random set of operations
    let operations = [
        Operation::Correction(5, CodecCorrection::DistanceCountCorrection),
        Operation::Correction(1, CodecCorrection::BlockTypeCorrection),
        Operation::Correction(2, CodecCorrection::BlockTypeCorrection),
        Operation::Correction(3, CodecCorrection::BlockTypeCorrection),
        Operation::Correction(4, CodecCorrection::BlockTypeCorrection),
        Operation::Correction(0, CodecCorrection::DistAfterLenCorrection),
        Operation::Correction(0, CodecCorrection::DistOnlyCorrection),
        Operation::Correction(7, CodecCorrection::LDTypeCorrection),
        Operation::Correction(9, CodecCorrection::LenCorrection),
        Operation::Correction(0, CodecCorrection::LiteralPredictionWrong),
        Operation::Correction(100000, CodecCorrection::TokenCount),
        Operation::Correction(1, CodecCorrection::ReferencePredictionWrong),
    ];

    let mut buffer = Vec::new();

    let writer = DebugWriter::new(&mut buffer).unwrap();

    let mut encoder = PredictionEncoderCabac::new(writer);

    for &o in operations.iter() {
        match o {
            Operation::Correction(val, context_type) => {
                encoder.encode_correction(context_type, val);
            }
        }
    }

    encoder.finish();

    let reader = DebugReader::new(Cursor::new(&buffer)).unwrap();

    let mut decoder = PredictionDecoderCabac::new(reader);

    for (i, &o) in operations.iter().enumerate() {
        match o {
            Operation::Correction(val, context_type) => {
                assert_eq!(
                    val,
                    decoder.decode_correction(context_type),
                    "operation {}",
                    i
                );
            }
        }
    }
}

#[test]
fn roundtrip_cabac_write_value() {
    use cabac::vp8::{VP8Context, VP8Reader, VP8Writer};
    use std::io::Cursor;

    let mut buffer = Vec::new();

    let mut writer = VP8Writer::new(&mut buffer).unwrap();

    let mut context = [
        VP8Context::default(),
        VP8Context::default(),
        VP8Context::default(),
        VP8Context::default(),
    ];

    let mut context_bits = [
        VP8Context::default(),
        VP8Context::default(),
        VP8Context::default(),
        VP8Context::default(),
    ];

    for i in 0..10 {
        PredictionCabacContext::write_exp_encoded(
            i * 13,
            &mut context,
            &mut context_bits,
            &mut writer,
        );
    }

    writer.finish().unwrap();

    let mut reader = VP8Reader::new(Cursor::new(&buffer)).unwrap();
    context = [
        VP8Context::default(),
        VP8Context::default(),
        VP8Context::default(),
        VP8Context::default(),
    ];

    context_bits = [
        VP8Context::default(),
        VP8Context::default(),
        VP8Context::default(),
        VP8Context::default(),
    ];

    for i in 0..10 {
        assert_eq!(
            i * 13,
            PredictionCabacContext::read_exp_value(&mut context, &mut context_bits, &mut reader)
        );
    }
}
