/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

pub fn bit_length(n: u32) -> u32 {
    32 - n.leading_zeros()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct DebugHash {
    hash: u64,
}

#[allow(dead_code)]
impl DebugHash {
    pub fn update<T: Into<i64>>(&mut self, v: T) {
        self.hash = self.hash.wrapping_mul(13).wrapping_add(v.into() as u64);
    }

    pub fn update_slice<T: Into<i64> + Copy>(&mut self, v: &[T]) {
        v.iter().for_each(|x| self.update(*x));
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }
}
