use std::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CodecError {
    InvalidMagic,
    UnsupportedVersion(u16),
    CorruptedBitstream(&'static str),
    OffsetOutOfBounds { offset: usize, available: usize },
    OutputBufferTooSmall { required: usize, provided: usize },
    ChecksumMismatch { expected: u32, computed: u32 },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::InvalidMagic => write!(f, "Invalid magic block header"),
            CodecError::UnsupportedVersion(v) => write!(f, "Unsupported bitstream version: {}", v),
            CodecError::CorruptedBitstream(msg) => write!(f, "Corrupted bitstream: {}", msg),
            CodecError::OffsetOutOfBounds { offset, available } => {
                write!(f, "Match offset out of bounds: offset={}, available={}", offset, available)
            }
            CodecError::OutputBufferTooSmall { required, provided } => {
                write!(f, "Output buffer too small: required={}, provided={}", required, provided)
            }
            CodecError::ChecksumMismatch { expected, computed } => {
                write!(f, "CRC32 checksum mismatch: expected=0x{:08X}, computed=0x{:08X}", expected, computed)
            }
        }
    }
}

impl std::error::Error for CodecError {}

pub type Result<T> = std::result::Result<T, CodecError>;
