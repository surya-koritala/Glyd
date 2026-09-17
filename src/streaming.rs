use std::io::{self, Read, Write};
use crate::format::{
    BlockHeader, HEADER_SIZE, MAGIC, CURRENT_VERSION, MAX_BLOCK_SIZE, PADDING,
    FLAG_RAW_UNCOMPRESSED,
};
use crate::{compress_block_into, decompress_into};

/// A streaming compressor that wraps any `std::io::Write` sink.
///
/// Accumulates uncompressed data into 64 KB blocks and writes them out
/// as fast SIMD-stream blocks.
pub struct AlatirokWriter<W: Write> {
    writer: Option<W>,
    buffer: Vec<u8>,
    scratch_compressed: Vec<u8>,
}

impl<W: Write> AlatirokWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer: Some(writer),
            buffer: Vec::with_capacity(MAX_BLOCK_SIZE),
            scratch_compressed: Vec::with_capacity(MAX_BLOCK_SIZE + 512),
        }
    }

    /// Flush any remaining buffered uncompressed data into a final compressed block.
    pub fn finish(mut self) -> io::Result<W> {
        self.flush_current_block()?;
        if let Some(mut w) = self.writer.take() {
            w.flush()?;
            Ok(w)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "Writer already closed"))
        }
    }

    fn flush_current_block(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        if let Some(ref mut w) = self.writer {
            self.scratch_compressed.clear();
            compress_block_into(&self.buffer, &mut self.scratch_compressed);
            w.write_all(&self.scratch_compressed)?;
            self.buffer.clear();
        }
        Ok(())
    }
}

impl<W: Write> Write for AlatirokWriter<W> {
    fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
        let total_in = buf.len();

        while !buf.is_empty() {
            let available = MAX_BLOCK_SIZE - self.buffer.len();
            if available == 0 {
                self.flush_current_block()?;
                continue;
            }

            let to_copy = buf.len().min(available);
            self.buffer.extend_from_slice(&buf[..to_copy]);
            buf = &buf[to_copy..];

            if self.buffer.len() == MAX_BLOCK_SIZE {
                self.flush_current_block()?;
            }
        }

        Ok(total_in)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_current_block()?;
        if let Some(ref mut w) = self.writer {
            w.flush()
        } else {
            Ok(())
        }
    }
}

impl<W: Write> Drop for AlatirokWriter<W> {
    fn drop(&mut self) {
        let _ = self.flush_current_block();
        if let Some(ref mut w) = self.writer {
            let _ = w.flush();
        }
    }
}

/// A streaming decompressor that wraps any `std::io::Read` source.
///
/// Decodes blocks on the fly into an internal 64 KB buffer, enabling
/// streaming reads without loading the entire archive into memory.
pub struct AlatirokReader<R: Read> {
    reader: R,
    block_buffer: Vec<u8>,
    decomp_buffer: Vec<u8>,
    decomp_cursor: usize,
    decomp_len: usize,
    eof_reached: bool,
}

impl<R: Read> AlatirokReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            block_buffer: Vec::with_capacity(MAX_BLOCK_SIZE + 512),
            decomp_buffer: vec![0u8; MAX_BLOCK_SIZE + PADDING * 2],
            decomp_cursor: 0,
            decomp_len: 0,
            eof_reached: false,
        }
    }

    fn read_next_block(&mut self) -> io::Result<bool> {
        let mut header_bytes = [0u8; HEADER_SIZE];
        match self.reader.read_exact(&mut header_bytes) {
            Ok(()) => {}
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                self.eof_reached = true;
                return Ok(false);
            }
            Err(e) => return Err(e),
        }

        let header: BlockHeader = unsafe { std::ptr::read_unaligned(header_bytes.as_ptr() as *const BlockHeader) };
        if header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid Alatirok magic"));
        }
        if header.version > CURRENT_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported Alatirok bitstream version"));
        }

        let uncomp_len = header.uncompressed_len as usize;
        let payload_len = if (header.flags & FLAG_RAW_UNCOMPRESSED) != 0 {
            uncomp_len
        } else {
            crate::format::payload_len(
                header.token_count as usize,
                header.offset_count as usize,
                header.extras_count as usize,
                header.literal_len as usize,
            )
        };

        self.block_buffer.clear();
        self.block_buffer.extend_from_slice(&header_bytes);
        let current_len = self.block_buffer.len();
        self.block_buffer.resize(current_len + payload_len, 0);

        self.reader.read_exact(&mut self.block_buffer[current_len..])?;

        let written = decompress_into(&self.block_buffer, &mut self.decomp_buffer)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Decompression error: {:?}", e)))?;

        self.decomp_cursor = 0;
        self.decomp_len = written;
        Ok(true)
    }
}

impl<R: Read> Read for AlatirokReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        while self.decomp_cursor >= self.decomp_len {
            if self.eof_reached {
                return Ok(0);
            }
            let has_more = self.read_next_block()?;
            if !has_more {
                return Ok(0);
            }
        }

        let available = self.decomp_len - self.decomp_cursor;
        let to_copy = buf.len().min(available);
        buf[..to_copy].copy_from_slice(&self.decomp_buffer[self.decomp_cursor..self.decomp_cursor + to_copy]);
        self.decomp_cursor += to_copy;
        Ok(to_copy)
    }
}
