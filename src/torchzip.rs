//! PyTorch checkpoints (`torch.save`): a zip, stored (no compression),
//! of `<archive>/data.pkl` (a pickle of the object, each tensor's storage
//! a persistent id `('storage', torch.<Type>Storage, key, location,
//! numel)`) and `<archive>/data/<key>`, each storage's raw elements. The
//! storages' element widths come from the pickle, so the container can
//! hold them as byte planes, as it does a safetensors file's tensors. No
//! dependency: the pickle reader knows the opcodes `torch.save` writes and
//! gives up on any other.

use std::collections::HashMap;

#[derive(Clone, Debug)]
enum Val {
    Str(Vec<u8>),
    Global(Vec<u8>),
    Tuple(Vec<Val>),
    Other,
}

/// Each storage's element width by its key, from a checkpoint's pickle;
/// `None` for a pickle this reader does not follow.
pub fn storage_widths(pkl: &[u8]) -> Option<HashMap<Vec<u8>, usize>> {
    let mut frames: Vec<Vec<Val>> = vec![Vec::new()];
    let mut memo: HashMap<u32, Val> = HashMap::new();
    let mut out = HashMap::new();
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> Option<&[u8]> {
        let s = pkl.get(*i..i.checked_add(n)?)?;
        *i += n;
        Some(s)
    };
    let le = |s: &[u8]| s.iter().rev().fold(0u64, |a, &b| a << 8 | b as u64);
    loop {
        let op = *pkl.get(i)?;
        i += 1;
        let top = frames.last_mut()?;
        match op {
            0x80 => {
                take(&mut i, 1)?;
            }
            0x95 => {
                take(&mut i, 8)?;
            }
            b'(' => frames.push(Vec::new()),
            b'}' | b']' | b')' | b'N' | 0x88 | 0x89 => top.push(Val::Other),
            b'X' => {
                let n = le(take(&mut i, 4)?) as usize;
                top.push(Val::Str(take(&mut i, n)?.to_vec()))
            }
            0x8c | b'U' => {
                let n = take(&mut i, 1)?[0] as usize;
                top.push(Val::Str(take(&mut i, n)?.to_vec()))
            }
            b'T' | b'B' => {
                let n = le(take(&mut i, 4)?) as usize;
                let s = take(&mut i, n)?.to_vec();
                top.push(if op == b'T' { Val::Str(s) } else { Val::Other })
            }
            b'C' => {
                let n = take(&mut i, 1)?[0] as usize;
                take(&mut i, n)?;
                top.push(Val::Other)
            }
            b'c' => {
                let rest = pkl.get(i..)?;
                let a = rest.iter().position(|&b| b == b'\n')?;
                let b = rest[a + 1..].iter().position(|&b| b == b'\n')?;
                let mut g = rest[..a].to_vec();
                g.push(b'.');
                g.extend_from_slice(&rest[a + 1..a + 1 + b]);
                i += a + b + 2;
                top.push(Val::Global(g))
            }
            0x93 => {
                let name = top.pop()?;
                let module = top.pop()?;
                match (module, name) {
                    (Val::Str(mut m), Val::Str(n)) => {
                        m.push(b'.');
                        m.extend_from_slice(&n);
                        top.push(Val::Global(m))
                    }
                    _ => top.push(Val::Other),
                }
            }
            b'q' | b'r' | 0x94 => {
                let k = match op {
                    b'q' => take(&mut i, 1)?[0] as u32,
                    b'r' => le(take(&mut i, 4)?) as u32,
                    _ => memo.len() as u32,
                };
                memo.insert(k, top.last().cloned().unwrap_or(Val::Other));
            }
            b'h' | b'j' => {
                let k = if op == b'h' { take(&mut i, 1)?[0] as u32 } else { le(take(&mut i, 4)?) as u32 };
                top.push(memo.get(&k).cloned().unwrap_or(Val::Other))
            }
            b'K' => {
                take(&mut i, 1)?;
                top.push(Val::Other)
            }
            b'M' => {
                take(&mut i, 2)?;
                top.push(Val::Other)
            }
            b'J' => {
                take(&mut i, 4)?;
                top.push(Val::Other)
            }
            0x8a => {
                let n = take(&mut i, 1)?[0] as usize;
                take(&mut i, n)?;
                top.push(Val::Other)
            }
            b'G' => {
                take(&mut i, 8)?;
                top.push(Val::Other)
            }
            b't' => {
                let items = frames.pop()?;
                frames.last_mut()?.push(Val::Tuple(items))
            }
            0x85 | 0x86 | 0x87 => {
                let n = (op - 0x84) as usize;
                let at = top.len().checked_sub(n)?;
                let items = top.split_off(at);
                top.push(Val::Tuple(items))
            }
            b'Q' => {
                if let Some(Val::Tuple(t)) = top.pop() {
                    if let [Val::Str(kind), Val::Global(ty), Val::Str(key), ..] = t.as_slice() {
                        if kind == b"storage" {
                            out.insert(key.clone(), width_of(ty));
                        }
                    }
                }
                top.push(Val::Other)
            }
            b'R' | 0x81 => {
                top.pop();
                top.pop();
                top.push(Val::Other)
            }
            b's' => {
                top.pop();
                top.pop();
            }
            b'a' | b'b' => {
                top.pop();
            }
            b'u' | b'e' => {
                frames.pop()?;
            }
            b'.' => break,
            _ => return None,
        }
        if frames.is_empty() {
            return None;
        }
    }
    Some(out)
}

fn width_of(ty: &[u8]) -> usize {
    match ty {
        b"torch.DoubleStorage" | b"torch.LongStorage" => 8,
        b"torch.FloatStorage" | b"torch.IntStorage" => 4,
        b"torch.BFloat16Storage" | b"torch.HalfStorage" | b"torch.ShortStorage" => 2,
        _ => 1,
    }
}

/// The part of an entry's name after the archive's top directory
/// (`step00100/data/7` -> `data/7`): the same across checkpoints of one run.
pub fn inner_name(name: &[u8]) -> &[u8] {
    name.iter().position(|&b| b == b'/').map_or(name, |p| &name[p + 1..])
}

/// For a checkpoint's zip entries (name, data start, end, method): each
/// stored storage entry's element width (0 for any other entry), or
/// `None` when this is not a PyTorch checkpoint.
pub fn entry_widths(input: &[u8], entries: &[(&[u8], usize, usize, usize)]) -> Option<Vec<usize>> {
    let pkl = entries.iter().find(|e| inner_name(e.0) == b"data.pkl" && e.3 == 0)?;
    let widths = storage_widths(input.get(pkl.1..pkl.2)?)?;
    Some(
        entries
            .iter()
            .map(|&(name, data, end, method)| {
                let inner = inner_name(name);
                match inner.strip_prefix(b"data/") {
                    Some(key) if method == 0 => widths.get(key).copied().filter(|&w| w >= 2 && (end - data) % w == 0 && end - data >= 64).unwrap_or(0),
                    _ => 0,
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_storages_torch_save_writes() {
        // data.pkl of torch.save({"model": {"w": f32 [2, 4], "h": bf16 [3],
        // "i": int64 [2]}, "optimizer": AdamW's state_dict after a step,
        // "step": 7}) (PyTorch 2.14): storages 0-8.
        const PKL: &[u8] = b"\x80\x02\x7d\x71\x00\x28\x58\x05\x00\x00\x00\x6d\x6f\x64\x65\x6c\x71\x01\x7d\x71\x02\x28\x58\x01\x00\x00\x00\x77\x71\x03\x63\x74\x6f\x72\x63\x68\x2e\x5f\x75\x74\x69\x6c\x73\x0a\x5f\x72\x65\x62\x75\x69\x6c\x64\x5f\x74\x65\x6e\x73\x6f\x72\x5f\x76\x32\x0a\x71\x04\x28\x28\x58\x07\x00\x00\x00\x73\x74\x6f\x72\x61\x67\x65\x71\x05\x63\x74\x6f\x72\x63\x68\x0a\x46\x6c\x6f\x61\x74\x53\x74\x6f\x72\x61\x67\x65\x0a\x71\x06\x58\x01\x00\x00\x00\x30\x71\x07\x58\x03\x00\x00\x00\x63\x70\x75\x71\x08\x4b\x08\x74\x71\x09\x51\x4b\x00\x4b\x02\x4b\x04\x86\x71\x0a\x4b\x04\x4b\x01\x86\x71\x0b\x89\x63\x63\x6f\x6c\x6c\x65\x63\x74\x69\x6f\x6e\x73\x0a\x4f\x72\x64\x65\x72\x65\x64\x44\x69\x63\x74\x0a\x71\x0c\x29\x52\x71\x0d\x74\x71\x0e\x52\x71\x0f\x58\x01\x00\x00\x00\x68\x71\x10\x68\x04\x28\x28\x68\x05\x63\x74\x6f\x72\x63\x68\x0a\x42\x46\x6c\x6f\x61\x74\x31\x36\x53\x74\x6f\x72\x61\x67\x65\x0a\x71\x11\x58\x01\x00\x00\x00\x31\x71\x12\x68\x08\x4b\x03\x74\x71\x13\x51\x4b\x00\x4b\x03\x85\x71\x14\x4b\x01\x85\x71\x15\x89\x68\x0c\x29\x52\x71\x16\x74\x71\x17\x52\x71\x18\x58\x01\x00\x00\x00\x69\x71\x19\x68\x04\x28\x28\x68\x05\x63\x74\x6f\x72\x63\x68\x0a\x4c\x6f\x6e\x67\x53\x74\x6f\x72\x61\x67\x65\x0a\x71\x1a\x58\x01\x00\x00\x00\x32\x71\x1b\x68\x08\x4b\x02\x74\x71\x1c\x51\x4b\x00\x4b\x02\x85\x71\x1d\x4b\x01\x85\x71\x1e\x89\x68\x0c\x29\x52\x71\x1f\x74\x71\x20\x52\x71\x21\x75\x58\x09\x00\x00\x00\x6f\x70\x74\x69\x6d\x69\x7a\x65\x72\x71\x22\x7d\x71\x23\x28\x58\x05\x00\x00\x00\x73\x74\x61\x74\x65\x71\x24\x7d\x71\x25\x28\x4b\x00\x7d\x71\x26\x28\x58\x04\x00\x00\x00\x73\x74\x65\x70\x71\x27\x68\x04\x28\x28\x68\x05\x68\x06\x58\x01\x00\x00\x00\x33\x71\x28\x68\x08\x4b\x01\x74\x71\x29\x51\x4b\x00\x29\x29\x89\x68\x0c\x29\x52\x71\x2a\x74\x71\x2b\x52\x71\x2c\x58\x07\x00\x00\x00\x65\x78\x70\x5f\x61\x76\x67\x71\x2d\x68\x04\x28\x28\x68\x05\x68\x06\x58\x01\x00\x00\x00\x34\x71\x2e\x68\x08\x4b\x08\x74\x71\x2f\x51\x4b\x00\x4b\x02\x4b\x04\x86\x71\x30\x4b\x04\x4b\x01\x86\x71\x31\x89\x68\x0c\x29\x52\x71\x32\x74\x71\x33\x52\x71\x34\x58\x0a\x00\x00\x00\x65\x78\x70\x5f\x61\x76\x67\x5f\x73\x71\x71\x35\x68\x04\x28\x28\x68\x05\x68\x06\x58\x01\x00\x00\x00\x35\x71\x36\x68\x08\x4b\x08\x74\x71\x37\x51\x4b\x00\x4b\x02\x4b\x04\x86\x71\x38\x4b\x04\x4b\x01\x86\x71\x39\x89\x68\x0c\x29\x52\x71\x3a\x74\x71\x3b\x52\x71\x3c\x75\x4b\x01\x7d\x71\x3d\x28\x68\x27\x68\x04\x28\x28\x68\x05\x68\x06\x58\x01\x00\x00\x00\x36\x71\x3e\x68\x08\x4b\x01\x74\x71\x3f\x51\x4b\x00\x29\x29\x89\x68\x0c\x29\x52\x71\x40\x74\x71\x41\x52\x71\x42\x68\x2d\x68\x04\x28\x28\x68\x05\x68\x06\x58\x01\x00\x00\x00\x37\x71\x43\x68\x08\x4b\x02\x74\x71\x44\x51\x4b\x00\x4b\x02\x85\x71\x45\x4b\x01\x85\x71\x46\x89\x68\x0c\x29\x52\x71\x47\x74\x71\x48\x52\x71\x49\x68\x35\x68\x04\x28\x28\x68\x05\x68\x06\x58\x01\x00\x00\x00\x38\x71\x4a\x68\x08\x4b\x02\x74\x71\x4b\x51\x4b\x00\x4b\x02\x85\x71\x4c\x4b\x01\x85\x71\x4d\x89\x68\x0c\x29\x52\x71\x4e\x74\x71\x4f\x52\x71\x50\x75\x75\x58\x0c\x00\x00\x00\x70\x61\x72\x61\x6d\x5f\x67\x72\x6f\x75\x70\x73\x71\x51\x5d\x71\x52\x7d\x71\x53\x28\x58\x02\x00\x00\x00\x6c\x72\x71\x54\x47\x3f\x50\x62\x4d\xd2\xf1\xa9\xfc\x58\x05\x00\x00\x00\x62\x65\x74\x61\x73\x71\x55\x47\x3f\xec\xcc\xcc\xcc\xcc\xcc\xcd\x47\x3f\xef\xf7\xce\xd9\x16\x87\x2b\x86\x71\x56\x58\x03\x00\x00\x00\x65\x70\x73\x71\x57\x47\x3e\x45\x79\x8e\xe2\x30\x8c\x3a\x58\x0c\x00\x00\x00\x77\x65\x69\x67\x68\x74\x5f\x64\x65\x63\x61\x79\x71\x58\x47\x3f\x84\x7a\xe1\x47\xae\x14\x7b\x58\x07\x00\x00\x00\x61\x6d\x73\x67\x72\x61\x64\x71\x59\x89\x58\x08\x00\x00\x00\x6d\x61\x78\x69\x6d\x69\x7a\x65\x71\x5a\x89\x58\x07\x00\x00\x00\x66\x6f\x72\x65\x61\x63\x68\x71\x5b\x4e\x58\x0a\x00\x00\x00\x63\x61\x70\x74\x75\x72\x61\x62\x6c\x65\x71\x5c\x89\x58\x0e\x00\x00\x00\x64\x69\x66\x66\x65\x72\x65\x6e\x74\x69\x61\x62\x6c\x65\x71\x5d\x89\x58\x05\x00\x00\x00\x66\x75\x73\x65\x64\x71\x5e\x4e\x58\x16\x00\x00\x00\x64\x65\x63\x6f\x75\x70\x6c\x65\x64\x5f\x77\x65\x69\x67\x68\x74\x5f\x64\x65\x63\x61\x79\x71\x5f\x88\x58\x06\x00\x00\x00\x70\x61\x72\x61\x6d\x73\x71\x60\x5d\x71\x61\x28\x4b\x00\x4b\x01\x65\x75\x61\x75\x68\x27\x4b\x07\x75\x2e";
        let w = storage_widths(PKL).unwrap();
        let got: Vec<usize> = (0..9).map(|k| w[k.to_string().as_bytes()]).collect();
        assert_eq!(got, vec![4, 2, 8, 4, 4, 4, 4, 4, 4]);
        assert!(storage_widths(b"\x80\x02\xff").is_none(), "an opcode it does not know");
        for cut in 1..PKL.len() {
            let _ = storage_widths(&PKL[..cut]);
        }
    }
}
