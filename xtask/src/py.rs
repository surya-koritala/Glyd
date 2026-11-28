//! What the Python scripts these tasks replace did by default, where a file's bytes or a message depend on it.

use crate::{fail, Res};
use std::path::Path;

/// A string as Python's repr() writes it: single quotes, or double when it holds a single quote and no double one.
pub fn repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut o = String::from(q);
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == q => {
                o.push('\\');
                o.push(c)
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => o.push_str(&format!("\\x{:02x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push(q);
    o
}

/// str.strip()'s whitespace: Unicode's, and the four separators U+001C to U+001F.
fn space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn strip(s: &str) -> &str {
    s.trim_matches(space)
}

/// str.lstrip()
pub fn strip_start(s: &str) -> &str {
    s.trim_start_matches(space)
}

/// A text file as Python reads one: UTF-8, every line ending a "\n".
pub fn read(path: &Path) -> Res<String> {
    let bytes = std::fs::read(path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    let text = String::from_utf8(bytes).map_err(|_| fail(format!("{}: not UTF-8", path.display())))?;
    Ok(if text.contains('\r') { text.replace("\r\n", "\n").replace('\r', "\n") } else { text })
}

pub fn write(path: &Path, text: &str) -> Res<()> {
    std::fs::write(path, text).map_err(|e| fail(format!("{}: {e}", path.display())))
}
