//! The tests' records of what a Python script did (tests/*/NAME.txt): sections headed `--- KIND [NAME]`, a text's lines after it, each starting with `|` (the last
//! with `\` when no newline ends it), <CR> for a carriage return.

use std::path::{Path, PathBuf};

pub struct Record(pub Vec<(String, String)>);

pub fn folder(of: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join(of)
}

pub fn read(path: &Path) -> Record {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut all: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        if let Some(head) = line.strip_prefix("--- ") {
            all.push((head.to_string(), String::new()));
        } else {
            let body = &mut all.last_mut().expect("a record starts with a heading").1;
            *body += &line[1..].replace("<CR>", "\r");
            if line.starts_with('|') {
                body.push('\n');
            }
        }
    }
    Record(all)
}

/// The records of a folder whose names start with `prefix`, by name.
pub fn cases(of: &str, prefix: &str) -> Vec<(String, Record)> {
    let mut names: Vec<String> = std::fs::read_dir(folder(of)).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    names.iter().filter(|n| n.starts_with(prefix) && n.ends_with(".txt")).map(|n| (n.trim_end_matches(".txt").to_string(), read(&folder(of).join(n)))).collect()
}

impl Record {
    /// The text of the one section headed so.
    pub fn one(&self, head: &str) -> &str {
        self.0.iter().find(|(h, _)| h == head).map_or("", |(_, t)| t)
    }

    /// The text of `--- IN name`.
    pub fn file(&self, name: &str) -> &str {
        self.one(&format!("IN {name}"))
    }

    pub fn exit(&self) -> i32 {
        self.one("EXIT").trim().parse().unwrap()
    }
}
