// R15: a type for each of the engine's kernels, made from its signature in kernels/*.cu, so that a launch of it with arguments of other types does not compile
// (`launch!` in src/dev.rs, `crate::kt`). Read as text, with no compiler: comments out, the lines a backslash continues joined, the kernels a macro makes (`C2_UP(2, 2, 2)`
// after `#define C2_UP(B, S, BLK) ... c2_up##B##_##S(...)`) expanded by putting the arguments in for the parameters. A type the table below does not know stops the build:
// a kernel's argument is a pointer (a device address, u64), a number, or one of the structs that src/ has as #[repr(C)].

use std::collections::BTreeMap;

/// The C types of the kernels' arguments that are not pointers, and the Rust type passed for each.
const TYPES: [(&str, &str); 30] = [
    ("int", "i32"),
    ("int32_t", "i32"),
    ("unsigned", "u32"),
    ("unsigned int", "u32"),
    ("uint32_t", "u32"),
    ("float", "f32"),
    ("double", "f64"),
    ("long long", "i64"),
    ("int64_t", "i64"),
    ("unsigned long long", "u64"),
    ("uint64_t", "u64"),
    ("size_t", "u64"),
    ("short", "i16"),
    ("unsigned short", "u16"),
    ("signed char", "i8"),
    ("unsigned char", "u8"),
    ("bool", "bool"),
    ("NextPf", "crate::kern::NextPf"),
    ("C2Segs", "crate::cb2_prefill::Segs"),
    ("AttnArgs", "crate::dynamic::attn::AttnArgs"),
    ("glib::kvc::AttnParams", "crate::fold::KvAttnParams"),
    ("glib::kvc::PrefillParams", "crate::fold::KvPrefillParams"),
    ("glib::kvc::WriteParams", "crate::fold::KvWriteParams"),
    ("glib::kvc::RawPool", "crate::fold::KvRawPool"),
    ("glib::kvc::EscapePool", "crate::fold::KvEscapePool"),
    ("glib::f8c::F8Pk", "crate::fold::F8Pk"),
    ("glib::f8c::F8Un", "crate::fold::F8Un"),
    ("glib::f8c::F8Lin", "crate::fold::F8Lin"),
    ("glib::MoePairs", "crate::fold::MoePairs"),
    ("CUtensorMap", "crate::device::TensorMap"),
];

/// The source without comments, its continued lines joined.
fn clean(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
        } else if b[i] == b'"' {
            // (a string, kept whole: `extern "C"`)
            let start = i;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += 1 + (b[i] == b'\\') as usize;
            }
            i += 1;
            out.push_str(&src[start..i.min(b.len())]);
        } else {
            let c = src[i..].chars().next().expect("a char");
            out.push(c);
            i += c.len_utf8();
        }
    }
    out.replace("\\\r\n", " ").replace("\\\n", " ")
}

/// The text from `open` (just after a parenthesis) to the one that closes it, and the place after it.
fn balanced(s: &str, open: usize) -> Option<(&str, usize)> {
    let mut depth = 1;
    for (j, c) in s[open..].char_indices() {
        depth += (c == '(') as i32 - (c == ')') as i32;
        if depth == 0 {
            return Some((&s[open..open + j], open + j + 1));
        }
    }
    None
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `s` with each parameter, as a whole word, replaced by its argument, and the pastes (`##`) joined.
fn substitute(s: &str, params: &[String], args: &[String]) -> String {
    let mut out = String::new();
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        match params.iter().position(|p| p == word) {
            Some(i) => out.push_str(&args[i]),
            None => out.push_str(word),
        }
        word.clear();
    };
    for c in s.chars() {
        if is_ident(c) {
            word.push(c);
        } else {
            flush(&mut word, &mut out);
            out.push(c);
        }
    }
    flush(&mut word, &mut out);
    let mut joined = out;
    while let Some(i) = joined.find("##") {
        let left = joined[..i].trim_end().len();
        let right = i + 2 + (joined[i + 2..].len() - joined[i + 2..].trim_start().len());
        joined = format!("{}{}", &joined[..left], &joined[right..]);
    }
    joined
}

/// The Rust type of one argument's declaration (`const bf16* __restrict__ x`, `int K`), or an error naming it.
fn rust_type(decl: &str) -> Result<String, String> {
    let words: Vec<&str> =
        decl.split_whitespace().filter(|w| !matches!(*w, "const" | "__restrict__" | "__restrict" | "volatile" | "__grid_constant__")).collect();
    let text = words.join(" ");
    if text.contains('*') {
        return Ok("u64".to_string());
    }
    let (ty, name) = text.rsplit_once(' ').ok_or_else(|| format!("an argument with no type: {decl:?}"))?;
    if name.is_empty() || !name.chars().all(is_ident) {
        return Err(format!("an argument not read: {decl:?}"));
    }
    TYPES
        .iter()
        .find(|(c, _)| *c == ty)
        .map(|(_, r)| r.to_string())
        .ok_or_else(|| format!("the C type {ty:?} (of {decl:?}) has no Rust type in build/kernel_types.rs"))
}

/// A kernel's name, its arguments' Rust types and its `__launch_bounds__`'s first argument (its most threads a block), where it has one.
type Found = (String, Vec<String>, Option<String>);

/// The text before the first comma outside parentheses.
fn first_arg(s: &str) -> &str {
    let mut depth = 0;
    for (j, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => return s[..j].trim(),
            _ => {}
        }
    }
    s.trim()
}

/// Each `extern "C" __global__ void` kernel of `code`: its name, its arguments' Rust types and its bound.
fn kernels_in(code: &str, out: &mut Vec<Found>) -> Result<(), String> {
    const HEAD: &str = "extern \"C\" __global__ void";
    let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut at = 0;
    while let Some(k) = flat[at..].find(HEAD) {
        let mut i = at + k + HEAD.len();
        let rest = flat[i..].trim_start();
        i = flat.len() - rest.len();
        let mut bound = None;
        if rest.starts_with("__launch_bounds__") {
            let open = i + flat[i..].find('(').ok_or("a __launch_bounds__ with no parenthesis")? + 1;
            let (text, next) = balanced(&flat, open).ok_or("an unclosed __launch_bounds__")?;
            (bound, i) = (Some(first_arg(text).to_string()), next);
        }
        let rest = flat[i..].trim_start();
        i = flat.len() - rest.len();
        let name: String = rest.chars().take_while(|&c| is_ident(c)).collect();
        if name.is_empty() {
            return Err(format!("a kernel with no name at {:?}", &rest[..rest.len().min(60)]));
        }
        let open = i + flat[i..].find('(').ok_or("a kernel with no arguments' list")? + 1;
        let (args, next) = balanced(&flat, open).ok_or_else(|| format!("{name}: an unclosed arguments' list"))?;
        let types = if args.trim().is_empty() || args.trim() == "void" {
            Vec::new()
        } else {
            args.split(',').map(rust_type).collect::<Result<Vec<_>, _>>().map_err(|e| format!("{name}: {e}"))?
        };
        out.push((name, types, bound));
        at = next;
    }
    Ok(())
}

/// The kernels of a source file, those its macros make included.
fn kernels_of(src: &str) -> Result<Vec<Found>, String> {
    let text = clean(src);
    let mut macros: BTreeMap<String, (Vec<String>, String)> = BTreeMap::new();
    let mut code = String::new();
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        if let Some(def) = t.strip_prefix("#define ") {
            // (a macro that makes kernels: kept, to be expanded where it is used)
            let def = def.trim_start();
            let name: String = def.chars().take_while(|&c| is_ident(c)).collect();
            if def[name.len()..].starts_with('(') && def.contains("__global__") {
                let (params, end) = balanced(def, name.len() + 1).ok_or_else(|| format!("#define {name}: an unclosed parameters' list"))?;
                let params = params.split(',').map(|p| p.trim().to_string()).collect();
                macros.insert(name, (params, def[end..].to_string()));
            }
            continue;
        }
        if t.starts_with('#') {
            continue;
        }
        let name: String = t.chars().take_while(|&c| is_ident(c)).collect();
        if let Some((params, body)) = macros.get(&name).filter(|_| t[name.len()..].starts_with('(')) {
            let (args, _) = balanced(t, name.len() + 1).ok_or_else(|| format!("{name}(...): unclosed"))?;
            let args: Vec<String> = args.split(',').map(|a| a.trim().to_string()).collect();
            if args.len() != params.len() {
                return Err(format!("{name}: {} arguments for {} parameters", args.len(), params.len()));
            }
            kernels_in(&substitute(body, params, &args), &mut out)?;
            continue;
        }
        code.push_str(line);
        code.push('\n');
    }
    kernels_in(&code, &mut out)?;
    Ok(out)
}

/// The Rust source of `crate::kt`: for each kernel of `sources` (file name, text), a type that holds it and says its arguments' types.
pub fn generate(sources: &[(String, String)]) -> Result<String, String> {
    let mut all: BTreeMap<String, (Vec<String>, String, Option<String>)> = BTreeMap::new();
    for (file, src) in sources {
        for (name, types, bound) in kernels_of(src).map_err(|e| format!("{file}: {e}"))? {
            if let Some((_, other, _)) = all.get(&name) {
                return Err(format!("the kernel {name} is in {other} and in {file}"));
            }
            all.insert(name, (types, file.clone(), bound));
        }
    }
    let mut out = String::from(
        "// Made by build.rs (build/kernel_types.rs) from kernels/*.cu: one type a kernel, with its arguments' types. Not edited by hand.\n\
         use crate::dev::{Kernel, Launch, Sig, Typed};\n\n",
    );
    for (name, (types, file, _)) in &all {
        let args = if types.is_empty() { "()".to_string() } else { format!("({},)", types.join(", ")) };
        out.push_str(&format!(
            "/// The kernel `{name}` of kernels/{file}.\n#[allow(non_camel_case_types)]\npub(crate) struct {name}(Kernel);\n\
             impl Typed for {name} {{\n    const NAME: &'static str = \"{name}\";\n    type Args = {args};\n    fn of(k: Kernel) -> Self {{\n        Self(k)\n    }}\n    fn kernel(&self) -> &Kernel {{\n        &self.0\n    }}\n}}\n\
             impl Launch<{args}> for {name} {{\n    fn kernel_of(&self) -> &Kernel {{\n        &self.0\n    }}\n}}\n\
             impl Launch<{args}> for &{name} {{\n    fn kernel_of(&self) -> &Kernel {{\n        &self.0\n    }}\n}}\n\
             impl std::ops::Deref for {name} {{\n    type Target = Kernel;\n    fn deref(&self) -> &Kernel {{\n        &self.0\n    }}\n}}\n\
             impl {name} {{\n    /// This kernel by its arguments' types alone (`dev::Sig`).\n    pub(crate) fn sig(&self) -> Sig<'_, {args}> {{\n        Sig::of(&self.0)\n    }}\n}}\n\n"
        ));
    }
    out.push_str(&format!("/// The kernels: {}.\npub const COUNT: usize = {};\n", all.len(), all.len()));
    // (each kernel with its file: a check's census of what it launched says which of a file's kernels it did not)
    out.push_str("/// Each kernel's name and its file of kernels/.\npub const NAMES: &[(&str, &str)] = &[\n");
    for (name, (_, file, _)) in &all {
        out.push_str(&format!("    (\"{name}\", \"{file}\"),\n"));
    }
    out.push_str("];\n");
    // (a launch's block is at most its kernel's bound: src/fold.rs's tests hold the prompt kernels' to it)
    out.push_str("/// Each kernel's `__launch_bounds__` as its source gives the most threads a block (a macro's arguments put in), where it has one.\npub const BOUNDS: &[(&str, &str)] = &[\n");
    for (name, (_, _, bound)) in &all {
        if let Some(b) = bound {
            out.push_str(&format!("    (\"{name}\", {b:?}),\n"));
        }
    }
    out.push_str("];\n");
    Ok(out)
}
