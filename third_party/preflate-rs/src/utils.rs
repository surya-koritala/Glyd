#[allow(dead_code)]
#[cfg(test)]
pub fn write_file(filename: &str, data: &[u8]) {
    let filename = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("samples")
        .join(filename);

    let mut writecomp = std::fs::File::create(filename).unwrap();
    std::io::Write::write_all(&mut writecomp, data).unwrap();
}

#[cfg(test)]
pub fn read_file(filename: &str) -> Vec<u8> {
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;

    let filename = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("samples")
        .join(filename);
    println!("reading {0}", filename.to_str().unwrap());
    let mut f = File::open(filename).unwrap();

    let mut content = Vec::new();
    f.read_to_end(&mut content).unwrap();

    content
}

/// handy function to compare two arrays, and print the first mismatch. Useful for debugging.
#[cfg(test)]
#[track_caller]
pub fn assert_eq_array<T: PartialEq + std::fmt::Debug>(a: &[T], b: &[T]) {
    use core::panic;

    if a.len() != b.len() {
        for i in 0..std::cmp::min(a.len(), b.len()) {
            assert_eq!(
                a[i],
                b[i],
                "length mismatch {},{} and first mismatch at offset {}",
                a.len(),
                b.len(),
                i
            );
        }
        panic!(
            "length mismatch {} and {}, but common prefix identical",
            a.len(),
            b.len()
        );
    } else {
        for i in 0..a.len() {
            assert_eq!(
                a[i],
                b[i],
                "length identical {}, but first mismatch at offset {}",
                a.len(),
                i
            );
        }
    }
}

#[cfg(test)]
#[track_caller]
pub fn assert_block_eq(
    a: &crate::deflate::deflate_token::DeflateTokenBlock,
    b: &crate::deflate::deflate_token::DeflateTokenBlock,
) {
    use crate::deflate::deflate_token::DeflateTokenBlockType;

    if a == b {
        return;
    }

    if std::mem::discriminant(&a.block_type) != std::mem::discriminant(&b.block_type) {
        panic!("block type mismatch {:?} {:?}", a.block_type, b.block_type);
    }

    match (&a.block_type, &b.block_type) {
        (
            DeflateTokenBlockType::Huffman {
                tokens: ta,
                huffman_type: ha,
            },
            DeflateTokenBlockType::Huffman {
                tokens: tb,
                huffman_type: hb,
            },
        ) => {
            assert_eq_array(&ta, &tb);
            assert_eq!(ha, hb);
        }
        (
            DeflateTokenBlockType::Stored { uncompressed: ua },
            DeflateTokenBlockType::Stored { uncompressed: ub },
        ) => {
            assert_eq_array(&ua, &ub);
        }
        _ => {
            panic!("unexpected block type");
        }
    }
    assert_eq!(a.last, b.last, "last flag mismatch");
}

#[cfg(test)]
pub fn test_on_all_deflate_files(f: impl Fn(&[u8])) {
    use std::io::Read;

    let searchpath = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("samples");

    for entry in std::fs::read_dir(searchpath).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension().is_some() && path.extension().unwrap() == "deflate" {
            println!("Testing {:?}", path);

            let mut file = std::fs::File::open(&path).unwrap();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();

            f(&buffer);
        }
    }
}
