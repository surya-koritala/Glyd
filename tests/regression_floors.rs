use simd_stream_codec::{compress, compress_parallel, decompress, decompress_parallel};
use std::path::Path;

#[test]
fn test_silesia_ratio_regression_floors() {
    let corpus_dir = Path::new("corpus");
    if !corpus_dir.exists() {
        eprintln!("Corpus directory not found, skipping ratio floor test");
        return;
    }

    let files = [
        ("dickens", 1.25),
        ("mozilla", 1.65),
        ("mr", 1.45),
        ("nci", 5.00),
        ("ooffice", 1.25),
        ("osdb", 1.95),
        ("reymont", 1.45),
        ("samba", 2.20),
        ("sao", 1.02),
        ("webster", 1.65),
        ("xml", 3.50),
        // GOAL3 section 3: liblz4 itself reaches 1.010 here; the dense retry that
        // gave 1.08 is retired for speed and the fast level will restore it.
        ("x-ray", 1.00),
    ];

    let mut total_orig = 0usize;
    let mut total_comp = 0usize;

    for (name, min_ratio) in &files {
        let path = corpus_dir.join(name);
        if !path.exists() {
            continue;
        }

        let data = std::fs::read(&path).unwrap();
        let compressed = compress(&data);

        // Verification of correctness
        let restored = decompress(&compressed).expect("Sequential roundtrip failed");
        assert_eq!(restored, data, "Roundtrip corruption on {}", name);

        let ratio = data.len() as f64 / compressed.len() as f64;
        println!("Corpus {:<10}: ratio {:.2}x (floor: {:.2}x)", name, ratio, min_ratio);
        assert!(
            ratio >= *min_ratio,
            "Ratio regression on {}: got {:.2}x, floor is {:.2}x",
            name,
            ratio,
            min_ratio
        );

        total_orig += data.len();
        total_comp += compressed.len();
    }

    if total_orig > 0 {
        let total_ratio = total_orig as f64 / total_comp as f64;
        // GOAL3 S1.3: the floor ratchets up to liblz4's 2.1009; being denser than
        // LZ4 is the whole point of beating it on speed.
        println!("TOTAL Silesia ratio: {:.4}x (floor: 2.1009x)", total_ratio);
        assert!(
            total_ratio >= 2.1009,
            "Total Silesia ratio regression: got {:.4}x, floor is 2.1009x",
            total_ratio
        );
    }
}

#[test]
fn test_parallel_pipeline_roundtrip_and_interop_floors() {
    let corpus_dir = Path::new("corpus");
    if !corpus_dir.exists() {
        return;
    }

    for file_name in &["mozilla", "nci", "samba"] {
        let path = corpus_dir.join(file_name);
        if !path.exists() {
            continue;
        }

        let data = std::fs::read(&path).unwrap();

        // 1. Parallel compress -> Parallel decompress
        let par_compressed = compress_parallel(&data);
        let par_restored = decompress_parallel(&par_compressed)
            .expect("Parallel decompress of parallel stream failed");
        assert_eq!(par_restored, data);

        // 2. Parallel compress -> Sequential decompress
        let seq_restored = decompress(&par_compressed)
            .expect("Sequential decompress of parallel stream failed");
        assert_eq!(seq_restored, data);

        // 3. Sequential compress -> Parallel decompress (must safely fall back to sequential)
        let seq_compressed = compress(&data);
        let fallback_restored = decompress_parallel(&seq_compressed)
            .expect("Parallel decompress fallback of sequential stream failed");
        assert_eq!(fallback_restored, data);
    }
}
