//! R15: the kernels' types are read from their sources by build.rs (build/kernel_types.rs). The reading, on sources made to look as kernels/*.cu do.

#[path = "../build/kernel_types.rs"]
#[allow(dead_code)]
mod kernel_types;

fn args_of(out: &str, kernel: &str) -> String {
    let at = out.find(&format!("struct {kernel}(")).unwrap_or_else(|| panic!("{kernel} not made: {out}"));
    let line = out[at..].lines().find(|l| l.contains("type Args")).expect("its arguments");
    line.trim().trim_start_matches("type Args = ").trim_end_matches(';').to_string()
}

#[test]
fn a_kernel_is_read_with_its_arguments_types() {
    let src = r#"
// a comment with extern "C" __global__ void not_a_kernel(int x) in it
/* and a block
   extern "C" __global__ void neither(int y) */
struct G4Pf { int a; };
extern "C" __global__ void __launch_bounds__(256, 2) one(const bf16* __restrict__ x, int n, float eps, unsigned m, long long o, size_t z, G4Pf pf) {
    if (n) return;
}
extern "C" __global__ void two(
    unsigned long long a,   // a comment between arguments
    const unsigned char* b) {}
__device__ void helper(int q) {}
"#;
    let out = kernel_types::generate(&[("x.cu".to_string(), src.to_string())]).unwrap();
    assert_eq!(args_of(&out, "one"), "(u64, i32, f32, u32, i64, u64, crate::gemma4::G4Pf,)");
    assert_eq!(args_of(&out, "two"), "(u64, u64,)");
    assert!(!out.contains("not_a_kernel") && !out.contains("neither") && !out.contains("helper"), "{out}");
    assert!(out.contains("pub const COUNT: usize = 2;"));
}

#[test]
fn the_kernels_a_macro_makes_are_read_with_their_pasted_names() {
    let src = "#define UP(B, S, BLK) \\\n    extern \"C\" __global__ void __launch_bounds__(128, BLK) up##B##_##S(const int* tok, int n) { body<B, S>(tok, n); }\nUP(2, 3, 1)\nUP(4, 2, 2)\n";
    let out = kernel_types::generate(&[("m.cu".to_string(), src.to_string())]).unwrap();
    assert_eq!((args_of(&out, "up2_3"), args_of(&out, "up4_2")), ("(u64, i32,)".to_string(), "(u64, i32,)".to_string()));
    assert!(out.contains("pub const COUNT: usize = 2;"), "{out}");
}

#[test]
fn a_type_with_no_rust_type_or_a_name_twice_stops_the_build() {
    let e = kernel_types::generate(&[("a.cu".to_string(), "extern \"C\" __global__ void k(half2 h) {}".to_string())]).unwrap_err();
    assert!(e.contains("half2") && e.contains("a.cu"), "{e}");
    let k = "extern \"C\" __global__ void k(int n) {}".to_string();
    let e = kernel_types::generate(&[("a.cu".to_string(), k.clone()), ("b.cu".to_string(), k)]).unwrap_err();
    assert!(e.contains("a.cu") && e.contains("b.cu"), "{e}");
}
