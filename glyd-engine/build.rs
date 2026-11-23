// The engine's own CUDA kernels (kernels/engine.cu) compiled by nvcc into a fatbin that the binary embeds and loads
// with the driver (cuModuleLoadData): no CUDA runtime is linked, and no nvcc is needed to build the rest. Without nvcc
// the fatbin is empty and the CUDA paths report that the build has no kernels (`cargo check` works on any machine).
//
// GLYD_ENGINE_ARCHS: the architectures to build for, as nvcc's numbers (default "80 86 89 90a", and the compute_80 PTX); NVCC: the compiler.
// GLYD_ENGINE_RELEASE=1 (release-gpu.yml): the machine code for 80, 86, 89, 90a, 100 and 120 (and 110, Jetson Thor, in an aarch64 build) and the compute_89 PTX (a GPU newer than these gets its code compiled by the driver); a build without nvcc,
// or with an nvcc that has no Blackwell, is an error.
// NVCC12: a CUDA 12.9 (or newer 12) nvcc, with the one in NVCC (CUDA 13) or on PATH: the kernels are built a second time, uncompressed (a fatbin compressed for size needs a driver of
// CUDA 12.4 or newer), as engine12.fatbin, which the binary loads on a driver that reports CUDA 12 (cfg `kernels12`: dev.rs picks by the driver's version, machine.rs says what the build runs on).
// Without NVCC12 the build carries the one nvcc's kernels, and the cfg `kernels12` where that nvcc is CUDA 12's.
// Every build keeps the names engine.cu gives its kernels (the source is open: a release build is this source as it reads).
//
// GLYD_HIP=1: AMD GPUs in place of NVIDIA's. kernels/engine.cu by hipcc (HIPCC: hipcc, or a script that runs a clang with HIP as hipcc would) as code objects for RDNA 3 and 4 in
// wave32 (GLYD_HIP_ARCHS, default gfx1100 gfx1101 gfx1200 gfx1201: the RX 7900, RX 7800/7700, RX 9060 and RX 9070 cards), bundled in the one file the HIP runtime loads (it takes
// the device's code object from the bundle), and the cfg `glyd_hip`, under which ffi.rs takes the driver's table from HIP's runtime. No nvcc is asked for.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The AMD build of the kernels (GLYD_HIP=1): one bundle of code objects at `out`.
fn hip(out: &Path) {
    let hipcc = PathBuf::from(env::var_os("HIPCC").unwrap_or_else(|| "hipcc".into()));
    let archs = env::var("GLYD_HIP_ARCHS").unwrap_or_else(|_| "gfx1100 gfx1101 gfx1200 gfx1201".to_string());
    let mut cmd = Command::new(&hipcc);
    // (the device code alone; -mno-wavefrontsize64: the kernels' warps are 32 lanes; the loops clang does not unroll are warned of by the hundred)
    cmd.args(["-x", "hip", "--cuda-device-only", "-O3", "-std=c++17", "-mno-wavefrontsize64", "-Wno-pass-failed", "-o"]).arg(out);
    for a in archs.split_whitespace() {
        cmd.arg(format!("--offload-arch={a}"));
    }
    let ok =
        cmd.arg("kernels/engine.cu").status().unwrap_or_else(|e| panic!("{}: {e} (GLYD_HIP=1 builds the AMD kernels with hipcc: set HIPCC)", hipcc.display()));
    assert!(ok.success(), "{} failed on kernels/engine.cu", hipcc.display());
    println!("cargo:rustc-cfg=glyd_hip");
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(n) = env::var_os("NVCC") {
        return Some(PathBuf::from(n));
    }
    // (nvcc.exe where the build runs on Windows)
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    if let Some(home) = env::var_os("CUDA_HOME").or_else(|| env::var_os("CUDA_PATH")) {
        let p = Path::new(&home).join("bin").join(exe);
        if p.exists() {
            return Some(p);
        }
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path).map(|d| d.join(exe)).find(|p| p.exists())
}

#[path = "build/kernel_types.rs"]
mod kernel_types;

/// `crate::kt` (src/lib.rs): a type for each kernel of kernels/*.cu, written to OUT_DIR on every build, with or without a CUDA compiler.
fn write_kernel_types() {
    println!("cargo:rerun-if-changed=build/kernel_types.rs");
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir("kernels")
        .expect("kernels/")
        .map(|e| e.expect("an entry of kernels/").path())
        .filter(|p| p.extension().is_some_and(|x| x == "cu"))
        .collect();
    files.sort();
    let sources: Vec<(String, String)> =
        files.iter().map(|p| (p.file_name().expect("a name").to_string_lossy().into_owned(), std::fs::read_to_string(p).expect("a kernel source"))).collect();
    let out = kernel_types::generate(&sources).unwrap_or_else(|e| panic!("the kernels' types: {e}"));
    let dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    std::fs::write(std::path::Path::new(&dir).join("kernel_types.rs"), out).expect("kernel_types.rs written");
}

fn main() {
    // R9's MLX spike (src/runtime/mlx.rs): mlx-c, where the `mlx` feature is on
    if env::var_os("CARGO_FEATURE_MLX").is_some() {
        let prefix = env::var("MLX_PREFIX").unwrap_or_else(|_| "/opt/homebrew".into());
        println!("cargo:rustc-link-search=native={prefix}/lib");
        println!("cargo:rerun-if-env-changed=MLX_PREFIX");
    }
    write_kernel_types();
    println!("cargo:rerun-if-changed=kernels/engine.cu");
    println!("cargo:rerun-if-changed=kernels/qwen35.cu");
    println!("cargo:rerun-if-changed=kernels/moe.cu");
    println!("cargo:rerun-if-changed=kernels/moe_rows.cu");
    println!("cargo:rerun-if-changed=kernels/moe_hb.cu");
    println!("cargo:rerun-if-changed=kernels/moe_hbg.cu");
    println!("cargo:rerun-if-changed=kernels/nemotron.cu");
    println!("cargo:rerun-if-changed=kernels/glm.cu");
    println!("cargo:rerun-if-changed=kernels/glm_cb.cu");
    println!("cargo:rerun-if-changed=kernels/kimi.cu");
    println!("cargo:rerun-if-changed=kernels/dsv4.cu");
    println!("cargo:rerun-if-changed=kernels/muse.cu");
    println!("cargo:rerun-if-changed=kernels/cb2_prefill.cu");
    println!("cargo:rerun-if-changed=kernels/skinny.cu");
    println!("cargo:rerun-if-changed=kernels/qwen4x.cu");
    println!("cargo:rerun-if-changed=kernels/vision.cu");
    println!("cargo:rerun-if-changed=kernels/gemma4.cu");
    println!("cargo:rerun-if-changed=kernels/attn.cu");
    println!("cargo:rerun-if-changed=kernels/tp.cu");
    println!("cargo:rerun-if-changed=kernels/gemm_idx.h");
    println!("cargo:rerun-if-changed=../gpu/hip.h");
    println!("cargo:rerun-if-changed=../gpu/warp_emu.h");
    println!("cargo:rerun-if-env-changed=GLYD_HIP");
    println!("cargo:rerun-if-env-changed=GLYD_HIP_ARCHS");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo::rustc-check-cfg=cfg(glyd_hip)");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=GLYD_ENGINE_ARCHS");
    println!("cargo:rerun-if-env-changed=GLYD_ENGINE_LINEINFO");
    println!("cargo:rerun-if-env-changed=GLYD_ENGINE_RELEASE");
    // GLYD_SOURCE_COMMIT: the commit the sources are (release.yml's), which /v1/glyd/info says; a build not given one says none
    println!("cargo:rerun-if-env-changed=GLYD_SOURCE_COMMIT");
    println!("cargo:rustc-env=GLYD_SOURCE_COMMIT={}", env::var("GLYD_SOURCE_COMMIT").unwrap_or_default());
    println!("cargo::rustc-check-cfg=cfg(no_kernels)");
    println!("cargo::rustc-check-cfg=cfg(kernels12)");
    println!("cargo:rerun-if-env-changed=NVCC12");
    println!("cargo:rerun-if-env-changed=GLYD_ENGINE_FATBIN_CACHE");
    println!("cargo:rerun-if-env-changed=GLYD_ENGINE_NVCC_THREADS");
    // GLYD_ENGINE_FATBIN_CACHE=DIR: a fatbin found there (engine.fatbin, engine12.fatbin) is taken in place of compiling it, and one compiled is left there.
    // The workflows key DIR's cache by the kernels' sources, this file and the toolkits, so a build whose kernels did not change skips nvcc; nothing else sets it.
    let cache = env::var_os("GLYD_ENGINE_FATBIN_CACHE").filter(|d| !d.is_empty()).map(PathBuf::from);
    // GLYD_ENGINE_NVCC_THREADS=N: nvcc compiles N architectures at once (--threads N; each takes its own memory)
    let threads = env::var("GLYD_ENGINE_NVCC_THREADS").ok().filter(|s| !s.trim().is_empty());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out = out_dir.join("engine.fatbin");
    std::fs::write(out_dir.join("engine12.fatbin"), b"").unwrap(); // (dev.rs embeds both; the second is empty unless NVCC12 built it)
    let release = env::var_os("GLYD_ENGINE_RELEASE").is_some_and(|v| v == "1");
    if release && env::var_os("GLYD_ENGINE_LINEINFO").is_some() {
        panic!("GLYD_ENGINE_RELEASE=1 with GLYD_ENGINE_LINEINFO: a release build has no line information in its machine code");
    }
    if env::var_os("GLYD_HIP").is_some_and(|v| v == "1") {
        println!("cargo:rustc-env=GLYD_ENGINE_SASS="); // (no NVIDIA machine code in an AMD build)
        hip(&out);
        return;
    }
    let Some(nvcc) = find_nvcc() else {
        if release {
            panic!("GLYD_ENGINE_RELEASE=1: no nvcc found (set NVCC or CUDA_HOME): a release build has its CUDA kernels");
        }
        std::fs::write(&out, b"").unwrap();
        println!("cargo:rustc-env=GLYD_ENGINE_SASS=");
        println!("cargo:rustc-cfg=no_kernels");
        println!("cargo:warning=glyd-engine: no nvcc found (set NVCC or CUDA_HOME): built without its CUDA kernels");
        return;
    };
    // the architectures this nvcc has (nvcc --list-gpu-arch: compute_80 ...): a release build takes Blackwell's too, and refuses an nvcc without them
    let known = Command::new(&nvcc).arg("--list-gpu-arch").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let has = |a: &str| known.lines().any(|l| l.trim() == format!("compute_{a}"));
    let mut archs = env::var("GLYD_ENGINE_ARCHS").unwrap_or_else(|_| "80 86 89 90a".to_string());
    if release && env::var_os("GLYD_ENGINE_ARCHS").is_none() {
        for a in ["100", "120"] {
            if !has(a) {
                panic!("GLYD_ENGINE_RELEASE=1: this nvcc has no compute_{a} (CUDA 12.8 on): a release build needs it");
            }
            archs.push_str(&format!(" {a}"));
        }
        // Jetson Thor (sm_110, CUDA 13) is an Arm computer: the aarch64 build takes its machine code, which an x86_64 build would carry for no GPU it can run on.
        // (sm_120 is built as it is, not as the family target sm_120f: for these kernels the two are the same machine code, byte for byte in size and in the list nvcc's cuobjdump gives,
        // and sm_120 code runs on compute capability 12.0 and 12.1, GB10 included; the sm_120a code that does not run on 12.1 is not built.)
        if env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|a| a == "aarch64") && has("110") {
            archs.push_str(" 110");
        }
    }
    // one fatbin of kernels/engine.cu by `nvcc` for `archs`: CUDA 13's compressed for size (nvcc 12.4 on knows the flag; where it fails, as it is), CUDA 12's not compressed at all
    // (a driver below CUDA 12.4 cannot read a compressed fatbin); the compute_89 PTX in a release build, and in a development build the compute_80 PTX
    let fatbin = |nvcc: &Path, out: &Path, archs: &str, compress: bool, uncompressed: bool| -> bool {
        let name = out.file_name().expect("a fatbin's name");
        if let Some(hit) = cache.as_ref().map(|c| c.join(name)).filter(|f| f.metadata().is_ok_and(|m| m.len() > 0)) {
            std::fs::copy(&hit, out).unwrap_or_else(|e| panic!("{}: {e}", hit.display()));
            println!("cargo:warning=glyd-engine: {} from GLYD_ENGINE_FATBIN_CACHE ({})", name.to_string_lossy(), hit.display());
            return true;
        }
        let mut cmd = Command::new(nvcc);
        if let Some(n) = &threads {
            cmd.arg("--threads").arg(n.trim());
        }
        cmd.args(["-fatbin", "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-o"]).arg(out);
        if env::var_os("GLYD_ENGINE_LINEINFO").is_some() {
            cmd.arg("-lineinfo"); // (for a profiler: it makes the fatbin several times larger)
        }
        if compress {
            cmd.arg("--compress-mode=size");
        } else if uncompressed {
            cmd.arg("--compress-mode=none");
        }
        for a in archs.split_whitespace() {
            cmd.arg("-gencode").arg(format!("arch=compute_{a},code=sm_{a}"));
        }
        if release {
            // (PTX for a GPU newer than these: the driver compiles it at load, where a release build would otherwise refuse the GPU)
            cmd.arg("-gencode").arg("arch=compute_89,code=compute_89");
        } else if archs.split_whitespace().any(|a| a == "80") {
            // (a development build: the Ampere PTX, which a driver compiles for a GPU newer than these)
            cmd.arg("-gencode").arg("arch=compute_80,code=compute_80");
        }
        cmd.arg("kernels/engine.cu");
        let ok = cmd.status().unwrap_or_else(|e| panic!("{}: {e}", nvcc.display())).success();
        if let (true, Some(c)) = (ok, &cache) {
            std::fs::create_dir_all(c).and_then(|_| std::fs::copy(out, c.join(name))).unwrap_or_else(|e| panic!("{}: {e}", c.display()));
        }
        ok
    };
    let major = |nvcc: &Path| {
        let v = Command::new(nvcc).arg("--version").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        v.split("release ").nth(1).and_then(|s| s.split('.').next()).and_then(|s| s.trim().parse::<u32>().ok()).unwrap_or(13)
    };
    let primary12 = major(&nvcc) < 13;
    if !fatbin(&nvcc, &out, &archs, !primary12, primary12) && !fatbin(&nvcc, &out, &archs, false, primary12) {
        panic!("nvcc failed on kernels/engine.cu");
    }
    // the second fatbin, for drivers that report CUDA 12 (empty without NVCC12)
    let out12 = out_dir.join("engine12.fatbin");
    std::fs::write(&out12, b"").unwrap();
    let mut kernels12 = primary12;
    if let Some(nvcc12) = env::var_os("NVCC12").map(PathBuf::from) {
        if major(&nvcc12) >= 13 {
            panic!("NVCC12={}: this nvcc is CUDA {}'s, not 12's", nvcc12.display(), major(&nvcc12));
        }
        let known12 = Command::new(&nvcc12).arg("--list-gpu-arch").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        // (Jetson Thor's sm_110 is CUDA 13's alone, and a driver that reports CUDA 12 runs no Thor: the CUDA 12 fatbin leaves it out)
        const CUDA13_ONLY: &[&str] = &["110"];
        let wanted12: Vec<&str> = archs.split_whitespace().filter(|a| !CUDA13_ONLY.contains(&a.trim_end_matches(|c| c == 'a' || c == 'f'))).collect();
        // (Blackwell's code needs CUDA 12.8: the build refuses an older one, as the CUDA 13 build refuses an nvcc without it)
        let archs12: Vec<&str> =
            wanted12.iter().copied().filter(|a| known12.lines().any(|l| l.trim() == format!("compute_{}", a.trim_end_matches('a')))).collect(); // (nvcc lists compute_90, and takes 90a)
        if release && archs12.len() != wanted12.len() {
            panic!(
                "GLYD_ENGINE_RELEASE=1: NVCC12 has no compute_{{{}}}: a release build needs CUDA 12.8 or newer",
                wanted12.iter().filter(|a| !archs12.contains(a)).copied().collect::<Vec<_>>().join(", ")
            );
        }
        if !fatbin(&nvcc12, &out12, &archs12.join(" "), false, true) {
            panic!("nvcc ({}) failed on kernels/engine.cu", nvcc12.display());
        }
        kernels12 = true;
    }
    if kernels12 {
        println!("cargo:rustc-cfg=kernels12");
    }
    // the machine code the binary carries, which the refusal of a GPU it has no code for says
    println!("cargo:rustc-env=GLYD_ENGINE_SASS={}", archs.split_whitespace().map(|a| format!("sm_{a}")).collect::<Vec<_>>().join(" "));
}
