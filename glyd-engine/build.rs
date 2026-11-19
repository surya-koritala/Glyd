// The engine's own CUDA kernels (kernels/engine.cu) compiled by nvcc into a fatbin that the binary embeds and loads
// with the driver (cuModuleLoadData): no CUDA runtime is linked, and no nvcc is needed to build the rest. Without nvcc
// the fatbin is empty and the CUDA paths report that the build has no kernels (`cargo check` works on any machine).
//
// GLYD_ENGINE_ARCHS: the architectures to build for, as nvcc's numbers (default "80 86 89 90a", and the compute_80 PTX); NVCC: the compiler.
// GLYD_ENGINE_RELEASE=1 (release.yml): the machine code alone, for 80, 86, 89, 90a, 100 and 120 (no PTX); a build without nvcc, or with an nvcc that has no Blackwell, is an error.
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
    let ok = cmd.arg("kernels/engine.cu").status().unwrap_or_else(|e| panic!("{}: {e} (GLYD_HIP=1 builds the AMD kernels with hipcc: set HIPCC)", hipcc.display()));
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

fn main() {
    println!("cargo:rerun-if-changed=kernels/engine.cu");
    println!("cargo:rerun-if-changed=kernels/qwen35.cu");
    println!("cargo:rerun-if-changed=kernels/moe.cu");
    println!("cargo:rerun-if-changed=kernels/moe_rows.cu");
    println!("cargo:rerun-if-changed=kernels/moe_hb.cu");
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
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out = out_dir.join("engine.fatbin");
    let release = env::var_os("GLYD_ENGINE_RELEASE").is_some_and(|v| v == "1");
    if release && env::var_os("GLYD_ENGINE_LINEINFO").is_some() {
        panic!("GLYD_ENGINE_RELEASE=1 with GLYD_ENGINE_LINEINFO: a release build has no line information in its machine code");
    }
    if env::var_os("GLYD_HIP").is_some_and(|v| v == "1") {
        hip(&out);
        return;
    }
    let Some(nvcc) = find_nvcc() else {
        if release {
            panic!("GLYD_ENGINE_RELEASE=1: no nvcc found (set NVCC or CUDA_HOME): a release build has its CUDA kernels");
        }
        std::fs::write(&out, b"").unwrap();
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
    }
    // the fatbin compressed (nvcc 12.4 on; the driver decompresses at load): about a third of the size; where this nvcc has no such flag, as it is
    let build = |compress: bool| {
        let mut cmd = Command::new(&nvcc);
        cmd.args(["-fatbin", "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-o"]).arg(&out);
        if env::var_os("GLYD_ENGINE_LINEINFO").is_some() {
            cmd.arg("-lineinfo"); // (for a profiler: it makes the fatbin several times larger)
        }
        if compress {
            cmd.arg("--compress-mode=size");
        }
        for a in archs.split_whitespace() {
            cmd.arg("-gencode").arg(format!("arch=compute_{a},code=sm_{a}"));
        }
        // and the Ampere PTX, which a driver compiles for a GPU newer than these (not in a release build: the machine code alone)
        if !release && archs.split_whitespace().any(|a| a == "80") {
            cmd.arg("-gencode").arg("arch=compute_80,code=compute_80");
        }
        cmd.arg("kernels/engine.cu");
        cmd.status().unwrap_or_else(|e| panic!("{}: {e}", nvcc.display()))
    };
    if !build(true).success() && !build(false).success() {
        panic!("nvcc failed on kernels/engine.cu");
    }
}
