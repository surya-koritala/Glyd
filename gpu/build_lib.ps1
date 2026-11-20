# glyd_gpu.cu's kernels as a DLL behind its C API on Windows: build_lib.sh's twin (the same kernels, architectures and
# release rules; MSVC is nvcc's host compiler, so run it from a Developer PowerShell or after vcvars64). The CUDA runtime
# is linked in, so the DLL needs only the driver. The name carries the CUDA major version: libglyd_gpu_cuda13.dll (the .so's name).
#   powershell -ExecutionPolicy Bypass -File gpu\build_lib.ps1 [OUT_DIR]      (default: next to this script)
# GLYD_GPU_RELEASE=1: machine code for sm_80/86/89/90a/100/120 and Ampere's PTX, an nvcc without Blackwell refused; kernel
# names are the source's in every build. The DLL exports the functions of glyd_gpu.h by their names (__declspec(dllexport) in glyd_gpu.cu).
param([string]$Out = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$nvcc = (Get-Command nvcc.exe).Source
$major = [regex]::Match((& $nvcc --version | Out-String), 'release (\d+)\.').Groups[1].Value
$archs = & $nvcc --list-gpu-arch
$blackwell = @()
foreach ($a in 100, 120) {
    if ($archs -contains "compute_$a") { $blackwell += '-gencode', "arch=compute_$a,code=sm_$a" }
    elseif ($env:GLYD_GPU_RELEASE -eq '1') { throw "glyd_gpu: this nvcc has no compute_$a (CUDA 12.8 on): a release library needs it" }
}
$ptx = @('-gencode', 'arch=compute_80,code=compute_80')
$release = @()
$tmp = Join-Path $env:TEMP "glyd_gpu_$PID"
New-Item -ItemType Directory -Force $tmp | Out-Null
$flags = @('-O3', '-std=c++20', '--expt-relaxed-constexpr',
    '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
    '-gencode', 'arch=compute_80,code=sm_80', '-gencode', 'arch=compute_86,code=sm_86', '-gencode', 'arch=compute_89,code=sm_89',
    '-gencode', 'arch=compute_90a,code=sm_90a') + $blackwell + $ptx + $release
$threads = if ($env:GLYD_GPU_NVCC_THREADS) { $env:GLYD_GPU_NVCC_THREADS } else { '0' }
& $nvcc @flags --threads $threads -c -o "$tmp\glyd_gpu.obj" "$PSScriptRoot\glyd_gpu.cu"
if ($LASTEXITCODE) { throw 'nvcc: the compile failed' }
& $nvcc @flags -shared -cudart static -o "$Out\libglyd_gpu_cuda$major.dll" "$tmp\glyd_gpu.obj"
if ($LASTEXITCODE) { throw 'nvcc: the link failed' }
Remove-Item -Recurse -Force $tmp
"$Out\libglyd_gpu_cuda$major.dll"
