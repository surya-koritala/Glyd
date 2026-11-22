# One CUDA toolkit on a Windows runner, for release-gpu.yml: NVIDIA's network installer for VERSION, checked against SHA256, with only what nvcc
# needs to build the engine's kernels and the GPU library (the compiler, the runtime's headers and static library; from CUDA 13 the crt headers and
# NVVM are their own packages). (Install and Probe write with Write-Host: a function's output is its return value in PowerShell.) A probe .cu with the headers both use is compiled and linked; where that fails (a package named differently in a later
# installer), the whole toolkit is installed, as before.
#   pwsh .github/scripts/cuda-windows.ps1 -Version 13.0.3 -Sha256 6b54...
param([Parameter(Mandatory)][string]$Version, [Parameter(Mandatory)][string]$Sha256)
$ErrorActionPreference = 'Stop'
$mm = ($Version -split '\.')[0..1] -join '.'
$exe = "cuda_${Version}_windows_network.exe"
$t0 = Get-Date
Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/$Version/network_installers/$exe" -OutFile $exe
if ((Get-FileHash $exe -Algorithm SHA256).Hash.ToLower() -ne $Sha256) { throw "$exe is not the installer this workflow was written for" }
function Install($what) {
    $p = Start-Process -FilePath ".\$exe" -ArgumentList (@('-s') + $what) -Wait -PassThru
    Write-Host "$exe $($what -join ' '): exit code $($p.ExitCode)"
    return $p.ExitCode -eq 0
}
$parts = @("nvcc_$mm", "cudart_$mm")
if ([int]($mm -split '\.')[0] -ge 13) { $parts += @("crt_$mm", "nvvm_$mm") }
$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$mm\bin\nvcc.exe"
function Probe {
    if (-not (Test-Path $nvcc)) { return $false }
    $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
    $dir = Join-Path ([IO.Path]::GetTempPath()) "cuda-probe-$mm"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Set-Content "$dir\probe.cu" "#include <cuda.h>`n#include <cudaTypedefs.h>`n#include <cuda_runtime.h>`n#include <cuda_bf16.h>`n#include <mma.h>`n__global__ void k(__nv_bfloat16* x) { x[threadIdx.x] = __float2bfloat16(1.f); }`nint main() { return cudaGetLastError() == cudaSuccess ? 0 : 1; }"
    cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && `"$nvcc`" -allow-unsupported-compiler -arch=sm_80 -cudart static -o `"$dir\probe.exe`" `"$dir\probe.cu`"" | Write-Host
    return $LASTEXITCODE -eq 0
}
if ((Install $parts) -and (Probe)) {
    "CUDA $Version (nvcc and the runtime only) in $([int]((Get-Date) - $t0).TotalSeconds) s"
} else {
    "the probe did not build with $($parts -join ' '): installing the whole toolkit"
    if (-not (Install @())) { throw "$exe failed" }
    if (-not (Probe)) { throw "nvcc $Version does not build the probe" }
    "CUDA $Version (the whole toolkit) in $([int]((Get-Date) - $t0).TotalSeconds) s"
}
