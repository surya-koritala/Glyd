# Install Glyd's engine on Windows under $env:GLYD_QWEN (default ~\.glyd-qwen-3.8-flash): glyd-engine.exe and its GPU library, built from
# this repository's source. Windows 10/11 x64, an NVIDIA GPU (Ampere or newer) with driver 580+. No admin rights.
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 [VERSION]
# VERSION: a Glyd release whose assets hold glyd-engine.exe, libglyd_gpu_cuda13.dll and SHA256SUMS (a release built for Windows);
# default: Ibis's last Windows build (v0.2.4), until Glyd's release workflow builds Windows too (ROADMAP.md).
param([string]$Version = 'latest')
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $PSScriptRoot
$H = if ($env:GLYD_QWEN) { $env:GLYD_QWEN } else { Join-Path $HOME '.glyd-qwen-3.8-flash' }
New-Item -ItemType Directory -Force (Join-Path $H 'bin') | Out-Null
$Base = if ($Version -eq 'latest') { 'https://github.com/surya-koritala/ibis/releases/download/v0.2.4' } else { "https://github.com/surya-koritala/Glyd/releases/download/$Version" }
if ($env:GLYD_RELEASE_URL) { $Base = $env:GLYD_RELEASE_URL }   # (a mirror, or a local folder for a test before a release)
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) { throw 'no nvidia-smi: an NVIDIA driver is needed' }
$files = 'glyd-engine.exe', 'libglyd_gpu_cuda13.dll', 'SHA256SUMS'
foreach ($f in $files) {
    $to = Join-Path $H "$f.new"
    if (Test-Path $Base) { Copy-Item (Join-Path $Base $f) $to } else { curl.exe -fsSL --retry 3 -o $to "$Base/$f"; if ($LASTEXITCODE) { throw "downloading $f failed" } }
}
# (SHA256SUMS lists every platform's files: the two here must be in it and match)
$checked = 0
foreach ($line in Get-Content (Join-Path $H 'SHA256SUMS.new')) {
    $want, $name = $line -split '\s+\*?', 2
    if ($files -contains $name) {
        $got = (Get-FileHash -Algorithm SHA256 (Join-Path $H "$name.new")).Hash.ToLower()
        if ($got -ne $want.ToLower()) { throw "$name does not match SHA256SUMS" }
        $checked++
    }
}
if ($checked -ne 2) { throw 'SHA256SUMS does not list glyd-engine.exe and libglyd_gpu_cuda13.dll' }
foreach ($f in 'glyd-engine.exe', 'libglyd_gpu_cuda13.dll') { Move-Item -Force (Join-Path $H "$f.new") (Join-Path $H "bin\$f") }
Move-Item -Force (Join-Path $H 'SHA256SUMS.new') (Join-Path $H 'SHA256SUMS')
@"
`$env:GLYD_QWEN = '$H'
`$env:GLYD_REPO = '$Here'
`$env:GLYD_ENGINE = '$H\bin\glyd-engine.exe'
"@ | Set-Content -Encoding utf8 (Join-Path $H 'env.ps1')
& (Join-Path $H 'bin\glyd-engine.exe') --version
'installed. Next: tools\download.ps1, then tools\serve.ps1'
