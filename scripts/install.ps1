# Glyd's installer for Windows x86_64:   irm https://getglyd.com/install.ps1 | iex
#
# Installs glyd.exe (Glyd's engine linked in: `glyd run`, `serve`, `pull`, `fit`, `doctor`) and its GPU libraries (CUDA 13 and CUDA 12) from the release's zip, checked against the
# sha256 the release lists, under %LOCALAPPDATA%\Glyd, puts that folder on your user PATH, and ends with `glyd doctor`. No admin rights, no Python. It needs an NVIDIA GPU (Ampere or
# newer) with driver 535 or newer (570 for Blackwell). The download is not code-signed (Windows SmartScreen may ask): docs/verify-releases.md says how to check it instead.
#
#   $env:GLYD_VERSION      the release to install (default below)
#   $env:GLYD_RELEASE_URL  a mirror (https://...) or a folder (C:\dir) holding glyd-engine-vX.Y.Z-windows-x86_64.zip and its .sha256, in place of the GitHub release's
#   $env:GLYD_HOME         the folder to install into (default %LOCALAPPDATA%\Glyd)
#   $env:GLYD_NO_PATH      1: leave the PATH alone
$ErrorActionPreference = 'Stop'
$GlydVersion = if ($env:GLYD_VERSION) { $env:GLYD_VERSION } else { '0.34.0' }
$Repo = 'https://github.com/surya-koritala/Glyd'

function Say($m) { Write-Host "==> $m" }

if (-not [Environment]::Is64BitOperatingSystem) { throw 'Glyd for Windows needs a 64-bit Windows.' }
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') { throw 'Glyd for Windows is built for x86_64 (an Arm PC is not supported yet).' }
$Home_ = if ($env:GLYD_HOME) { $env:GLYD_HOME } else { Join-Path $env:LOCALAPPDATA 'Glyd' }
$Name = "glyd-engine-v$GlydVersion-windows-x86_64"
$Base = if ($env:GLYD_RELEASE_URL) { $env:GLYD_RELEASE_URL.TrimEnd('\', '/') } else { "$Repo/releases/download/v$GlydVersion" }
$Tmp = Join-Path ([IO.Path]::GetTempPath()) "glyd-install-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force $Tmp | Out-Null
try {
    Say "Installing Glyd $GlydVersion for Windows x86_64 (the release's zip, checked against its sha256)"
    foreach ($f in "$Name.zip", "$Name.zip.sha256") {
        $to = Join-Path $Tmp $f
        if (Test-Path -LiteralPath $Base) { Copy-Item -LiteralPath (Join-Path $Base $f) $to }
        else {
            curl.exe --proto '=https' --tlsv1.2 -fsSL --retry 3 -o $to "$Base/$f"
            if ($LASTEXITCODE) { throw "could not download $Base/${f}: check the network, and that release $GlydVersion has a Windows zip ($Repo/releases)." }
        }
    }
    $want = ((Get-Content -LiteralPath (Join-Path $Tmp "$Name.zip.sha256") -TotalCount 1) -split '\s+')[0].ToLower()
    $got = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Tmp "$Name.zip")).Hash.ToLower()
    if (-not $want -or $got -ne $want) { throw "the download is not the file the release lists (its sha256 is $got, the release says '$want'). Run this again; if it happens again, tell the Glyd project." }
    Expand-Archive -LiteralPath (Join-Path $Tmp "$Name.zip") -DestinationPath $Tmp -Force
    $src = Join-Path $Tmp $Name
    foreach ($f in 'glyd.exe', 'libglyd_gpu_cuda13.dll', 'libglyd_gpu_cuda12.dll', 'LICENSE') {
        if (-not (Test-Path -LiteralPath (Join-Path $src $f))) { throw "the zip is not laid out as this installer expects ($Name\$f is missing): run this again, or take it from $Repo/releases." }
    }
    New-Item -ItemType Directory -Force $Home_ | Out-Null
    foreach ($f in 'glyd.exe', 'libglyd_gpu_cuda13.dll', 'libglyd_gpu_cuda12.dll') {
        # (a running glyd.exe cannot be overwritten: move it aside first)
        $dest = Join-Path $Home_ $f
        if (Test-Path -LiteralPath $dest) { Move-Item -Force -LiteralPath $dest "$dest.old" }
        Copy-Item -Force -LiteralPath (Join-Path $src $f) $dest
        Remove-Item -Force -ErrorAction SilentlyContinue "$dest.old"
    }
    Copy-Item -Force -LiteralPath (Join-Path $src 'LICENSE') (Join-Path $Home_ 'LICENSE-engine')
    # glyd-engine.exe: the engine's old command line, which glyd takes the name it is started by for
    Copy-Item -Force -LiteralPath (Join-Path $Home_ 'glyd.exe') (Join-Path $Home_ 'glyd-engine.exe')
    Say "glyd $GlydVersion is installed in $Home_"
    if ($env:GLYD_NO_PATH -ne '1') {
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (($userPath -split ';') -notcontains $Home_) {
            [Environment]::SetEnvironmentVariable('Path', (($userPath, $Home_) -join ';').Trim(';'), 'User')
            Say "$Home_ is now on your user PATH: open a new terminal to use glyd"
        }
        if (($env:Path -split ';') -notcontains $Home_) { $env:Path = "$env:Path;$Home_" }
    }
    Say 'Checking this machine (glyd doctor)'
    & (Join-Path $Home_ 'glyd.exe') doctor
    if ($LASTEXITCODE -eq 0) { Say 'Next: glyd run gemma4:12b:wren (glyd pull MODEL:LEVEL downloads without running)' }
    else { Say 'Fix the lines marked NO above, then run: glyd doctor' }
}
finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Tmp
}
