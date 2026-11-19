# Qwen3.8-Flash-Next on Glyd in one command, on Windows: checks this computer, installs the engine, downloads the model, asks two questions, then
# starts the chat page and the API. Run it again to start it next time (nothing is downloaded twice).
#   powershell -ExecutionPolicy Bypass -File setup.ps1        (or add -Yes for the defaults)
param([switch]$Yes)
$ErrorActionPreference = 'Stop'
$H = if ($env:GLYD_QWEN) { $env:GLYD_QWEN } else { Join-Path $HOME '.glyd-qwen-3.8-flash' }
function Say($t) { Write-Host $t -ForegroundColor White }
function Ask($q, $d) { if ($Yes) { return $d }; $a = Read-Host "$q [$d]"; if ($a) { $a } else { $d } }

Say 'Condor: Qwen3.8-Flash-Next (Hummingbird level) on your GPU'
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) { throw 'No NVIDIA driver found (nvidia-smi). Install the NVIDIA driver 580 or newer, then run this again.' }
$gpu, $vram, $drv = (nvidia-smi.exe --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | Select-Object -First 1) -split ',\s*'
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$disk = [math]::Floor((Get-PSDrive ($H.Substring(0, 1))).Free / 1GB)
"  GPU $gpu, $([math]::Round([int]$vram / 1024)) GB, driver $drv;  RAM $ram GiB;  free disk $disk GB"
if ([int]$vram -lt 15000) { '  Note: less than 16 GB on the GPU: fewer experts fit there (on a 16 GB card held to 12 GB, about 10% slower). Under 12 GB is untested.' }
if ($ram -lt 60) { '  Note: less than 64 GB of RAM: fast mode is the one to pick (precise''s experts do not fit in RAM and run about 40% slower); the first answer after a start is slower while experts load from the disk.' }
if ([int]($drv -split '\.')[0] -lt 580) { throw "The NVIDIA driver is $drv; 580 or newer is needed (nvidia.com/drivers), then run this again." }

if (-not (Test-Path (Join-Path $H 'bin\glyd-engine.exe'))) { Say 'Installing the engine (about 50 MB)'; & (Join-Path $PSScriptRoot 'tools\install.ps1') }
. (Join-Path $H 'env.ps1')
$def = if ($ram -lt 60) { 'fast' } else { 'precise' }
$mode = Ask 'Mode: precise (accurate, good for code; wants 64 GB of RAM), best (more accurate still, a little slower, 8 GB more to download; 64 GB of RAM) or fast (quicker answers, a smaller download)' $def
switch ($mode) { 'precise' { $part = 'precise-v2'; $gb = 82 } 'best' { $part = 'best-v2'; $gb = 90 } 'fast' { $part = 'fast'; $gb = 66 } default { throw 'Answer precise, best or fast.' } }
if (-not (Test-Path (Join-Path $H 'hf\model\config.json')) -or -not (Test-Path (Join-Path $H "hf\$part"))) {
    if ($disk -lt $gb + 7) { throw "About $($gb + 7) GB of free disk is needed for $mode mode; $disk GB is free." }
    Say "Downloading the model (about $gb GB; it resumes if it stops)"
    & (Join-Path $PSScriptRoot 'tools\download.ps1') -Fast:($mode -eq 'fast') -Best:($mode -eq 'best')
    if ($LASTEXITCODE) { throw 'the download stopped: run this again to resume it' }
}
$env:GLYD_CTX = Ask 'Context (tokens the model keeps in mind): 32768 is fastest; 65536, 131072 or 262144 hold more, with less room for the model on the GPU' '32768'
Say 'Starting: chat at http://127.0.0.1:8080/  |  API at http://127.0.0.1:8080/v1 (OpenAI and Anthropic)  |  Ctrl+C to stop'
Start-Job { Start-Sleep 3; Start-Process 'http://127.0.0.1:8080/' } | Out-Null
& (Join-Path $PSScriptRoot 'tools\serve.ps1') -Fast:($mode -eq 'fast') -Best:($mode -eq 'best') 8080
