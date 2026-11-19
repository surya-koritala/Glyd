# How fast it is on this computer (bench.sh's twin): a short chat, then prompts of about 4K, 16K and 32K tokens, against the running server
# (tools\serve.ps1). About 5 minutes; writes bench-report.md, then prints one block to paste into an issue.   tools\bench.ps1 [URL]
param([string]$Url = 'http://127.0.0.1:8080')
$ErrorActionPreference = 'Stop'
. (Join-Path $(if ($env:GLYD_QWEN) { $env:GLYD_QWEN } else { Join-Path $HOME '.glyd-qwen-3.8-flash' }) 'env.ps1')
& $env:GLYD_ENGINE qwenbench --url $Url --out bench-report.md
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$q = { param($f) (nvidia-smi.exe "--query-gpu=$f" --format=csv,noheader | Select-Object -First 1) }
$mode = try { (Invoke-RestMethod "$Url/v1/glyd/status").mode } catch { '?' }
''
'---- paste from here ----'
'```'
"GPU:     $(& $q 'name,memory.total')"
"Driver:  $(& $q 'driver_version')"
"PCIe:    now gen $(& $q 'pcie.link.gen.current') x$(& $q 'pcie.link.width.current'), max gen $(& $q 'pcie.link.gen.max') x$(& $q 'pcie.link.width.max')"
"CPU:     $((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)"
"RAM:     $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)) GiB"
"OS:      $((Get-CimInstance Win32_OperatingSystem).Caption)"
"Mode:    $mode"
"Engine:  $(& $env:GLYD_ENGINE --version), sha256 $((Get-FileHash -Algorithm SHA256 $env:GLYD_ENGINE).Hash.Substring(0, 12).ToLower())"
Get-Content bench-report.md | Where-Object { $_ -like '|*' }
'```'
'---- to here ----'
