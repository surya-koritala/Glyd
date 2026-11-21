<#
The M1 gate on a clean Windows machine with only the NVIDIA driver: the PowerShell twin of m1-gate.sh (same steps, same list, same result lines; no Python anywhere: PowerShell reads the JSON).
  powershell -File benchmarks\gates\m1-gate.ps1 [-Tier 8gb|16gb|24gb] [-Models FILE] [-Glyd glyd.exe] [-InstallCmd "..."] [-Out DIR] [-Ext DIR] [-Limit 120]
  powershell -File benchmarks\gates\m1-gate.ps1 -PlanOnly -Machine rtx4080s [-Glyd glyd.exe]
-InstallCmd: the command that installs glyd; by default scripts\install.ps1 of this checkout (the release's Windows installer, which installs the Windows zip), where it exists; else install by hand and omit it.
-Ext: NAME.result files ("PASS|FAIL|SKIP<TAB>NAME<TAB>detail"): driver-r535, driver-r550, driver-r570, p2a-no-oom, r5a-schema-real-weights, r8-reference-gpu-results, p17-open-webui, p17-continue, p17-openai-sdk.
Exit status 0 only where nothing FAILs and no row is blocked on packs. Not yet run on Windows (the machine is the AWS one of the gate plan): the first run may need fixes.
#>
param([string]$Tier = "", [string]$Models = "$PSScriptRoot\m1-models.txt", [string]$Glyd = "glyd", [string]$InstallCmd = "", [string]$Out = "", [string]$Ext = "",
      [int]$Limit = 120, [switch]$PlanOnly, [string]$Machine = "")
$ErrorActionPreference = "Continue"
if (-not $Out) { $Out = Join-Path ([IO.Path]::GetTempPath()) ("m1-gate-" + [Guid]::NewGuid().ToString("N").Substring(0, 8)) }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Results = Join-Path $Out "results.tsv"; Set-Content -Path $Results -Value ""
$script:Fails = 0; $script:Blocked = 0; $script:Passes = 0
function Step($status, $name, $detail) {
    "{0,-5} {1,-34} {2}" -f $status, $name, $detail | Write-Host
    Add-Content -Path $Results -Value "$status`t$name`t$detail"
    if ($status -eq "FAIL") { $script:Fails++ } elseif ($status -eq "PASS") { $script:Passes++ }
}
function Rows { Get-Content $Models | Where-Object { $_.Trim() -ne "" -and -not $_.TrimStart().StartsWith("#") } | ForEach-Object { , ($_ -split "\|" | ForEach-Object { $_.Trim() }) } }
function Plan-Of($model) {
    $fa = @("fit", $model); if ($Machine) { $fa += @("--machine", $Machine) }
    $text = (& $Glyd @fa 2>&1 | Out-String)
    $plan = ($text -split "`r?`n" | Where-Object { $_ -like "Plan: *" } | Select-Object -First 1)
    if ($plan -and $plan -notlike "*none fits*") { return @("run", $plan, $text) }
    $why = ($text -split "`r?`n" | Where-Object { $_ -match "^glyd: |does not fit|none fits|does not run" } | Select-Object -First 1)
    return @("refuse", $why, $text)
}
$col = switch ($Tier) { "8gb" { 2 } "16gb" { 3 } "24gb" { 4 } default { -1 } }

if ($PlanOnly) {
    $col = switch ($Machine) { "rtx4060-laptop" { 2 } "rtx4080s" { 3 } "rtx4090" { 4 } default { -1 } }
    foreach ($r in Rows) { $p = Plan-Of $r[0]; $want = if ($col -ge 0) { $r[$col] } else { "-" }
        Step $(if ($col -lt 0) { "INFO" } elseif ($p[0] -eq $want) { "PASS" } else { "FAIL" }) "plan:$($r[0])" "$($p[0]) (expected $want) $($p[1])" }
    exit $(if ($script:Fails -eq 0) { 0 } else { 1 })
}

# the machine
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | Select-Object -First 1)
    $mib = [int](($gpu -split ",")[1] -replace "[^0-9]", "")
    if (-not $Tier) { $Tier = if ($mib -lt 9000) { "8gb" } elseif ($mib -lt 20000) { "16gb" } else { "24gb" }; $col = switch ($Tier) { "8gb" { 2 } "16gb" { 3 } default { 4 } } }
    $os = (Get-CimInstance Win32_OperatingSystem).Caption; $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    Step "PASS" "machine" "$gpu; tier $Tier; $ram GiB RAM; $os"
} else { Step "INFO" "machine" "no nvidia-smi: the one-line answer is what is checked" }

# python and pip are shimmed for the whole run: a call is a FAIL
$shim = Join-Path $Out "nopython"; New-Item -ItemType Directory -Force -Path $shim | Out-Null; $called = Join-Path $Out "python-called"; Set-Content $called ""
foreach ($n in "python", "python3", "pip", "pip3", "py") { Set-Content -Path (Join-Path $shim "$n.cmd") -Value "@echo %0 %* >> `"$called`"`r`n@exit /b 127" }
$env:PATH = "$shim;$env:PATH"

$defaultInstaller = Join-Path $PSScriptRoot "..\..\scripts\install.ps1"   # (the release's Windows installer, P6/P7: it installs the Windows zip)
if (-not $InstallCmd -and (Test-Path $defaultInstaller)) { $InstallCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$defaultInstaller`"" }
if ($InstallCmd) {
    $t0 = Get-Date; & cmd /c $InstallCmd *> (Join-Path $Out "install.log")
    if ($LASTEXITCODE -eq 0) { Step "PASS" "install" ("{0:N0} s: {1}" -f ((Get-Date) - $t0).TotalSeconds, $InstallCmd) } else { Step "FAIL" "install" "failed: see $Out\install.log" }
}
$v = (& $Glyd --version 2>&1 | Out-String); if ($LASTEXITCODE -ne 0) { Step "FAIL" "glyd-version" "$Glyd --version failed" }

if (-not $smi) {
    $first = (Rows | Select-Object -First 1)[0]; $o = (& $Glyd run $first --prompt hi 2>&1 | Out-String); $code = $LASTEXITCODE; $n = ($o -split "`r?`n" | Where-Object { $_ }).Count
    if ($code -ne 0 -and $n -le 2) { Step "PASS" "vendor-one-line" "exit ${code}: $(($o -split "`n")[0])" } else { Step "FAIL" "vendor-one-line" "exit $code, $n lines" }
}

$schema = '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"},"coastal":{"type":"boolean"}},"required":["city","population","coastal"],"additionalProperties":false}'
$port = 18734; $base = "http://127.0.0.1:$port"
foreach ($r in Rows) {
    $m = $r[0]; $want = if ($col -ge 0) { $r[$col] } else { "" }; $blocked = ($r.Count -ge 6 -and $r[5] -like "*BLOCKED(packs)*")
    $p = Plan-Of $m; $got = $p[0]
    if ($want -and $got -ne $want) { Step "FAIL" "plan:$m" "the planner says $got, m1-models.txt expects $want on ${Tier}: $($p[1])"; continue }
    Step "PASS" "plan:$m" "${got}: $($p[1])"
    if ($got -eq "refuse") {
        $o = (& $Glyd run $m --prompt hi 2>&1 | Out-String); $code = $LASTEXITCODE; $n = ($o -split "`r?`n" | Where-Object { $_ }).Count
        if ($code -ne 0 -and $n -le 3) { Step "PASS" "refusal:$m" "exit $code, $n lines: $(($o -split "`n")[0])" } else { Step "FAIL" "refusal:$m" "exit $code, $n lines (a refusal is a few lines and a non-zero exit)" }
        continue
    }
    $t0 = Get-Date; $log = Join-Path $Out ("serve-" + ($m -replace "[/:]", "_") + ".log")
    $srv = Start-Process -FilePath $Glyd -ArgumentList @("serve", $m, "--host", "127.0.0.1", "--port", $port, "--users", 1) -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $up = $false
    while (((Get-Date) - $t0).TotalSeconds -lt 900 -and -not $srv.HasExited) { try { Invoke-RestMethod "$base/v1/models" -TimeoutSec 2 | Out-Null; $up = $true; break } catch { Start-Sleep 1 } }
    if (-not $up) { if ($blocked) { Step "SKIP" "first-token:$m" "BLOCKED(packs): the server did not come up (the planned level's pre-packed repo is not published)"; $script:Blocked++ } else { Step "FAIL" "first-token:$m" "the server did not come up" }; if (-not $srv.HasExited) { $srv.Kill() }; continue }
    $body = '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":4,"stream":true}'
    $tok = (& curl.exe -sN -m 120 -H "Content-Type: application/json" -d $body "$base/v1/chat/completions" | Select-String -Pattern '"content":"[^"]' -List | Select-Object -First 1)
    $total = [int]((Get-Date) - $t0).TotalSeconds
    if ($tok -and $total -le $Limit) { Step "PASS" "first-token:$m" "$total s (limit $Limit)" }
    elseif ($blocked) { Step "SKIP" "first-token:$m" "BLOCKED(packs): $total s (limit $Limit): the planned level's pre-packed repo is not published"; $script:Blocked++ }
    else { Step "FAIL" "first-token:$m" "$total s (limit $Limit), token: $(if ($tok) { 'yes' } else { 'none' })" }
    $info = try { Invoke-RestMethod "$base/v1/glyd/info" -TimeoutSec 10 } catch { $null }
    $lvl = $info.level.effective; $wb = $info.memory.weights_bytes
    $planLvl = if ($p[1] -match "^Plan: ([a-z]+)") { $Matches[1] } else { "" }
    $planGib = 0.0; $gl = ($p[2] -split "`n" | Where-Object { $_ -match ": ([0-9.]+) GiB of weights on the GPU" } | Select-Object -First 1)
    if ($gl -and $gl -match ": ([0-9.]+) GiB of weights on the GPU") { $planGib = [double]$Matches[1] }
    if ($lvl -and $lvl -eq $planLvl) {
        $gib = $wb / 1GB
        if ($wb -and $planGib -gt 0 -and [math]::Abs($gib - $planGib) / $planGib -le 0.10) { Step "PASS" "plan-is-load:$m" ("level $lvl, weights {0:N2} GiB against the plan's $planGib GiB" -f $gib) } else { Step "FAIL" "plan-is-load:$m" "weights $wb bytes against the plan's $planGib GiB" }
    } else { Step "FAIL" "plan-is-load:$m" "level loaded '$lvl', planned '$planLvl'" }
    $req = @{ messages = @(@{ role = "user"; content = "Give a city, its population and whether it is coastal." }); max_tokens = 200; temperature = 0
              response_format = @{ type = "json_schema"; json_schema = @{ name = "city"; strict = $true; schema = ($schema | ConvertFrom-Json) } } } | ConvertTo-Json -Depth 10
    $txt = try { (Invoke-RestMethod "$base/v1/chat/completions" -Method Post -ContentType "application/json" -Body $req -TimeoutSec 120).choices[0].message.content } catch { "" }
    $ok = $false; try { $j = $txt | ConvertFrom-Json; $ok = ((($j.PSObject.Properties.Name | Sort-Object) -join ",") -eq "city,coastal,population") -and ($j.city -is [string]) -and (($j.population -is [int64]) -or ($j.population -is [int])) -and ($j.coastal -is [bool]) } catch {}
    if ($ok) { Step "PASS" "json-schema:$m" ($txt -replace "\s+", " ") } else { Step "FAIL" "json-schema:$m" "not valid against the schema: $txt" }
    if (-not $srv.HasExited) { $srv.Kill() }; Start-Sleep 2
}

if ((Get-Content $called | Where-Object { $_ }).Count -gt 0) { Step "FAIL" "no-python" ("called: " + ((Get-Content $called | Select-Object -First 3) -join "; ")) } else { Step "PASS" "no-python" "no call to python or pip during the run" }

foreach ($name in "driver-r535", "driver-r550", "driver-r570", "p2a-no-oom", "r5a-schema-real-weights", "r8-reference-gpu-results", "p17-open-webui", "p17-continue", "p17-openai-sdk") {
    $f = if ($Ext) { Join-Path $Ext "$name.result" } else { "" }
    if ($f -and (Test-Path $f)) { $c = (Get-Content $f | Select-Object -First 1) -split "`t", 3; Step $c[0] "ext:$name" $c[2] } else { Step "FAIL" "ext:$name" "no result (-Ext DIR with $name.result)" }
}
Write-Host "---"; Write-Host "$($script:Passes) passed, $($script:Fails) failed, $($script:Blocked) rows blocked on packs (a blocked row keeps the gate red); results in $Results"
exit $(if ($script:Fails -eq 0 -and $script:Blocked -eq 0) { 0 } else { 1 })
