<#
.SYNOPSIS
  Test the agents. Run from anywhere; uses the project's .venv.

.DESCRIPTION
  offline  Unit tests with fake KBs and a fake model. No AWS login needed. Fast (about 5 seconds). Default.
  live     Ask the REAL agents a few questions and show how each was routed. Needs: saml2aws login --profile sand
  eval     Score the real agents on eval/cases.yaml (23 labelled cases) and save eval/results/<time>-<label>.json.
  all      offline, then live, then eval. Stops early if offline fails.

.EXAMPLE
  .\tests\scripts\run_tests.ps1                                  # offline unit tests
  .\tests\scripts\run_tests.ps1 live                             # one question per route against the real agents
  .\tests\scripts\run_tests.ps1 live -Question "How much is Fazkurooni?", "Who won the AFL?"
  .\tests\scripts\run_tests.ps1 live -Model nova                 # same, with another configured model
  .\tests\scripts\run_tests.ps1 eval -Label router-v1 -Repeat 3  # scored run; compare later with --eval-compare
  .\tests\scripts\run_tests.ps1 eval -Filter both                # only the "both" cases
#>
[CmdletBinding(PositionalBinding = $false)]   # a stray argument is an error, never silently bound to another parameter
param(
    [Parameter(Position = 0)][ValidateSet('offline', 'live', 'eval', 'all')][string]$Mode = 'offline',
    # live: your own questions instead of the defaults. Works as  -Question "a","b"  and as plain trailing arguments
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Question,
    [string]$Model,                   # live: model key from config.yaml
    [string]$Profile,                 # live/eval: AWS profile (default: config.yaml aws.profile)
    [string]$Label = 'run',           # eval: name for the result file
    [int]$Repeat = 3,                 # eval: runs per case (3+ shows flaky routing)
    [string]$Filter                   # eval: only cases whose id or category contains this text
)

$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent   # project root (this script lives in tests/scripts)
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "No virtual environment at $py. Create it first: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}
$env:PYTHONUTF8 = '1'
Push-Location $root

function Invoke-Step([string]$Name, [scriptblock]$Run) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Run | Out-Host            # show the output; otherwise it would become the function's return value
    $code = $LASTEXITCODE
    $script:results += [pscustomobject]@{ Step = $Name; Result = $(if ($code -eq 0) { 'PASS' } else { "FAIL ($code)" }) }
    return $code
}

$results = @()
try {
    if ($Mode -in 'offline', 'all') {
        $code = Invoke-Step 'offline unit tests' { & $py -m unittest discover -s tests -t . -v }
        if ($code -ne 0 -and $Mode -eq 'all') { Write-Host 'Offline tests failed; skipping live steps.' -ForegroundColor Red }
    }
    if (($Mode -eq 'live') -or ($Mode -eq 'all' -and $code -eq 0)) {
        $liveArgs = @('-m', 'tests.live_smoke')
        if ($Model)   { $liveArgs += @('--model', $Model) }
        if ($Profile) { $liveArgs += @('--profile', $Profile) }
        if ($Question) { $liveArgs += $Question }
        Invoke-Step 'live smoke (real agents)' { & $py @liveArgs } | Out-Null
    }
    if (($Mode -eq 'eval') -or ($Mode -eq 'all' -and $code -eq 0)) {
        $evalArgs = @('main.py', '--eval', '--eval-label', $Label, '--eval-repeat', $Repeat)
        if ($Profile) { $evalArgs += @('--profile', $Profile) }
        if ($Filter)  { $evalArgs += @('--eval-filter', $Filter) }
        Invoke-Step "evaluation ($Repeat run(s) per case)" { & $py @evalArgs } | Out-Null
    }
}
finally {
    Pop-Location
}

Write-Host "`n=== summary ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize | Out-String | Write-Host
if ($results.Result -match 'FAIL') {
    Write-Host 'If a live step failed with ExpiredToken: saml2aws login --profile sand' -ForegroundColor Yellow
    exit 1
}
