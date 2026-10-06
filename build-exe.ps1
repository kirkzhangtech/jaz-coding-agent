<#
.SYNOPSIS
    Build dist\jaz-agent.exe, deploy it where PATH will find it, and check it.

.DESCRIPTION
    Four steps, in the order that matters:

      1. the offline test suite (241 tests: no network, no API key)
      2. PyInstaller, driven by jaz_agent.spec
      3. copy dist\jaz-agent.exe over the copy on PATH
      4. run --selftest on the deployed copy

    Step 3 is the one that gets forgotten. `jaz-agent` on PATH resolves to
    C:\Users\<you>\bin\jaz-agent.exe, so a build that stops at dist\ leaves the old
    binary in place -- and a stale binary is indistinguishable from a fix that did
    not work. Onefile compresses its payload, so grepping the exe for a new string
    or a new function name proves nothing; the script reports timestamps instead.

.PARAMETER SkipTests
    Build without running the suite first. Faster, and only sensible when the
    suite was just run.

.PARAMETER SkipDeploy
    Stop after the build: leave dist\jaz-agent.exe in place and copy nothing.

.PARAMETER DeployTo
    Directory to deploy into. Defaults to the directory of the jaz-agent.exe that
    PATH resolves to, or %USERPROFILE%\bin when nothing is on PATH.

.EXAMPLE
    .\build-exe.ps1

.EXAMPLE
    .\build-exe.ps1 -SkipTests

.EXAMPLE
    .\build-exe.ps1 -DeployTo D:\tools
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipDeploy,
    [string]$DeployTo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root   = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$spec   = Join-Path $root 'jaz_agent.spec'
$built  = Join-Path $root 'dist\jaz-agent.exe'

function Write-Step { param([string]$Text) Write-Host "`n== $Text" -ForegroundColor Cyan }
function Write-Note { param([string]$Text) Write-Host "   $Text" -ForegroundColor DarkGray }

function Invoke-Exe {
    <#
        Run a native command and hand back its exit code plus both streams.

        Start-Process with redirects, rather than `& exe 2>&1`: under
        $ErrorActionPreference = 'Stop' anything a child writes to stderr becomes a
        terminating NativeCommandError in Windows PowerShell 5.1, and jaz writes
        warnings there. The exe would look broken when it had not even failed.
    #>
    param([string]$Path, [string[]]$Arguments = @())

    $out = [IO.Path]::GetTempFileName()
    $err = [IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath $Path -ArgumentList $Arguments -NoNewWindow -Wait -PassThru `
                           -RedirectStandardOutput $out -RedirectStandardError $err
        return [pscustomobject]@{
            ExitCode = $p.ExitCode
            Output   = ((Get-Content $out -Raw) + (Get-Content $err -Raw))
        }
    } finally {
        Remove-Item $out, $err -ErrorAction SilentlyContinue
    }
}

try {
    if (-not (Test-Path $python)) {
        throw "no interpreter at $python -- create the venv first (README, Setup)"
    }
    if (-not (Test-Path $spec)) {
        throw "no PyInstaller spec at $spec"
    }

    # Resolved before the build, so a typo or a locked copy is known about before
    # three minutes of PyInstaller rather than after it.
    $target = $null
    if (-not $SkipDeploy) {
        if ($DeployTo) {
            $target = Join-Path $DeployTo 'jaz-agent.exe'
        } else {
            # Deploy where the shell will actually find it. Nothing on PATH means
            # a fresh machine, so fall back to the conventional per-user bin.
            $dir = Join-Path $env:USERPROFILE 'bin'
            $found = Get-Command 'jaz-agent.exe' -ErrorAction SilentlyContinue
            if ($found) { $dir = Split-Path -Parent $found.Source }
            $target = Join-Path $dir 'jaz-agent.exe'
        }
        Write-Note "will deploy to $target"
    }

    $running = @(Get-Process -Name 'jaz-agent' -ErrorAction SilentlyContinue)
    if ($running.Count -gt 0) {
        Write-Host "   warning: $($running.Count) jaz-agent process(es) are running." -ForegroundColor Yellow
        Write-Host "   They keep the old build until restarted, and a locked exe cannot be replaced." -ForegroundColor Yellow
    }

    if (-not $SkipTests) {
        Write-Step 'Tests (offline)'
        Write-Note '241 tests, about two minutes; the summary appears when they finish.'
        # A stub key, so the suite cannot reach the live catalogue: several tests
        # only stay fast and deterministic when the fetch fails immediately.
        $env:OPENROUTER_API_KEY = 'sk-or-test'
        $r = Invoke-Exe $python @('-m', 'pytest', '-q', '-p', 'no:warnings')
        Write-Host $r.Output.TrimEnd()
        if ($r.ExitCode -ne 0) {
            throw "tests failed (exit $($r.ExitCode)) -- refusing to build a revision that does not pass"
        }
    }

    Write-Step 'PyInstaller'
    $r = Invoke-Exe $python @('-m', 'PyInstaller', $spec, '--noconfirm')
    (($r.Output -split "`n" | Select-Object -Last 2) -join "`n") | Write-Host
    if ($r.ExitCode -ne 0) { throw "PyInstaller failed (exit $($r.ExitCode))" }
    if (-not (Test-Path $built)) { throw "PyInstaller reported success but $built is missing" }

    if (-not $SkipDeploy) {
        Write-Step 'Deploy'
        $dir = Split-Path -Parent $target
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        try {
            Copy-Item $built $target -Force
        } catch {
            throw "could not replace $target -- is that copy running? Close it and re-run. ($($_.Exception.Message))"
        }
        Write-Note "copied $((Get-Item $built).Length) bytes"

        Write-Step 'Smoke test on the deployed copy'
        $r = Invoke-Exe $target @('--selftest')
        (($r.Output -split "`n" | Where-Object { $_ -match 'SELFTEST|FAIL' }) -join "`n") | Write-Host
        if ($r.ExitCode -ne 0) { throw "the deployed copy failed its selftest (exit $($r.ExitCode))" }
    }

    Write-Step 'Result'
    $paths = if ($target) { @($built, $target) } else { @($built) }
    Get-Item $paths | Select-Object FullName, LastWriteTime, Length |
        Format-Table -AutoSize | Out-String | Write-Host
    Write-Host 'Start it with:  jaz-agent -m openai/gpt-5-mini' -ForegroundColor Green
    Write-Host 'An already-running window keeps the old build until it is restarted.' -ForegroundColor DarkGray
    exit 0
} catch {
    Write-Host "`nbuild failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
