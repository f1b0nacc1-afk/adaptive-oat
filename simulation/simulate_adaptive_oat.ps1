<#
.SYNOPSIS
    Simulation: Adaptive Action Token Refinement for OAT
    Pure PowerShell — no dependencies.
.PARAMETER Seed
    Random seed (default 42)
.PARAMETER NEpisodes
    Episodes per task type (default 100, total = 3x)
.PARAMETER NSteps
    Steps per episode (default 20)
.PARAMETER Eps
    Stopping threshold: stop when err_k < eps * action_norm (default 0.05)
.PARAMETER OutDir
    Output directory for results
#>
param(
    [int]$Seed       = 42,
    [int]$NEpisodes  = 100,
    [int]$NSteps     = 20,
    [double]$Eps     = 0.05,
    [string]$OutDir  = "$PSScriptRoot\..\results"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── RNG with seed ─────────────────────────────────────────────────────────────
$rng = [System.Random]::new($Seed)

function Get-UniformRandom([System.Random]$r, [double]$a, [double]$b) {
    return $a + ($b - $a) * $r.NextDouble()
}

function Get-GaussRandom([System.Random]$r, [double]$mu, [double]$sigma) {
    # Box-Muller
    $u1 = [Math]::Max(1e-15, $r.NextDouble())
    $u2 = $r.NextDouble()
    $z  = [Math]::Sqrt(-2.0 * [Math]::Log($u1)) * [Math]::Cos(2.0 * [Math]::PI * $u2)
    return $mu + $sigma * $z
}

function Get-L2Norm([double[]]$v) {
    $sum = 0.0
    foreach ($x in $v) { $sum += $x * $x }
    return [Math]::Sqrt($sum)
}

# ── Trajectory generation ─────────────────────────────────────────────────────
function New-Trajectory([System.Random]$r, [string]$TaskType, [int]$NSteps) {
    $steps = @()
    for ($i = 0; $i -lt $NSteps; $i++) {
        $complexity = switch ($TaskType) {
            "simple"  { Get-UniformRandom $r 0.05 0.25 }
            "complex" { Get-UniformRandom $r 0.60 0.95 }
            default   {
                $phase = [Math]::Sin([Math]::PI * $i / $NSteps)
                0.10 + 0.80 * [Math]::Abs($phase)
            }
        }
        $action = @()
        for ($d = 0; $d -lt 7; $d++) {
            $action += Get-GaussRandom $r 0.0 (0.08 + $complexity * 0.85)
        }
        $steps += [pscustomobject]@{ StepId = $i; Complexity = $complexity; Action = $action }
    }
    return $steps
}

# ── VQ encoder ────────────────────────────────────────────────────────────────
function Get-VQErrors([double[]]$action, [int]$NTokens, [System.Random]$r) {
    $norm = Get-L2Norm $action
    $inferredComplexity = [Math]::Min(1.0, $norm / ([Math]::Sqrt(7) * 0.70))

    $initialFraction = 0.35 + 0.30 * $inferredComplexity + (Get-UniformRandom $r -0.05 0.05)
    $initialFraction = [Math]::Max(0.05, $initialFraction)
    $remaining = $norm * $initialFraction

    $baseDecay = 0.30 + 0.40 * $inferredComplexity
    $errors = @()
    for ($k = 0; $k -lt $NTokens; $k++) {
        $decay = [Math]::Max(0.15, $baseDecay + (Get-GaussRandom $r 0.0 0.04))
        $remaining *= $decay
        $errors += $remaining
    }
    return $errors
}

# ── Fixed OAT ─────────────────────────────────────────────────────────────────
function Get-FixedOATError([array]$traj, [int]$NTokens, [System.Random]$r) {
    $errs = @()
    foreach ($step in $traj) {
        $vqErrors = Get-VQErrors $step.Action $NTokens $r
        $errs += $vqErrors[-1]
    }
    return ($errs | Measure-Object -Average).Average
}

# ── Adaptive OAT ──────────────────────────────────────────────────────────────
function Get-AdaptiveOATResult([array]$traj, [int]$MaxTokens, [double]$eps, [System.Random]$r) {
    $reconErrors = @()
    $tokensUsed  = @()

    foreach ($step in $traj) {
        $norm      = Get-L2Norm $step.Action
        $threshold = $eps * $norm
        $vqErrors  = Get-VQErrors $step.Action $MaxTokens $r

        $chosenK = $MaxTokens
        for ($k = 1; $k -le $MaxTokens; $k++) {
            if ($vqErrors[$k - 1] -lt $threshold) {
                $chosenK = $k
                break
            }
        }
        $reconErrors += $vqErrors[$chosenK - 1]
        $tokensUsed  += $chosenK
    }

    $meanErr = ($reconErrors | Measure-Object -Average).Average
    $meanTok = ($tokensUsed  | Measure-Object -Average).Average
    $meanCmx = ($traj | ForEach-Object { $_.Complexity } | Measure-Object -Average).Average
    return [pscustomobject]@{ Error = $meanErr; Tokens = $meanTok; Complexity = $meanCmx }
}

# ── Experiment ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Simulation: Adaptive Action Token Refinement for OAT" -ForegroundColor Cyan
Write-Host "  seed=$Seed  episodes_per_type=$NEpisodes  steps=$NSteps  eps=$Eps"
Write-Host ""

$taskTypes = @("simple", "complex", "mixed")
$allResults = @()

$episodeId = 0
foreach ($task in $taskTypes) {
    Write-Host "  Running task type: $task ..." -NoNewline
    for ($ep = 0; $ep -lt $NEpisodes; $ep++) {
        $epSeed = $rng.Next(0, [int]::MaxValue)
        $epRng  = [System.Random]::new($epSeed)

        $traj = New-Trajectory $epRng $task $NSteps

        $r1 = [System.Random]::new($epRng.Next(0,[int]::MaxValue))
        $r2 = [System.Random]::new($epRng.Next(0,[int]::MaxValue))
        $r4 = [System.Random]::new($epRng.Next(0,[int]::MaxValue))
        $r8 = [System.Random]::new($epRng.Next(0,[int]::MaxValue))
        $ra = [System.Random]::new($epRng.Next(0,[int]::MaxValue))

        $oat1err = Get-FixedOATError $traj 1 $r1
        $oat2err = Get-FixedOATError $traj 2 $r2
        $oat4err = Get-FixedOATError $traj 4 $r4
        $oat8err = Get-FixedOATError $traj 8 $r8
        $adRes   = Get-AdaptiveOATResult $traj 8 $Eps $ra

        $allResults += [pscustomobject]@{
            Episode         = $episodeId
            TaskType        = $task
            MeanComplexity  = [Math]::Round($adRes.Complexity, 4)
            OAT1Error       = [Math]::Round($oat1err, 6)
            OAT2Error       = [Math]::Round($oat2err, 6)
            OAT4Error       = [Math]::Round($oat4err, 6)
            OAT8Error       = [Math]::Round($oat8err, 6)
            AdaptiveError   = [Math]::Round($adRes.Error, 6)
            AdaptiveTokens  = [Math]::Round($adRes.Tokens, 3)
            Eps             = $Eps
        }
        $episodeId++
    }
    Write-Host " done ($NEpisodes episodes)" -ForegroundColor Green
}

# ── Summary stats ─────────────────────────────────────────────────────────────
function Get-Summary([array]$rows) {
    $n       = $rows.Count
    $cplx    = ($rows | Measure-Object MeanComplexity -Average).Average
    $e1      = ($rows | Measure-Object OAT1Error      -Average).Average
    $e2      = ($rows | Measure-Object OAT2Error      -Average).Average
    $e4      = ($rows | Measure-Object OAT4Error      -Average).Average
    $e8      = ($rows | Measure-Object OAT8Error      -Average).Average
    $ea      = ($rows | Measure-Object AdaptiveError  -Average).Average
    $tok     = ($rows | Measure-Object AdaptiveTokens -Average).Average

    # stdev for adaptive error
    $errList = $rows | ForEach-Object { $_.AdaptiveError }
    $mean_ea = $ea
    $variance = ($errList | ForEach-Object { ($_ - $mean_ea) * ($_ - $mean_ea) } | Measure-Object -Sum).Sum / [Math]::Max(1, ($n - 1))
    $std_ea  = [Math]::Sqrt($variance)

    $tokList = $rows | ForEach-Object { $_.AdaptiveTokens }
    $mean_tok = $tok
    $var_tok = ($tokList | ForEach-Object { ($_ - $mean_tok) * ($_ - $mean_tok) } | Measure-Object -Sum).Sum / [Math]::Max(1, ($n - 1))
    $std_tok = [Math]::Sqrt($var_tok)

    $saving_pct = (8.0 - $tok) / 8.0 * 100.0
    $delta_pct  = if ($e8 -gt 1e-12) { ($ea - $e8) / $e8 * 100.0 } else { 0 }

    return [pscustomobject]@{
        n_episodes                  = $n
        mean_complexity             = [Math]::Round($cplx, 3)
        oat1_error                  = [Math]::Round($e1, 6)
        oat2_error                  = [Math]::Round($e2, 6)
        oat4_error                  = [Math]::Round($e4, 6)
        oat8_error                  = [Math]::Round($e8, 6)
        adaptive_error              = [Math]::Round($ea, 6)
        adaptive_error_std          = [Math]::Round($std_ea, 6)
        adaptive_tokens_mean        = [Math]::Round($tok, 2)
        adaptive_tokens_std         = [Math]::Round($std_tok, 2)
        token_saving_pct            = [Math]::Round($saving_pct, 1)
        quality_delta_vs_oat8_pct   = [Math]::Round($delta_pct, 1)
    }
}

$summaryByTask = @{}
foreach ($task in $taskTypes) {
    $subset = $allResults | Where-Object { $_.TaskType -eq $task }
    $summaryByTask[$task] = Get-Summary $subset
}
$summaryByTask["all"] = Get-Summary $allResults

# ── Print table ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("  " + "="*76) -ForegroundColor Yellow
Write-Host "  Results: eps=$Eps   (seed=$Seed, $($allResults.Count) total episodes)" -ForegroundColor Yellow
Write-Host ("  " + "="*76) -ForegroundColor Yellow
Write-Host ("  {0,-9} {1,6} {2,9} {3,9} {4,9} {5,9} {6,9} {7,5} {8,6} {9,7}" -f `
    "Task","Cplx","OAT1","OAT4","OAT8","Adapt","Tok","Save%","Δerr%","")
Write-Host ("  " + "-"*76)

foreach ($task in @("simple","complex","mixed","all")) {
    $s = $summaryByTask[$task]
    $sign = if ($s.quality_delta_vs_oat8_pct -ge 0) {"+"} else {""}
    Write-Host ("  {0,-9} {1,6:F3} {2,9:F5} {3,9:F5} {4,9:F5} {5,9:F5} {6,9:F5} {7,5:F1} {8,5:F1}% {9}" -f `
        $task,
        $s.mean_complexity,
        $s.oat1_error,
        $s.oat4_error,
        $s.oat8_error,
        $s.adaptive_error,
        $s.adaptive_tokens_mean,
        $s.token_saving_pct,
        $s.quality_delta_vs_oat8_pct,
        "${sign}$($s.quality_delta_vs_oat8_pct)%"
    )
}
Write-Host ("  " + "="*76) -ForegroundColor Yellow
Write-Host "  Tok=tokens used by adaptive | Save%=savings vs OAT8 | Δerr%=quality vs OAT8"

# ASCII bar chart
Write-Host ""
Write-Host "  Token savings by task type:" -ForegroundColor Cyan
foreach ($task in @("simple","complex","mixed")) {
    $s = $summaryByTask[$task]
    $barLen = [int]($s.token_saving_pct / 3.5)
    $bar = "█" * $barLen
    Write-Host ("  {0,-8} |{1,-20}| {2,5:F1}%  ({3:F1}/8 tokens)" -f `
        $task, $bar, $s.token_saving_pct, $s.adaptive_tokens_mean)
}
Write-Host ""

# ── Save results ──────────────────────────────────────────────────────────────
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

# CSV
$csvPath = Join-Path $OutDir "episodes.csv"
$allResults | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host "  Saved: $csvPath  ($($allResults.Count) rows)" -ForegroundColor Green

# Summary JSON
$jsonObj = [ordered]@{}
foreach ($task in @("simple","complex","mixed","all")) {
    $s = $summaryByTask[$task]
    $jsonObj[$task] = [ordered]@{
        n_episodes                = $s.n_episodes
        mean_complexity           = $s.mean_complexity
        oat1_error                = $s.oat1_error
        oat2_error                = $s.oat2_error
        oat4_error                = $s.oat4_error
        oat8_error                = $s.oat8_error
        adaptive_error            = $s.adaptive_error
        adaptive_error_std        = $s.adaptive_error_std
        adaptive_tokens_mean      = $s.adaptive_tokens_mean
        adaptive_tokens_std       = $s.adaptive_tokens_std
        token_saving_pct          = $s.token_saving_pct
        quality_delta_vs_oat8_pct = $s.quality_delta_vs_oat8_pct
    }
}
$jsonObj["meta"] = [ordered]@{
    seed               = $Seed
    n_episodes_per_task = $NEpisodes
    n_steps            = $NSteps
    eps                = $Eps
    max_tokens         = 8
    min_tokens         = 1
    run_date           = (Get-Date -Format "yyyy-MM-dd")
}
$jsonPath = Join-Path $OutDir "summary.json"
$jsonObj | ConvertTo-Json -Depth 4 | Set-Content -Path $jsonPath -Encoding UTF8
Write-Host "  Saved: $jsonPath" -ForegroundColor Green
Write-Host ""
Write-Host "  Done." -ForegroundColor Cyan
Write-Host ""
