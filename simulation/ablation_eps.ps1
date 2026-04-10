<#
.SYNOPSIS
    Ablation study: effect of stopping threshold (eps) on token efficiency vs quality.
#>
param(
    [int]$Seed      = 42,
    [int]$NEpisodes = 100,
    [int]$NSteps    = 20,
    [string]$OutDir = "$PSScriptRoot\..\results"
)

$epsValues = @(0.02, 0.05, 0.10, 0.15, 0.20)
$ablation  = @{}

foreach ($eps in $epsValues) {
    Write-Host "  Running eps=$eps ..." -NoNewline
    $result = & "$PSScriptRoot\simulate_adaptive_oat.ps1" `
        -Seed $Seed -NEpisodes $NEpisodes -NSteps $NSteps -Eps $eps -OutDir "$OutDir\tmp_eps_$($eps -replace '\.','_')" `
        6>$null 2>$null

    # re-run silently and capture summary JSON
    & "$PSScriptRoot\simulate_adaptive_oat.ps1" `
        -Seed $Seed -NEpisodes $NEpisodes -NSteps $NSteps -Eps $eps `
        -OutDir "$OutDir\tmp_eps_$($eps -replace '\.','_')" | Out-Null

    $sumPath = "$OutDir\tmp_eps_$($eps -replace '\.','_')\summary.json"
    if (Test-Path $sumPath) {
        $ablation["$eps"] = (Get-Content $sumPath -Raw | ConvertFrom-Json)
        Write-Host " done" -ForegroundColor Green
    } else {
        Write-Host " ERROR: summary not found" -ForegroundColor Red
    }
}

# Print table
Write-Host ""
Write-Host ("  " + "="*65) -ForegroundColor Yellow
Write-Host "  Ablation: Stopping Threshold (eps) — All Task Types" -ForegroundColor Yellow
Write-Host ("  " + "="*65) -ForegroundColor Yellow
Write-Host ("  {0,6}  {1,7}  {2,7}  {3,8}  {4,10}  {5,10}" -f "eps","Tokens","Save%","Δerr%","AdaptErr","OAT8Err")
Write-Host ("  " + "-"*60)

foreach ($eps in $epsValues) {
    $s = $ablation["$eps"].all
    $sign = if ($s.quality_delta_vs_oat8_pct -ge 0) {"+"} else {""}
    Write-Host ("  {0,6:F2}  {1,7:F2}  {2,6:F1}%  {3}{4,6:F1}%  {5,10:F6}  {6,10:F6}" -f `
        $eps,
        $s.adaptive_tokens_mean,
        $s.token_saving_pct,
        $sign, $s.quality_delta_vs_oat8_pct,
        $s.adaptive_error,
        $s.oat8_error
    )
}
Write-Host ("  " + "="*65) -ForegroundColor Yellow

# Save consolidated ablation JSON
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$outPath = Join-Path $OutDir "ablation_eps.json"
$ablation | ConvertTo-Json -Depth 6 | Set-Content -Path $outPath -Encoding UTF8
Write-Host ""
Write-Host "  Saved: $outPath" -ForegroundColor Green

# Cleanup tmp dirs
foreach ($eps in $epsValues) {
    $tmpDir = "$OutDir\tmp_eps_$($eps -replace '\.','_')"
    if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
}
Write-Host "  Done." -ForegroundColor Cyan
