# monitor_v16.ps1 - progress window for run_v16.bat (opened automatically).
# Refreshes every 30 s. While this window is open the PC is kept awake
# (a per-process request; no power settings are changed). Close it to stop monitoring.
param([string]$Res = 'results_v16')
Set-Location $PSScriptRoot
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$start = Get-Date
while ($true) {
    [void][W.P]::SetThreadExecutionState([uint32]"0x80000001")   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    Clear-Host
    $el = (Get-Date) - $start
    Write-Host ("v16 monitor   {0}   watching for {1:hh\:mm}   results: {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm'), $el, $Res) -ForegroundColor Cyan
    $store = Join-Path $Res 'runs.jsonl'
    if (Test-Path $store) {
        $lines = Get-Content $store
        $n = $lines.Count
        Write-Host ("Finished training runs on disk: {0}" -f $n) -ForegroundColor Green
        $parts = @{}
        foreach ($l in $lines) { if ($l -match '"part":\s*"([^"]+)"') { $parts[$matches[1]]++ } }
        if ($parts.Count) { Write-Host ('By part: ' + (($parts.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join '  ')) }
        $age = (Get-Date) - (Get-Item $store).LastWriteTime
        $col = 'Gray'; if ($age.TotalMinutes -gt 60) { $col = 'Yellow' }
        Write-Host ("Last run saved {0:N0} min ago" -f $age.TotalMinutes) -ForegroundColor $col
    } else { Write-Host 'No runs saved yet.' }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $g = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
        Write-Host "GPU: $g"
    }
    $log = Get-ChildItem logs\run_*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($log) {
        Write-Host ("`n--- {0} (last 20 lines) ---" -f $log.Name) -ForegroundColor Cyan
        Get-Content $log.FullName -Tail 20
    }
    $fix = Get-ChildItem logs\claude_fix_*.log -ErrorAction SilentlyContinue
    if ($fix) { Write-Host ("`nAuto-fix attempts so far: {0} (see V16_FIX_NOTES.md)" -f $fix.Count) -ForegroundColor Yellow }
    if (Test-Path STOP_AUTOFIX.txt) { Write-Host "`nSTOP_AUTOFIX.txt present - auto-fix gave up:" -ForegroundColor Red; Get-Content STOP_AUTOFIX.txt }
    Start-Sleep -Seconds 30
}
