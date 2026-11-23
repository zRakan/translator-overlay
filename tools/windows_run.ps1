# Start/stop the whole capture-card setup on Windows, in the background: translation server + the app, which opens a
# FULLSCREEN window with the card's video and the Arabic subtitles. Both keep running after this script ends.
#   powershell -ExecutionPolicy Bypass -File tools\windows_run.ps1 start [-Config config.local.toml | config.capture.toml] [-Windowed] [-DebugLog]
#   -Record: only show and record the card (no OCR, no translation, no server) to [record] dir of the config (default recordings\rec-<time>.mkv, + .mka sound, k = mark the time); run the pipeline on the file afterwards
#   powershell -ExecutionPolicy Bypass -File tools\windows_run.ps1 stop
# In the window: q / Esc = quit, f = fullscreen <-> windowed, r = show/hide the subtitle-region rectangle, p = performance panel, k = mark a problem (with -DebugLog). Log: cam.out.log; -DebugLog writes debug\trace-<time>.jsonl (python tools\diagnose.py)
# (If you call this from a program that captures its output through a pipe, redirect to a file: the programs started here
# inherit that handle, so the caller would otherwise wait for them to finish.)
param([Parameter(Mandatory = $true)][ValidateSet('start', 'stop')][string]$Action, [string]$Config = '', [switch]$Windowed, [switch]$DebugLog, [switch]$Record)
$root = Split-Path -Parent $PSScriptRoot
if (-not $Config) { $Config = if (Test-Path (Join-Path $root 'config.local.toml')) { 'config.local.toml' } else { 'config.capture.toml' } }  # config.local.toml (git-ignored) holds this PC's device names and folders
$pidFile = Join-Path $root 'pipe.pid'
function Stop-App {
    if (Test-Path $pidFile) { Stop-Process -Id ([int](Get-Content $pidFile)) -Force -ErrorAction SilentlyContinue; Remove-Item $pidFile }
    # the sound recorder of --record is a separate ffmpeg: it would keep recording after the app is killed (Matroska audio survives that)
    Get-CimInstance Win32_Process -Filter "Name='ffmpeg.exe'" | Where-Object { $_.CommandLine -match 'dshow.*audio=' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'translator_overlay' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}
if ($Action -eq 'stop') { Stop-App; & (Join-Path $PSScriptRoot 'windows_server.ps1') stop; return }
Stop-App
if (-not $Record) { & (Join-Path $PSScriptRoot 'windows_server.ps1') start }  # recording needs no translation server
$env:PYTHONIOENCODING = 'utf-8'
$args = @('-m', 'translator_overlay', '-c', $Config, '--show') + $(if ($Windowed) { '--windowed' }) + $(if ($DebugLog) { '--debug' }) + $(if ($Record) { '--record', 'auto', '--no-ocr' })
$p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardError (Join-Path $root 'cam.err.log') -RedirectStandardOutput (Join-Path $root 'cam.out.log')
$p.Id | Out-File -Encoding ascii $pidFile
'app started in the background (fullscreen window opens in a few seconds); stop with: tools\windows_run.ps1 stop'
