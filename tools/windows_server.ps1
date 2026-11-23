# Start/stop the translation server (llama-server, CUDA build) on Windows in the background; it outlives this script.
#   powershell -ExecutionPolicy Bypass -File tools\windows_server.ps1 start|stop
param([Parameter(Mandatory = $true)][ValidateSet('start', 'stop')][string]$Action)
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root 'server.pid'
if ($Action -eq 'stop') {
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    'server stopped'
    return
}
$exe = Join-Path $root 'third_party\llama-b12600-win\llama-server.exe'
$model = Join-Path $root 'models\plamo-3-translate-0.8b-Q4_K_M.gguf'
if (-not (Test-Path $exe)) { throw "missing $exe (official llama.cpp b12600 win-cuda-13.5 zip + its cudart zip, extracted together)" }
if (Get-Process llama-server -ErrorAction SilentlyContinue) { 'server already running'; return }
$p = Start-Process -FilePath $exe -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardError (Join-Path $root 'server.err.log') -RedirectStandardOutput (Join-Path $root 'server.out.log') `
    -ArgumentList @('-m', $model, '-ngl', '99', '-c', '2048', '-np', '1', '--host', '127.0.0.1', '--port', '8080', '--jinja', '--no-prefill-assistant')
$p.Id | Out-File -Encoding ascii $pidFile
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2).StatusCode -eq 200) { 'server ready (background)'; return } } catch { Start-Sleep -Milliseconds 500 }
}
throw 'server did not become ready; see server.err.log'
