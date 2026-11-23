# One-time setup of translator-overlay on a Windows 10/11 machine with an NVIDIA GPU:
#   1. the Python environment (pinned versions), 2. llama-server (prebuilt CUDA release, SHA-256 verified),
#   3. the translation model (SHA-256 verified).
# Needs: `uv` (winget install --id=astral-sh.uv -e) and an NVIDIA driver that supports CUDA 13.
#   powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1 [-SkipModel]
param([switch]$SkipModel)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$llama = 'https://github.com/ggml-org/llama.cpp/releases/download/b12600'
$assets = @(
    @{ name = 'llama-b12600-bin-win-cuda-13.5-x64.zip';  sha = '6c3d4724732b1a520e7e100585a0928e5a27521cd6d022fbab4b573dd1c0b52a' },
    @{ name = 'cudart-llama-bin-win-cuda-13.5-x64.zip'; sha = 'c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba' })
$model = @{ url = 'https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/main/plamo-3-translate-0.8b-Q4_K_M.gguf'
            file = 'models\plamo-3-translate-0.8b-Q4_K_M.gguf'; sha = '70891ec70dc7ab4205745a38307867c2881b99239f20327927633abfe4bc9ec6' }

function Get-Verified([string]$url, [string]$dest, [string]$sha) {
    if (-not (Test-Path $dest) -or (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower() -ne $sha) {
        Write-Host "downloading $url"
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        Invoke-WebRequest -Uri $url -OutFile $dest
    }
    $got = (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $sha) { Remove-Item $dest; throw "checksum mismatch for $dest (got $got, expected $sha); deleted, nothing was installed from it" }
    Write-Host "verified $dest"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is not installed: winget install --id=astral-sh.uv -e' }
if (-not (Test-Path .venv)) { uv venv --python 3.12 .venv }
uv pip install --python .venv\Scripts\python.exe -r requirements-windows.txt

$dir = Join-Path $root 'third_party\llama-b12600-win'
$tmp = Join-Path $root 'third_party\_downloads'
foreach ($a in $assets) {
    Get-Verified "$llama/$($a.name)" (Join-Path $tmp $a.name) $a.sha
    Expand-Archive -Force -Path (Join-Path $tmp $a.name) -DestinationPath $dir
}
if (-not (Test-Path (Join-Path $dir 'llama-server.exe'))) { throw "llama-server.exe not found in $dir after extracting (the zip layout changed?)" }
Remove-Item -Recurse -Force $tmp

if (-not $SkipModel) { Get-Verified $model.url (Join-Path $root $model.file) $model.sha }

Write-Host ''
Write-Host 'Done. Next: copy config.capture.toml to config.local.toml, set your capture card under [source] (and [audio] for sound), then run tools\windows_run.ps1 start (see the README, Quick start).'
