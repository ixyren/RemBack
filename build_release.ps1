$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Read workspace version from Cargo.toml
$cargoToml = Get-Content -Raw -Path (Join-Path $root "Cargo.toml")
if ($cargoToml -match 'version\s*=\s*"([^"]+)"') {
    $version = $Matches[1]
} else {
    throw "Could not determine version from Cargo.toml"
}

Write-Host "Building RemBack v$version..." -ForegroundColor Cyan

Write-Host "[1/3] Building remback-app (release)..." -ForegroundColor Green
cargo build --release -p remback-app
if ($LASTEXITCODE -ne 0) { throw "remback-app build failed" }

Write-Host "[2/3] Building remback-installer (release)..." -ForegroundColor Green
cargo build --release -p remback-installer
if ($LASTEXITCODE -ne 0) { throw "remback-installer build failed" }

Write-Host "[3/3] Packaging dist/RemBack-v$version-Setup.exe..." -ForegroundColor Green
$distDir = Join-Path $root "dist"
if (-not (Test-Path $distDir)) {
    New-Item -ItemType Directory -Path $distDir | Out-Null
}

$installerSrc = Join-Path $root "target\release\remback-installer.exe"
$installerDst = Join-Path $distDir "RemBack-v$version-Setup.exe"
Copy-Item -Force -Path $installerSrc -Destination $installerDst

$appSizeMB = [math]::Round((Get-Item (Join-Path $root "target\release\remback.exe")).Length / 1MB, 2)
$setupSizeMB = [math]::Round((Get-Item $installerDst).Length / 1MB, 2)

Write-Host "Done!" -ForegroundColor Cyan
Write-Host "  remback.exe                : $appSizeMB MB"
Write-Host "  RemBack-v$version-Setup.exe : $setupSizeMB MB"
