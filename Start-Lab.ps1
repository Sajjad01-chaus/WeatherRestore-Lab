$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Missing project environment. Follow README setup instructions.' }
New-Item -ItemType Directory -Force -Path '.runtime/tmp', 'logs' | Out-Null
$env:TEMP = Join-Path $PSScriptRoot '.runtime/tmp'
$env:TMP = $env:TEMP
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
$taskLog = Join-Path $PSScriptRoot ('logs/viewer-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Start-Transcript -Path $taskLog
try {
    & $taskPython -m streamlit run app.py --server.address 127.0.0.1
    if ($LASTEXITCODE -ne 0) { throw "Viewer exited with code $LASTEXITCODE. See $taskLog" }
} finally {
    Stop-Transcript
}
