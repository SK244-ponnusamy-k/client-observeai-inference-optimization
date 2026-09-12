# ==============================================================================
# oai.ps1 — launcher shim (Windows PowerShell)
#
# Runs the Python `oai` CLI from tools/. Call it as .\oai.ps1, or add this folder
# to PATH so you can type:  oai model new
# ==============================================================================
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$ScriptDir\tools;$env:PYTHONPATH"
$python = if ($env:OAI_PYTHON) { $env:OAI_PYTHON } else { "python" }
& $python -m oai.cli @args
exit $LASTEXITCODE
