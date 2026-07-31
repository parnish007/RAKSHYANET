# Launch the RakshyaNet imagery classifier sidecar on port 8011.
#
# Everything heavy (venv, torch, HF weights, pip cache) lives on D: by design --
# C: does not have room for a 2.6 GB torch wheel plus model weights.

$ErrorActionPreference = "Stop"

$env:HF_HOME = "D:\rakshyanet-models\hf"
$env:TMP     = "D:\rakshyanet-models\tmp"
$env:TEMP    = "D:\rakshyanet-models\tmp"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Run from the sidecar directory so `service:app` resolves.
Set-Location $PSScriptRoot

D:\rakshyanet-models\venv\Scripts\python.exe -m uvicorn service:app --host 127.0.0.1 --port 8011
