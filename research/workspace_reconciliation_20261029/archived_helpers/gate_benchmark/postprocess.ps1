$ErrorActionPreference = 'Stop'
Set-Location 'D:\files\MOTE'
$env:PATH='D:\files\MOTE\.venv-titan\Library\bin;'+$env:PATH
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:HF_HOME='D:\files\MOTE\outputs\hf_cache'
$env:HF_HUB_OFFLINE='1'
$python='D:\files\MOTE\.venv-titan\python.exe'
Wait-Process -Id 12116 -ErrorAction SilentlyContinue
if (-not (Test-Path outputs\gate_benchmark\45c21049b433\execution_complete.json)) { throw 'Formal matrix did not finish; failure artifacts retained.' }
& $python outputs\gate_benchmark\audit_results.py experiments/gate_benchmark/configs/titan_v.json --check-initial *> outputs\gate_benchmark\logs\formal_audit.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -m fitmotn.experiments.gate_benchmark summarize *> outputs\gate_benchmark\logs\formal_summarize.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python outputs\gate_benchmark\analyze_results.py *> outputs\gate_benchmark\logs\formal_analysis.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python outputs\gate_benchmark\serial_efficiency.py *> outputs\gate_benchmark\logs\serial_efficiency.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python outputs\gate_benchmark\plot_serial_efficiency.py *> outputs\gate_benchmark\logs\serial_efficiency_plot.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python outputs\gate_benchmark\make_report.py *> outputs\gate_benchmark\logs\make_report.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Get-Date -Format o | Set-Content outputs\gate_benchmark\logs\postprocess_complete.txt
