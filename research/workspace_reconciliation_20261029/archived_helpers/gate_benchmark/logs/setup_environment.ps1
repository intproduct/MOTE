$ErrorActionPreference = 'Stop'
Set-Location 'D:\files\MOTE'
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = $env:HTTP_PROXY
$env:PIP_CACHE_DIR = 'D:\files\MOTE\outputs\pip_cache'
$env:HF_HOME = 'D:\files\MOTE\outputs\hf_cache'
$env:HF_HUB_DISABLE_XET = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PATH = 'D:\files\MOTE\.venv-titan;D:\files\MOTE\.venv-titan\Library\bin;D:\files\MOTE\.venv-titan\Scripts;' + $env:PATH
$python = 'D:\files\MOTE\.venv-titan\python.exe'
& $python -m pip install --upgrade pip *> outputs\gate_benchmark\logs\pip_upgrade.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118 *> outputs\gate_benchmark\logs\pip_torch.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -m pip install -r experiments/gate_benchmark/requirements-titan-v.txt matplotlib *> outputs\gate_benchmark\logs\pip_requirements.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -m pip install -e . --no-deps *> outputs\gate_benchmark\logs\pip_editable.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -m pip check *> outputs\gate_benchmark\logs\pip_check.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -m pip freeze *> outputs\gate_benchmark\logs\pip_freeze.txt
& $python -m fitmotn.experiments.gate_benchmark doctor *> outputs\gate_benchmark\logs\doctor.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -m pytest -q tests/test_structured_routers.py tests/test_gate_benchmark.py tests/test_quantum_gate.py tests/test_sparse_mixt.py tests/test_gate_router_arch.py tests/test_config_resume.py tests/test_repo_hygiene.py research/gate_design/test_candidates.py *> outputs\gate_benchmark\logs\pytest.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -m fitmotn.experiments.gate_benchmark smoke --output outputs/gate_smoke_20261027 *> outputs\gate_benchmark\logs\smoke.log
if ($LASTEXITCODE) { exit $LASTEXITCODE }
'PASS' | Set-Content outputs\gate_benchmark\logs\setup_complete.txt
