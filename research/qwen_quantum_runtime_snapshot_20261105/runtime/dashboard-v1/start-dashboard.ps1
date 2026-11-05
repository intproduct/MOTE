$ErrorActionPreference = 'Stop'
$dashboardRoot = 'E:\AI\MOTE-quantum\runtime\dashboard-v1'
$dashboardPython = 'E:\AI\MOTE-quantum\envs\qwen-native-py312-cu128\Scripts\python.exe'
try {
    $dashboardReply = Invoke-RestMethod -Uri 'http://127.0.0.1:8766/api/status' -TimeoutSec 5
    if ($dashboardReply.summary.total -eq 30 -and $dashboardReply.rows.Count -eq 30) {
        Write-Output 'Dashboard already running: http://127.0.0.1:8766'
        exit 0
    }
} catch { }
$env:PYTHONUTF8 = '1'
$env:PYTHONPYCACHEPREFIX = 'E:\AI\MOTE-quantum\runtime\pycache'
$env:CUDA_VISIBLE_DEVICES = ''
$env:TEMP = $dashboardRoot
$env:TMP = $dashboardRoot
$dashboardStamp = Get-Date -Format 'yyyyMMddTHHmmss'
$dashboardProcess = Start-Process -FilePath $dashboardPython -ArgumentList @('-u', "$dashboardRoot\server.py", '--port', '8766') -WorkingDirectory 'D:\AI\MOTE' -WindowStyle Hidden -RedirectStandardOutput "$dashboardRoot\server-$dashboardStamp.stdout.log" -RedirectStandardError "$dashboardRoot\server-$dashboardStamp.stderr.log" -PassThru
Write-Output "Dashboard launcher PID: $($dashboardProcess.Id)"
