"""Native device observation and exclusive, owned per-UUID leases."""
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

PRO4000 = "GPU-67f0c332-3f01-a733-4651-5532e94fc051"
V100 = ("GPU-f145a48f-ccc7-cccf-94ad-acf680ff0891", "GPU-3753e6c9-3189-164f-a329-507d9a4a16f6")


def desktop_process(name, path):
    """Recognize the observed WDDM desktop processes, not arbitrary compute jobs."""
    name, path = name.lower(), (path or "").replace("/", "\\").lower()
    if name in ("dwm.exe", "system"):
        return not path or path.startswith("c:\\windows\\system32\\")
    allowed = {
        "explorer.exe": ("c:\\windows\\",),
        "startmenuexperiencehost.exe": ("c:\\windows\\systemapps\\",),
        "searchapp.exe": ("c:\\windows\\systemapps\\",),
        "textinputhost.exe": ("c:\\windows\\systemapps\\",),
        "shellexperiencehost.exe": ("c:\\windows\\systemapps\\",),
        "msedgewebview2.exe": ("c:\\program files (x86)\\microsoft\\edgewebview\\",),
        "msedge.exe": ("c:\\program files (x86)\\microsoft\\edge\\",),
        "copilotapp.exe": ("c:\\program files (x86)\\microsoft\\copilot\\",),
        "chatgpt.exe": ("c:\\program files\\windowsapps\\openai.codex_",),
        "phoneexperiencehost.exe": ("c:\\program files\\windowsapps\\microsoft.yourphone_",),
        "microsoft.media.player.exe": ("c:\\program files\\windowsapps\\microsoft.zunemusic_",),
        "cortana.exe": ("c:\\program files\\windowsapps\\microsoft.549981c3f5f10_",),
        "steamwebhelper.exe": ("d:\\steam\\bin\\cef\\",),
        "wallpaper64.exe": ("d:\\steam\\steamapps\\common\\wallpaper_engine\\",),
    }
    return name in allowed and path.startswith(allowed[name]) and path.endswith("\\" + name)


def command(args):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.check_output(args, text=True, encoding="utf8", errors="replace", timeout=25, creationflags=flags).strip()


def gpu_snapshot(device_uuid):
    if device_uuid not in (PRO4000, *V100):
        raise ValueError("Unknown experiment GPU UUID")
    row = next(csv.reader([command(["nvidia-smi", "-i", device_uuid,
        "--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])]))
    actual, name, memory, utilization, temperature = [v.strip() for v in row]
    if actual != device_uuid:
        raise RuntimeError("GPU identity mismatch")
    native = command(["nvidia-smi", "-i", device_uuid, "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"])
    process_info = {}
    if os.name == "nt" and native:
        info = json.loads(command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,Name,ExecutablePath | ConvertTo-Json -Compress"]))
        process_info = {str(p["ProcessId"]): p for p in info}
    applications, display = [], []
    for app in csv.reader(native.splitlines()):
        if len(app) < 3:
            continue
        app = [v.strip() for v in app]
        info = process_info.get(app[1], {})
        path = info.get("ExecutablePath") or app[2]
        if path == "[Insufficient Permissions]":
            path = ""
        is_display = desktop_process(info.get("Name", ""), path)
        (display if is_display and device_uuid == PRO4000 else applications).append(app)
    wsl = None
    if os.name == "nt" and device_uuid == PRO4000:
        wsl = command(["wsl", "-d", "Ubuntu-24.04", "--", "nvidia-smi", "-i", "0",
                       "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"])
        for app in csv.reader(wsl.splitlines()):
            if len(app) >= 3 and app[0].strip() == device_uuid:
                applications.append([v.strip() for v in app])
    limit = 3000 if device_uuid == PRO4000 else 512
    return {"epoch": time.time(), "uuid": actual, "name": name, "memory_mib": int(memory),
            "utilization": int(utilization), "temperature_c": int(temperature), "compute_apps": applications,
            "display_apps": display, "wsl_raw": wsl,
            "idle": not applications and int(memory) <= limit and int(utilization) <= 5 and int(temperature) < 75}


class GPULease:
    def __init__(self, root, device_uuid, purpose, run_directory):
        self.root = Path(root)
        self.device_uuid, self.purpose, self.run_directory = device_uuid, purpose, str(run_directory)
        self.token = str(uuid.uuid4())
        self.path = self.root / "locks" / (device_uuid + ".lock")
        self.record = None

    def __enter__(self):
        for stop in (self.root / "STOP", self.root / "queue-20261030" / "STOP", Path(self.run_directory) / "STOP"):
            if stop.exists():
                raise RuntimeError(f"Stop marker: {stop}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        first = gpu_snapshot(self.device_uuid)
        if not first["idle"]:
            raise RuntimeError(f"Device busy or hot: {first}")
        self.record = {"uuid": self.device_uuid, "pid": os.getpid(), "token": self.token,
                       "created_epoch": time.time(), "purpose": self.purpose,
                       "run_directory": self.run_directory, "observation": first}
        # Never steal an existing lock, including one whose process looks stale.
        with self.path.open("x", encoding="utf8") as f:
            json.dump(self.record, f, indent=2)
        try:
            second = gpu_snapshot(self.device_uuid)
            if not second["idle"]:
                raise RuntimeError(f"Device changed during acquisition: {second}")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_uuid
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        if self.path.exists():
            record = json.loads(self.path.read_text(encoding="utf8"))
            if record.get("token") == self.token and record.get("pid") == os.getpid():
                self.path.unlink()
