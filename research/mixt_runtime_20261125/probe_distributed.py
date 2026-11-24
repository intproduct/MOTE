"""Check real two-GPU collective availability without allocating a model."""
import datetime
import json
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, devices, port, output):
    row = dict(rank=rank, device=devices[rank], gpu=torch.cuda.get_device_name(devices[rank]))
    try:
        torch.cuda.set_device(devices[rank])
        row["native_bf16"] = torch.cuda.is_bf16_supported(including_emulation=False)
        dist.init_process_group("gloo", rank=rank, world_size=2,
            init_method=f"tcp://127.0.0.1:{port}?use_libuv=0", timeout=datetime.timedelta(seconds=25))
        cpu = torch.tensor(float(rank+1))
        dist.all_reduce(cpu)
        row["cpu_all_reduce"] = float(cpu)
        gpu = torch.tensor(float(rank+1), device=f"cuda:{devices[rank]}")
        dist.all_reduce(gpu)
        row["cuda_all_reduce"] = float(gpu)
        row["ok"] = float(gpu) == 3.0
    except Exception as exc:
        row.update(ok=False, error=repr(exc))
    finally:
        Path(output, f"rank_{rank}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["USE_LIBUV"] = "0"
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    devices = [i for i in range(torch.cuda.device_count()) if "V100" in torch.cuda.get_device_name(i)][:2]
    assert len(devices) == 2
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(devices, port, str(output)), nprocs=2, join=True)
    rows = [json.loads((output / f"rank_{rank}.json").read_text()) for rank in range(2)]
    print(json.dumps(rows, indent=2))
