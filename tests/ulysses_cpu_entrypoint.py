"""CPU-only launcher acceptance shim: disable optional DeepSpeed C++ SHM JIT.

This does not replace Ulysses or Gloo collectives. Production CUDA launches use
python -m fitmotn.train.ulysses directly, without this shim.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(DS_ACCELERATOR="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")

import deepspeed

deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False

from fitmotn.train.ulysses import main

if __name__ == "__main__":
    main()
