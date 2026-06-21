from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import torch as tc


def get_mlp_module(model: tc.nn.Module, layer_idx: int):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise AttributeError("model does not have model.layers")
    if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
        raise IndexError(f"layer {layer_idx} out of range for {len(layers)} layers")
    mlp = getattr(layers[int(layer_idx)], "mlp", None)
    if mlp is None:
        raise AttributeError(f"model.model.layers[{layer_idx}] does not have mlp")
    return mlp


@contextmanager
def capture_mlp_io(model: tc.nn.Module, layers: Iterable[int], *, cpu: bool = False) -> Iterator[Tuple[Dict[int, Dict[str, tc.Tensor]], List[str]]]:
    captured: Dict[int, Dict[str, tc.Tensor]] = {}
    warnings: List[str] = []
    handles = []

    def store(t: Any) -> tc.Tensor:
        if isinstance(t, tuple):
            t = t[0]
        out = t.detach()
        return out.cpu() if cpu else out

    for layer_idx in layers:
        try:
            mlp = get_mlp_module(model, int(layer_idx))
        except Exception as exc:
            warnings.append(str(exc))
            continue

        def pre_hook(module, args, idx=int(layer_idx)):
            if args:
                captured.setdefault(idx, {})["mlp_input"] = store(args[0])

        def fwd_hook(module, args, output, idx=int(layer_idx)):
            captured.setdefault(idx, {})["mlp_output"] = store(output)

        handles.append(mlp.register_forward_pre_hook(pre_hook))
        handles.append(mlp.register_forward_hook(fwd_hook))

    try:
        yield captured, warnings
    finally:
        for handle in handles:
            handle.remove()
