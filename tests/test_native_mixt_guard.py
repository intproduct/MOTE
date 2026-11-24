from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fitmotn.patching import patch_qwen_ffn_layers


@pytest.mark.parametrize("backend", ["motn", "sparse_mixt", "adtn_fixed"])
def test_pretrained_mixt_rejected_before_mutating_any_selected_layer(backend):
    def make_mlp():
        mlp = nn.Module()
        mlp.gate_proj = nn.Linear(4, 8)
        mlp.up_proj = nn.Linear(4, 8)
        mlp.down_proj = nn.Linear(8, 4)
        return mlp
    original = [make_mlp(), make_mlp()]
    original[1].gate_proj.is_mixt_projector = True
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=x) for x in original]))
    with pytest.raises(ValueError, match="weight-preserving MixT importer"):
        patch_qwen_ffn_layers(model, [0, 1], {"patch_backend": backend}, torch.device("cpu"))
    assert all(layer.mlp is old for layer, old in zip(model.model.layers, original))
