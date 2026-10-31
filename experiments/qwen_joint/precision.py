"""Candidate precision placement: route tensors never pass through expert dtype."""
import torch


def set_precision_preserving_routers(model, dtype, device):
    from .routers import JointRouter

    def convert(value, target_dtype):
        return value.to(device=device, dtype=target_dtype if value.is_floating_point() else None)

    def visit(module):
        if isinstance(module, JointRouter):
            module._apply(lambda value: convert(value, torch.float32))
            return
        for child in module.children():
            visit(child)
        # Visit only this module's own parameters/buffers, without recasting children.
        module._apply(lambda value: convert(value, dtype), recurse=False)

    visit(model)
    return model
