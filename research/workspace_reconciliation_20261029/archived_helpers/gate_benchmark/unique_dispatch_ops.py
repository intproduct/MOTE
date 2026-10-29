import torch


class UniqueSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(idx)
        ctx.shape = x.shape
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        result = grad.new_zeros(ctx.shape)
        result.index_copy_(0, idx, grad)
        return result, None


class UniqueAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, idx, source):
        ctx.save_for_backward(idx)
        ctx.mark_dirty(out)
        out.index_copy_(0, idx, out.index_select(0, idx) + source)
        return out

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        return grad, None, grad.index_select(0, idx)


unique_select = UniqueSelect.apply
unique_add = UniqueAdd.apply
