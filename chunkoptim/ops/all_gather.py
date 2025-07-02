import torch
import torch.distributed as dist


def get_tensor_parallel_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def get_tensor_parallel_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.world_size = get_tensor_parallel_world_size()
        if ctx.world_size <= 1:
            return tensor

        output_tensors = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(output_tensors, tensor)
        
        return torch.cat(output_tensors, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.world_size <= 1:
            return grad_output

        total_vocab_size = grad_output.shape[-1]
        vocab_size_per_partition = total_vocab_size // ctx.world_size
        
        rank = get_tensor_parallel_rank()
        start_index = rank * vocab_size_per_partition
        end_index = start_index + vocab_size_per_partition
        
        return grad_output[..., start_index:end_index]