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


class _AllGatherUneven(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        world_size = get_tensor_parallel_world_size()
        if world_size <= 1:
            return tensor

        local_size = torch.tensor([tensor.shape[-1]], device=tensor.device, dtype=torch.long)
        all_sizes = [torch.empty_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        all_sizes = torch.cat(all_sizes).tolist()
        
        ctx.all_sizes = all_sizes
        ctx.rank = get_tensor_parallel_rank()

        max_size = max(all_sizes)
        if tensor.shape[-1] < max_size:
            pad_shape = list(tensor.shape)
            pad_shape[-1] = max_size - tensor.shape[-1]
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            padded_tensor = torch.cat([tensor, padding], dim=-1)
        else:
            padded_tensor = tensor
        
        output_tensors = [torch.empty_like(padded_tensor) for _ in range(world_size)]
        dist.all_gather(output_tensors, padded_tensor)
        concatenated = torch.cat(output_tensors, dim=-1)
        
        total_size = sum(all_sizes)
        final_output = concatenated[..., :total_size]
        
        return final_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        world_size = get_tensor_parallel_world_size()
        if world_size <= 1:
            return grad_output
        
        all_sizes = ctx.all_sizes
        rank = ctx.rank
        
        start_index = sum(all_sizes[:rank])
        end_index = start_index + all_sizes[rank]
        
        return grad_output[..., start_index:end_index]

all_gather = _AllGather.apply
all_gather_uneven = _AllGatherUneven.apply