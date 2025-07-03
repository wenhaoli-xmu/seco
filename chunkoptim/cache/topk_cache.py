import torch
from torch.cuda import Stream
from functools import partial
from ..ops.utils import IS_BF16_ATOM_ADD_SUPPORTED
from .kv_cache import CacheManager, LayerCache, KVCache


class SparseCacheManager(CacheManager):
    """
    An extension of CacheManager that incorporates a sparse block selection mechanism.
    
    This manager not only stores KV pages but also pre-computes and stores a summary
    (mean vector) for each key page. It provides a `select_blocks` method to efficiently
    find the most relevant pages for a given query, always including the first and the most
    recent blocks.
    """
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim, sparse_topk):
        super().__init__(batch_size, page_size, num_kv_heads, head_dim)
        
        if sparse_topk <= 0:
            raise ValueError("sparse_topk must be a positive integer.")
            
        self.sparse_topk = sparse_topk
        self.query_block_size = self.page_size
        
        self.k_block_summary_tensors = torch.nn.ParameterDict()
        
    def reset(self):
        super().reset()
        self.k_block_summary_tensors = torch.nn.ParameterDict()

    def remove_last_update(self):
        if not self.last_update_pages:
            return
        
        pages_to_remove_count = self.last_update_pages[-1]
        current_total_pages = sum(self.last_update_pages)
        
        start_page_idx = current_total_pages - pages_to_remove_count
        end_page_idx = current_total_pages

        for page_idx in range(start_page_idx, end_page_idx):
            page_str = str(page_idx)
            if page_str in self.k_block_summary_tensors:
                del self.k_block_summary_tensors[page_str]
        
        super().remove_last_update()


    @torch.inference_mode()
    def update(self, key, val):
        # Use the base class update for tensors
        super().update(key, val)

        # Now, handle the summary tensors specific to SparseCacheManager
        num_new_pages = (key.shape[1] + self.page_size - 1) // self.page_size
        key_padded = torch.cat([key, torch.zeros(key.shape[0], num_new_pages * self.page_size - key.shape[1], *key.shape[2:], device=key.device, dtype=key.dtype)], dim=1)
        key_pages = list(torch.chunk(key_padded, chunks=num_new_pages, dim=1))

        key_summaries = [k_page.mean(dim=1, keepdim=True) for k_page in key_pages]
        
        # The number of pages before this update
        current_num_pages = len(self.k_block_summary_tensors)
        summary_tensors = {
            # Note: The original code was creating summaries with a batch dim of 1.
            # We should squeeze it out before storing.
            str(current_num_pages + i): torch.nn.Parameter(summary.squeeze(0).data, requires_grad=False)
            for i, summary in enumerate(key_summaries)
        }
        self.k_block_summary_tensors.update(summary_tensors)


    @torch.inference_mode()
    def select_blocks(self, q: torch.Tensor) -> torch.Tensor:
        num_blocks = len(self.k_block_summary_tensors)
        if num_blocks == 0:
            num_query_blocks = (q.shape[1] + self.query_block_size - 1) // self.query_block_size
            return torch.empty(q.shape[0], q.shape[2], num_query_blocks, 0, dtype=torch.long, device=q.device)

        summaries = [self.k_block_summary_tensors[str(i)] for i in range(num_blocks)]
        key_gate_weight = torch.stack(summaries, dim=0).squeeze(1)

        if q.shape[2] > self.num_kv_heads:
            key_gate_weight = key_gate_weight.repeat_interleave(q.shape[2] // self.num_kv_heads, dim=1)

        q_transposed = q.transpose(1, 2)
        gate = torch.einsum("bhsd,nhd->bhsn", q_transposed, key_gate_weight)

        b, h, s, n = gate.shape

        if s % self.query_block_size != 0:
            raise ValueError(f"Query sequence length ({s}) must be a multiple of query_block_size ({self.query_block_size}).")

        num_query_blocks = s // self.query_block_size
        gate_reshaped = gate.view(b, h, num_query_blocks, self.query_block_size, n)
        block_scores = gate_reshaped.sum(dim=3)

        k = min(self.sparse_topk, num_blocks)

        block_scores = torch.where(
            torch.ones_like(block_scores,dtype=torch.bool).tril(block_scores.shape[-1] - block_scores.shape[-2]),
            block_scores,
            float('-inf'))
        
        block_scores = torch.diagonal_scatter(
            block_scores, 
            torch.full((1, block_scores.shape[1], block_scores.shape[-2],), fill_value=float('inf'), dtype=block_scores.dtype, device=block_scores.device),
            offset=block_scores.shape[-1] - block_scores.shape[-2],
            dim1=2, dim2=3)
        
        block_scores[:, :, :, 0] = float('inf')

        top_indices = torch.topk(block_scores, k=k, dim=-1).indices.to(torch.int32)

        return top_indices
    
class SparseLayerCache(LayerCache):
    def __init__(self, batch_size, page_size, num_heads, head_dim, page_budget):
        torch.nn.Module.__init__(self)
        self.manager = SparseCacheManager(
            batch_size=batch_size, 
            page_size=page_size, 
            num_kv_heads=num_heads, 
            head_dim=head_dim,
            sparse_topk=page_budget)
        self.reset()


class SparseKVCache(KVCache):
    def __init__(
        self, 
        num_layers: int = 28, 
        batch_size: int = 1, 
        page_size: int = 64,
        num_heads: int = 4,
        head_dim: int = 128,
        cpu_offload=None,
        num_streams: int = 4,
        page_budget: int = 128):
    
        self.num_layers = num_layers    
        self.cpu_offload = cpu_offload

        self.streams = [Stream() for _ in range(num_streams)]
        self.stream_idx = 0

        self.cache = [
            SparseLayerCache(
                batch_size,
                page_size,
                num_heads,
                head_dim,
                page_budget)
            for _ in range(num_layers)]