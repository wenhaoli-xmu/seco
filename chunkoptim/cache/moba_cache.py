import torch
from torch.cuda import Stream
from functools import partial

from ..page_attn.flash_paged_attn import IS_BF16_ATOM_ADD_SUPPORTED
from .kv_cache import CacheManager


class MoBACacheManager(CacheManager):
    """
    An extension of CacheManager that incorporates the MoBA (Mixture-of-Bases Attention)
    sparse block selection mechanism.
    
    This manager not only stores KV pages but also pre-computes and stores a summary
    (mean vector) for each key page. It provides a `select_blocks` method to efficiently
    find the most relevant pages for a given query.
    """
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim, moba_topk):
        super().__init__(batch_size, page_size, num_kv_heads, head_dim)
        
        if moba_topk <= 0:
            raise ValueError("moba_topk must be a positive integer.")
            
        self.moba_topk = moba_topk
        self.moba_chunk_size = self.page_size
        
        self.k_block_summary_tensors = torch.nn.ParameterDict()
        
    def reset(self):
        super().reset()
        self.k_block_summary_tensors = torch.nn.ParameterDict()

    def remove_last_update(self):
        if not self.last_update_pages:
            return
        
        update_pages_to_remove = self.last_update_pages[-1]
        current_total_pages = sum(self.last_update_pages)
        
        page_indices_to_remove = range(
            current_total_pages - update_pages_to_remove, 
            current_total_pages
        )

        for page_idx in page_indices_to_remove:
            page_str = str(page_idx)
            if page_str in self.k_block_summary_tensors:
                del self.k_block_summary_tensors[page_str]
        
        super().remove_last_update()


    @torch.inference_mode()
    def update(self, key, val):
        assert key.dtype == torch.bfloat16, 'only bfloat16 is supported'
        update_token = key.shape[1]
        self.device = key.device

        if update_token % self.page_size != 0:
            assert self.num_kv % self.page_size == 0
            pad_len = self.page_size - (update_token % self.page_size)
            key = torch.cat([key, torch.zeros((key.shape[0], pad_len, key.shape[2], key.shape[3]), device=key.device, dtype=key.dtype)], dim=1)
            val = torch.cat([val, torch.zeros((val.shape[0], pad_len, val.shape[2], val.shape[3]), device=val.device, dtype=val.dtype)], dim=1)
        
        gd_dtype = torch.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else torch.float32
        kgd = torch.zeros_like(key, dtype=gd_dtype)
        vgd = torch.zeros_like(val, dtype=gd_dtype)
        
        num_new_pages = key.shape[1] // self.page_size
        key_pages = list(torch.chunk(key, chunks=num_new_pages, dim=1))
        val_pages = list(torch.chunk(val, chunks=num_new_pages, dim=1))
        kgd_pages = list(torch.chunk(kgd, chunks=num_new_pages, dim=1))
        vgd_pages = list(torch.chunk(vgd, chunks=num_new_pages, dim=1))

        # ========================================================================
        key_summaries = [k_page.mean(dim=1, keepdim=True) for k_page in key_pages]
        
        current_num_pages = sum(self.last_update_pages)
        summary_tensors = {
            str(current_num_pages + i): torch.nn.Buffer(summary.data)
            for i, summary in enumerate(key_summaries)
        }
        self.k_block_summary_tensors.update(summary_tensors)
        # ========================================================================

        self.num_kv += update_token
        update_pages = len(key_pages)

        key_tensors = {str(current_num_pages + i): torch.nn.Buffer(x.data) for i, x in enumerate(key_pages)}
        val_tensors = {str(current_num_pages + i): torch.nn.Buffer(x.data) for i, x in enumerate(val_pages)}
        kgd_tensors = {str(current_num_pages + i): torch.nn.Buffer(x.data) for i, x in enumerate(kgd_pages)}
        vgd_tensors = {str(current_num_pages + i): torch.nn.Buffer(x.data) for i, x in enumerate(vgd_pages)}
        
        self.key_tensors.update(key_tensors)
        self.val_tensors.update(val_tensors)
        self.kgd_tensors.update(kgd_tensors)
        self.vgd_tensors.update(vgd_tensors)

        self.last_update_token.append(update_token)
        self.last_update_pages.append(update_pages)


    @torch.inference_mode()
    def select_blocks(self, q: torch.Tensor) -> torch.Tensor:
        num_blocks = len(self.k_block_summary_tensors)
        if num_blocks == 0:
            return torch.empty(q.shape[0], q.shape[2], q.shape[1], 0, dtype=torch.long, device=q.device)

        # TODO: 1. 加入sink选择机制，永远都选择第一个block并且放在序列的最前面
        # TODO: 2. 永远加入local window

        summaries = [self.k_block_summary_tensors[str(i)][0] for i in range(num_blocks)]
        key_gate_weight = torch.stack(summaries, dim=0)
        key_gate_weight = key_gate_weight.squeeze(1)
        key_gate_weight = key_gate_weight.repeat_interleave(q.shape[2] // self.num_kv_heads, dim=1)

        q = q.transpose(1, 2)
        gate = torch.einsum(
            "bhsd,nhd->bhsn", 
            q, key_gate_weight)

        k = min(self.moba_topk, num_blocks)
        _, topk_indices = torch.topk(gate, k=k, dim=-1, largest=True)
        
        # NOTE: 这是一个暂时的work around，并不能从根本上解决问题
        return topk_indices.sort(dim=-1).values