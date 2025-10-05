import torch
from torch.cuda import Stream
from ..ops.utils import IS_BF16_ATOM_ADD_SUPPORTED
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from .kv_cache import SimpleCacheManager
import torch.distributed as dist
from pygments.console import colorize


def maybe_pad_query(x, tile):
    if x.shape[1] % tile != 0:
        remain = tile - x.shape[1] % tile
        x_pad = torch.zeros(
            (x.shape[0], remain, x.shape[2], x.shape[3]), 
            dtype=x.dtype, device=x.device)
        x = torch.cat([x, x_pad], dim=1)
    return x 


class SimpleSparseCacheManager(SimpleCacheManager):
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim, page_budget, local_rank):
        super().__init__(batch_size, page_size, num_kv_heads, head_dim, local_rank)
        self.page_budget = page_budget

    def reset(self):
        super().reset()
        self.avg = None
        self.idx = []

    @torch.inference_mode()
    def remove_last_update(self):
        del self.idx[-1]
        return super().remove_last_update()
    
    @torch.inference_mode()
    def select(self, query, stage):
        if stage == 2:
            return
        end = sum(self.last_update_pages)
        if end <= self.page_budget:
            self.idx.append(None)
            return
        query = maybe_pad_query(query, self.page_size)
        query = query.transpose(1, 2).contiguous().unflatten(1, (self.num_kv_heads, -1)).sum(2)
        score = (query @ self.avg[..., :end]).softmax(dim=-1).unflatten(2, (-1, self.page_size)).sum([1,3])
        index = score.topk(k=self.page_budget, dim=-1, sorted=True).indices
        self.idx.append(index.to(torch.int32))

    @torch.inference_mode()
    def update(self, key, val, stage):
        if stage == 2:
            return
        key, val = super().update(key, val, stage)
        avg = torch.cat([k.mean(dim=1, keepdim=True) for k in key], dim=1)
        avg = avg.permute(0, 2, 3, 1).contiguous()
        self.avg = avg if self.avg is None else torch.cat([self.avg, avg], dim=-1)
        return key, val


class SparseCacheManager(SimpleSparseCacheManager):
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim, page_budget, local_rank):
        super().__init__(batch_size, page_size, num_kv_heads, head_dim, page_budget, local_rank)

        self.offload_future = None
        self.onload_future = None

        self.stream = Stream()
        self.pool = ThreadPoolExecutor(1)
        self.fake = torch.zeros(1, device='cuda')

    def reset(self):
        super().reset()
        self.key_cpu = []
        self.val_cpu = []
        self.kgd_cpu = []
        self.vgd_cpu = []

    @torch.inference_mode()
    def remove_last_update(self):
        self.wait_onload()
        self.wait_offload()

        update_token, update_pages = super().remove_last_update()

        if update_token is None:
            return 

        del self.key_cpu[-update_pages:]
        del self.val_cpu[-update_pages:]
        del self.kgd_cpu[-update_pages:]
        del self.vgd_cpu[-update_pages:]

    @torch.inference_mode()
    def update(self, key, val, stage):
        if stage == 2:
            return

        key, val = super().update(key, val, stage)
        gd_dtype = torch.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else torch.float32

        # Allocate memory on CPU
        key_cpu = [
            torch.empty_strided(
                size=x.data.size(), 
                stride=x.data.stride(), 
                dtype=x.data.dtype, 
                layout=x.data.layout, 
                device='cpu', 
                pin_memory=True)
            for x in key]
        
        val_cpu = [
            torch.empty_strided(
                size=x.data.size(),
                stride=x.data.stride(),
                dtype=x.data.dtype,
                layout=x.data.layout,
                device='cpu',
                pin_memory=True)
            for x in val]
        
        kgd_cpu = [
            torch.empty_strided(
                size=x.data.size(), 
                stride=x.data.stride(), 
                dtype=gd_dtype, 
                layout=x.data.layout, 
                device='cpu', 
                pin_memory=True)
            for x in key]
        
        vgd_cpu = [
            torch.empty_strided(
                size=x.data.size(), 
                stride=x.data.stride(), 
                dtype=gd_dtype, 
                layout=x.data.layout, 
                device='cpu', 
                pin_memory=True)
            for x in val]

        self.key_cpu.extend(key_cpu)
        self.val_cpu.extend(val_cpu)
        self.kgd_cpu.extend(kgd_cpu)
        self.vgd_cpu.extend(vgd_cpu)

    @torch.inference_mode()
    def select(self, query, stage):
        super().select(query, stage)

        sps_index = self.idx[-1]
        num_pages = len(self.key_cpu)
        rng_onload = num_pages if stage == 1 else (num_pages - self.last_update_pages[-1])
        
        @torch.inference_mode()
        def worker():

            if sps_index is None:
                # load all kv cache
                for i in range(num_pages):
                    self.key_gpu[i] = self.key_cpu[i].to(self.cuda, non_blocking=True)
                    self.val_gpu[i] = self.val_cpu[i].to(self.cuda, non_blocking=True)

                    if stage == 2:
                        self.kgd_gpu[i] = self.kgd_cpu[i].to(self.cuda, non_blocking=True)
                        self.vgd_gpu[i] = self.vgd_cpu[i].to(self.cuda, non_blocking=True)
                    else:
                        self.kgd_gpu[i] = self.fake
                        self.vgd_gpu[i] = self.fake
                return

            idx_set = set(sps_index.flatten().tolist())

            with torch.cuda.stream(self.stream):

                if stage == 1:
                    for i in range(rng_onload):
                        if i in idx_set:
                            self.key_gpu[i] = self.key_cpu[i].to(self.cuda, non_blocking=True)
                            self.val_gpu[i] = self.val_cpu[i].to(self.cuda, non_blocking=True)
                        else:
                            self.key_gpu[i] = self.fake
                            self.val_gpu[i] = self.fake

                        self.kgd_gpu[i] = self.fake
                        self.vgd_gpu[i] = self.fake

                elif stage == 2:
                    for i in range(rng_onload):
                        if i in idx_set:
                            self.key_gpu[i] = self.key_cpu[i].to(self.cuda, non_blocking=True)
                            self.val_gpu[i] = self.val_cpu[i].to(self.cuda, non_blocking=True)
                            self.kgd_gpu[i] = self.kgd_cpu[i].to(self.cuda, non_blocking=True)
                            self.vgd_gpu[i] = self.vgd_cpu[i].to(self.cuda, non_blocking=True)
                        else:
                            self.key_gpu[i] = self.fake
                            self.val_gpu[i] = self.fake
                            self.kgd_gpu[i] = self.fake
                            self.vgd_gpu[i] = self.fake

                    for i in range(rng_onload, num_pages):
                        self.key_gpu[i] = self.key_cpu[i].to(self.cuda, non_blocking=True)  
                        self.val_gpu[i] = self.val_cpu[i].to(self.cuda, non_blocking=True)
                        self.kgd_gpu[i] = self.kgd_cpu[i].to(self.cuda, non_blocking=True)
                        self.vgd_gpu[i] = self.vgd_cpu[i].to(self.cuda, non_blocking=True)

        if self.device != 'cuda':
            self.device = 'cuda'
            self.wait_onload()
            self.onload_future = self.pool.submit(worker)

    def offload(self, stage):
        num_pages = len(self.key_cpu)

        @torch.inference_mode()
        def worker():
            with torch.cuda.stream(self.stream):
                for i in range(num_pages):
                    if self.key_gpu[i] is not self.fake:
                        self.key_cpu[i].copy_(self.key_gpu[i])
                        self.val_cpu[i].copy_(self.val_gpu[i])

                        if stage == 2:
                            self.kgd_cpu[i].copy_(self.kgd_gpu[i])
                            self.vgd_cpu[i].copy_(self.vgd_gpu[i])

                    self.key_gpu[i] = None
                    self.val_gpu[i] = None
                    self.kgd_gpu[i] = None
                    self.vgd_gpu[i] = None

        if self.device != 'cpu':
            self.device = 'cpu'
            self.wait_offload()
            self.offload_future = self.pool.submit(worker)

    def wait_onload(self):
        if self.onload_future is not None:
            self.onload_future.result()
            self.onload_future = None

    def wait_offload(self):
        if self.offload_future is not None:
            self.offload_future.result()
            self.offload_future = None

    @property
    @torch.inference_mode()
    def page_table(self):
        self.wait_onload()
        return super().page_table

    @property
    @torch.inference_mode()
    def grad(self):
        self.wait_onload()
        return super().grad


class SparseKVCache:
    def __init__(
        self, 
        num_layers: int = 28, 
        batch_size: int = 1, 
        page_size: int = 64,
        num_heads: int = 4,
        head_dim: int = 128,
        page_budget: int = 128,
        cpu_offload=None,
        local_rank=None):

        self.num_layers = num_layers    
        self.cpu_offload = cpu_offload

        MANAGER_CLS = SimpleSparseCacheManager if cpu_offload is None else SparseCacheManager

        self.managers = [
            MANAGER_CLS(
                batch_size,
                page_size,
                num_heads,
                head_dim,
                page_budget,
                local_rank)
            for _ in range(num_layers)]

    def reset(self):
        for m in self.managers:
            m.reset()

    def visit(self, layer_idx, stage, reverse=False):
        if self.cpu_offload is not None:
            factor = -1 if reverse else 1
            cuda_layers = [
                (layer_idx + self.num_layers + factor * i) % self.num_layers 
                for i in range(self.cpu_offload)]
            cpu_layers = list(filter(lambda i: i not in cuda_layers, range(self.num_layers)))
            
            for lid in cpu_layers:
                self.managers[lid].offload(stage)

            if reverse:
                for lid in cuda_layers:
                    self.managers[lid].select(None, 2)

    def __getitem__(self, idx):
        return self.managers[idx]

    @property
    def device(self):
        return (m.device for m in self.managers)
    
    def pre_process(self):
        for idx, m in enumerate(self.managers):
            m.grad_hook = lambda idx=idx: self.visit(idx, 2, True)

    def post_process(self):
        for m in self.managers:
            m.remove_last_update()
