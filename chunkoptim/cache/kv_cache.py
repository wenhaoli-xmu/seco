import torch
from torch.cuda import Stream
from ..ops.utils import IS_BF16_ATOM_ADD_SUPPORTED
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import torch.distributed as dist


class SimpleCacheManager:
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim, local_rank):
        super().__init__()
        self.batch_size = batch_size
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = 'cuda'
        self.cuda = f'cuda:{dist.get_rank() if local_rank is None else local_rank}'
        self.reset()

    def reset(self):
        self.num_kv = 0
        self.last_update_token = []
        self.last_update_pages = []
        self.grad_hook = None
        self.key_gpu = []
        self.val_gpu = []
        self.kgd_gpu = []
        self.vgd_gpu = []

    @property
    @torch.inference_mode()
    def page_table(self):
        num_pages = sum(self.last_update_pages)
        assert num_pages == len(self.key_gpu)

        page_table = []
        for i in range(num_pages):
            page_table.append((
                self.key_gpu[i].data_ptr(), 
                self.val_gpu[i].data_ptr(), 
                self.kgd_gpu[i].data_ptr(), 
                self.vgd_gpu[i].data_ptr()))

        page_table = torch.tensor(
            page_table, 
            dtype=int, 
            device=self.cuda)

        return page_table

    @property
    @torch.inference_mode()
    def grad(self):
        if self.grad_hook is not None:
            self.grad_hook()

        num_pages = self.last_update_pages[-1]
        last_update_kgd = self.kgd_gpu[-num_pages:]
        last_update_vgd = self.vgd_gpu[-num_pages:]

        kgd = torch.cat(last_update_kgd, dim=1)[:, :self.last_update_token[-1]]
        vgd = torch.cat(last_update_vgd, dim=1)[:, :self.last_update_token[-1]]

        if not IS_BF16_ATOM_ADD_SUPPORTED:
            kgd = kgd.to(torch.bfloat16)
            vgd = vgd.to(torch.bfloat16)

        return kgd, vgd
    
    @torch.inference_mode()
    def onload(self, stage):
        ...

    @torch.inference_mode()
    def offload(self, stage):
        ...

    @torch.inference_mode()
    def remove_last_update(self):

        if len(self.last_update_pages) == 1:
            self.reset()
            return None, None
        
        # Update meta data
        self.last_update_token, update_token = self.last_update_token[:-1], self.last_update_token[-1]
        self.last_update_pages, update_pages = self.last_update_pages[:-1], self.last_update_pages[-1]
        self.num_kv -= update_token

        del self.key_gpu[-update_pages:]
        del self.val_gpu[-update_pages:]
        del self.kgd_gpu[-update_pages:]
        del self.vgd_gpu[-update_pages:]

        return update_token, update_pages

    @torch.inference_mode()
    def update(self, key, val, stage):
        assert stage in (1, 2)
        if stage == 2:
            return

        assert key.dtype == torch.bfloat16, 'only bfloat16 is supported'

        update_token = key.shape[1]

        # Pad key and value
        if update_token % self.page_size != 0:
            assert self.num_kv % self.page_size == 0
            pad_len = self.page_size - (update_token % self.page_size)
            key = torch.cat([
                key, 
                torch.zeros((key.shape[0], pad_len, key.shape[2], key.shape[3]), 
                device=key.device, 
                dtype=key.dtype)], 
                dim=1)
            val = torch.cat([
                val, 
                torch.zeros((val.shape[0], pad_len, val.shape[2], val.shape[3]), 
                device=val.device, 
                dtype=val.dtype)], 
                dim=1)
        
        # Split the key and value into pages, and allocate space for the corresponding gradient.  
        gd_dtype = torch.bfloat16 if IS_BF16_ATOM_ADD_SUPPORTED else torch.float32
        if key.shape[1] > self.page_size:
            key = list(torch.chunk(key, chunks=key.shape[1] // self.page_size, dim=1))
            val = list(torch.chunk(val, chunks=val.shape[1] // self.page_size, dim=1))
        else:
            key, val = [key], [val]

        self.num_kv += update_token
        update_pages = len(key)

        # Update tensor list
        self.key_gpu.extend(key)
        self.val_gpu.extend(val)
        self.kgd_gpu.extend([torch.zeros_like(x, dtype=gd_dtype) for x in key])
        self.vgd_gpu.extend([torch.zeros_like(x, dtype=gd_dtype) for x in val])

        # Update meta data
        self.last_update_token.append(update_token)
        self.last_update_pages.append(update_pages)

        return key, val
    

class CacheManager(SimpleCacheManager):
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim, local_rank):
        super().__init__(batch_size, page_size, num_kv_heads, head_dim, local_rank)

        self.offload_future = None
        self.onload_future = None

        self.stream = Stream()
        self.pool = ThreadPoolExecutor(1)
        self.fake = torch.zeros((1,), device=self.cuda)

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
    
    def onload(self, stage):
        @torch.inference_mode()
        def worker():
            with torch.cuda.stream(self.stream):
                for i in range(len(self.key_cpu)):
                    self.key_gpu[i] = self.key_cpu[i].to(self.cuda, non_blocking=True)
                    self.val_gpu[i] = self.val_cpu[i].to(self.cuda, non_blocking=True)

                    if stage == 2:
                        self.kgd_gpu[i] = self.kgd_cpu[i].to(self.cuda, non_blocking=True)
                        self.vgd_gpu[i] = self.vgd_cpu[i].to(self.cuda, non_blocking=True)
                    else:
                        self.kgd_gpu[i] = self.fake
                        self.vgd_gpu[i] = self.fake

        if self.device != 'cuda':
            self.device = 'cuda'
            self.wait_onload()
            self.onload_future = self.pool.submit(worker)

    def offload(self, stage):
        @torch.inference_mode()
        def worker():
            with torch.cuda.stream(self.stream):
                for i in range(len(self.key_cpu)):
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


class KVCache:
    def __init__(
        self, 
        num_layers: int = 28, 
        batch_size: int = 1, 
        page_size: int = 64,
        num_heads: int = 4,
        head_dim: int = 128,
        cpu_offload=None,
        local_rank=None):

        self.num_layers = num_layers    
        self.cpu_offload = cpu_offload

        MANAGER_CLS = SimpleCacheManager if cpu_offload is None else CacheManager

        self.managers = [
            MANAGER_CLS(
                batch_size,
                page_size,
                num_heads,
                head_dim,
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
            for lid in cuda_layers:
                self.managers[lid].onload(stage)

    def length(self, layer_idx):
        return self.managers[layer_idx].num_kv

    @property
    def device(self):
        return (m.device for m in self.managers)
    
    def __getitem__(self, idx):
        return self.managers[idx]
    
    def pre_process(self):
        for idx, m in enumerate(self.managers):
            m.grad_hook = lambda idx=idx: self.visit(idx, 2, True)

    def post_process(self):
        for m in self.managers:
            m.remove_last_update()
