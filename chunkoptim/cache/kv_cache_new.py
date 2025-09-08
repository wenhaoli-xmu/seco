import torch
from torch.cuda import Stream
from ..ops.utils import IS_BF16_ATOM_ADD_SUPPORTED
from concurrent.futures import ThreadPoolExecutor


class CacheManagerSimple:
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim):
        super().__init__()
        self.batch_size = batch_size
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.reset()
        
        self.key_tensors = []
        self.val_tensors = []
        self.kgd_tensors = []
        self.vgd_tensors = []

    def reset(self):
        # Update meta data
        self.num_kv = 0
        self.last_update_token = []
        self.last_update_pages = []

        self.key_tensors = []
        self.val_tensors = []
        self.kgd_tensors = []
        self.vgd_tensors = []

        # NOTE: Called before accessing grad, can be used for debugging or CPU offloading.
        self.grad_hook = None


    @torch.inference_mode()
    def remove_last_update(self):
        if len(self.last_update_pages) == 1:
            self.reset()
            return
        
        # Update meta data
        self.last_update_token, update_token = self.last_update_token[:-1], self.last_update_token[-1]
        self.last_update_pages, update_pages = self.last_update_pages[:-1], self.last_update_pages[-1]
        self.num_kv -= update_token

        del self.key_tensors[-update_pages:]
        del self.val_tensors[-update_pages:]
        del self.kgd_tensors[-update_pages:]
        del self.vgd_tensors[-update_pages:]


    @torch.inference_mode()
    def update(self, key, val):
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
        self.key_tensors.extend(key)
        self.val_tensors.extend(val)
        self.kgd_tensors.extend([torch.zeros_like(x, dtype=gd_dtype) for x in key])
        self.vgd_tensors.extend([torch.zeros_like(x, dtype=gd_dtype) for x in val])

        # Update meta data
        self.last_update_token.append(update_token)
        self.last_update_pages.append(update_pages)

    @property
    def device(self):
        return 'cuda'
    
    @torch.inference_mode()
    def onload(self):
        ...

    @torch.inference_mode()
    def offload(self):
        ...

    @property
    @torch.inference_mode()
    def page_table(self):
        num_pages = sum(self.last_update_pages)
        assert num_pages == len(self.key_tensors)

        page_table = []
        for i in range(num_pages):
            page_table.append((
                self.key_tensors[i].data_ptr(), 
                self.val_tensors[i].data_ptr(), 
                self.kgd_tensors[i].data_ptr(), 
                self.vgd_tensors[i].data_ptr()))

        page_table = torch.tensor(
            page_table, 
            dtype=int, 
            device='cuda')

        return page_table

    @property
    @torch.inference_mode()
    def grad(self):
        if self.grad_hook is not None:
            self.grad_hook()

        num_pages = self.last_update_pages[-1]
        last_update_kgd = self.kgd_tensors[-num_pages:]
        last_update_vgd = self.vgd_tensors[-num_pages:]

        kgd = torch.cat(last_update_kgd, dim=1)[:, :self.last_update_token[-1]]
        vgd = torch.cat(last_update_vgd, dim=1)[:, :self.last_update_token[-1]]

        if not IS_BF16_ATOM_ADD_SUPPORTED:
            kgd = kgd.to(torch.bfloat16)
            vgd = vgd.to(torch.bfloat16)

        return kgd, vgd


class CacheManager:
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim):
        super().__init__()
        self.batch_size = batch_size
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.reset()
        
        self.key_tensors_cpu = []
        self.val_tensors_cpu = []
        self.kgd_tensors_cpu = []
        self.vgd_tensors_cpu = []

        self.stream = Stream()
        self.pool = ThreadPoolExecutor(1)


    def reset(self):
        # Update meta data
        self.num_kv = 0
        self.last_update_token = []
        self.last_update_pages = []

        # Update tensor list
        if self.device == 'cuda':
            self.key_tensors_gpu = []
            self.val_tensors_gpu = []
            self.kgd_tensors_gpu = []
            self.vgd_tensors_gpu = []
        self.key_tensors_cpu = []
        self.val_tensors_cpu = []
        self.kgd_tensors_cpu = []
        self.vgd_tensors_cpu = []

        # NOTE: Called before accessing grad, can be used for debugging or CPU offloading.
        self.grad_hook = None


    @torch.inference_mode()
    def remove_last_update(self):
        if len(self.last_update_pages) == 1:
            self.reset()
            return
        
        # Update meta data
        self.last_update_token, update_token = self.last_update_token[:-1], self.last_update_token[-1]
        self.last_update_pages, update_pages = self.last_update_pages[:-1], self.last_update_pages[-1]
        self.num_kv -= update_token

        # Update tensor list
        if self.device == 'cuda':
            del self.key_tensors_gpu[-update_pages:]
            del self.val_tensors_gpu[-update_pages:]
            del self.kgd_tensors_gpu[-update_pages:]
            del self.vgd_tensors_gpu[-update_pages:]
        del self.key_tensors_cpu[-update_pages:]
        del self.val_tensors_cpu[-update_pages:]
        del self.kgd_tensors_cpu[-update_pages:]
        del self.vgd_tensors_cpu[-update_pages:]


    @torch.inference_mode()
    def update(self, key, val):
        assert self.device == 'cuda'
        assert key.dtype == torch.bfloat16, 'only bfloat16 is supported'

        self.stream.synchronize()
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

        # Update tensor list
        self.key_tensors_gpu.extend(key)
        self.val_tensors_gpu.extend(val)
        self.kgd_tensors_gpu.extend([torch.zeros_like(x, dtype=gd_dtype) for x in key])
        self.vgd_tensors_gpu.extend([torch.zeros_like(x, dtype=gd_dtype) for x in val])
        self.key_tensors_cpu.extend(key_cpu)
        self.val_tensors_cpu.extend(val_cpu)
        self.kgd_tensors_cpu.extend(kgd_cpu)
        self.vgd_tensors_cpu.extend(vgd_cpu)

        # Update meta data
        self.last_update_token.append(update_token)
        self.last_update_pages.append(update_pages)

    @property
    def device(self):
        return 'cuda' if hasattr(self, 'key_tensors_gpu') else 'cpu'
    
    @torch.inference_mode()
    def onload(self):

        def worker():
            with torch.cuda.stream(self.stream):
                self.key_tensors_gpu = [x.to('cuda', non_blocking=True) for x in self.key_tensors_cpu]
                self.val_tensors_gpu = [x.to('cuda', non_blocking=True) for x in self.val_tensors_cpu]
                self.kgd_tensors_gpu = [x.to('cuda', non_blocking=True) for x in self.kgd_tensors_cpu]
                self.vgd_tensors_gpu = [x.to('cuda', non_blocking=True) for x in self.vgd_tensors_cpu]

        if self.device != 'cuda':
            self.pool.submit(worker)

    @torch.inference_mode()
    def offload(self):

        def worker():
            with torch.cuda.stream(self.stream):
                for x_gpu, x_cpu in zip(self.key_tensors_gpu, self.key_tensors_cpu):
                    x_cpu.copy_(x_gpu.data)
                del self.key_tensors_gpu
                for x_gpu, x_cpu in zip(self.val_tensors_gpu, self.val_tensors_cpu):
                    x_cpu.copy_(x_gpu.data)
                del self.val_tensors_gpu
                for x_gpu, x_cpu in zip(self.kgd_tensors_gpu, self.kgd_tensors_cpu):
                    x_cpu.copy_(x_gpu.data)
                del self.kgd_tensors_gpu
                for x_gpu, x_cpu in zip(self.vgd_tensors_gpu, self.vgd_tensors_cpu):
                    x_cpu.copy_(x_gpu.data)
                del self.vgd_tensors_gpu

        if self.device != 'cpu':
            self.pool.submit(worker)

    @property
    @torch.inference_mode()
    def page_table(self):
        num_pages = sum(self.last_update_pages)
        assert num_pages == len(self.key_tensors_cpu)

        page_table = []
        for i in range(num_pages):
            page_table.append((
                self.key_tensors_gpu[i].data_ptr(), 
                self.val_tensors_gpu[i].data_ptr(), 
                self.kgd_tensors_gpu[i].data_ptr(), 
                self.vgd_tensors_gpu[i].data_ptr()))

        page_table = torch.tensor(
            page_table, 
            dtype=int, 
            device='cuda')

        return page_table

    @property
    @torch.inference_mode()
    def grad(self):
        if self.grad_hook is not None:
            self.grad_hook()

        num_pages = self.last_update_pages[-1]
        last_update_kgd = self.kgd_tensors_gpu[-num_pages:]
        last_update_vgd = self.vgd_tensors_gpu[-num_pages:]

        kgd = torch.cat(last_update_kgd, dim=1)[:, :self.last_update_token[-1]]
        vgd = torch.cat(last_update_vgd, dim=1)[:, :self.last_update_token[-1]]

        if not IS_BF16_ATOM_ADD_SUPPORTED:
            kgd = kgd.to(torch.bfloat16)
            vgd = vgd.to(torch.bfloat16)

        return kgd, vgd


class KVCache:
    def __init__(
        self, 
        num_layers: int = 28, 
        batch_size: int = 1, 
        page_size: int = 64,
        num_heads: int = 4,
        head_dim: int = 128,
        cpu_offload=None):

        self.num_layers = num_layers    
        self.cpu_offload = cpu_offload

        MANAGER_CLS = CacheManagerSimple if cpu_offload is None else CacheManager

        self.managers = [
            MANAGER_CLS(
                batch_size,
                page_size,
                num_heads,
                head_dim)
            for _ in range(num_layers)]

    def reset(self):
        for m in self.managers:
            m.reset()

    def visit(self, layer_idx, reverse=False):
        if self.cpu_offload is not None:
            factor = -1 if reverse else 1
            cuda_layers = [
                (layer_idx + self.num_layers + factor * i) % self.num_layers 
                for i in range(self.cpu_offload)]
            cpu_layers = filter(lambda x: x not in cuda_layers, range(self.num_layers))
            for lid in cpu_layers:
                self.managers[lid].offload()
            for lid in cuda_layers:
                self.managers[lid].onload()

    @property
    def device(self):
        return (m.device for m in self.managers)
    
    def __getitem__(self, idx):
        self.visit(idx)
        return self.managers[idx]
    
    def pre_process(self):
        for idx, m in enumerate(self.managers):
            m.grad_hook = lambda idx=idx: self.visit(idx, True)

    def post_process(self):
        for m in self.managers:
            m.remove_last_update()
