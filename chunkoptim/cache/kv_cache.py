import torch
from torch.cuda import Stream
from functools import partial
from ..ops.utils import IS_BF16_ATOM_ADD_SUPPORTED


class CacheManager(torch.nn.Module):
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim):
        super().__init__()
        self.batch_size = batch_size
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.reset()


    def reset(self):
        self.num_kv = 0
        self.last_update_token = []
        self.last_update_pages = []

        # NOTE: We use ParameterList to save the buffer so that they can be used as part of 
        # the module and conveniently offload the CPU using the to(device) method.
        self.key_tensors = torch.nn.ParameterDict()
        self.val_tensors = torch.nn.ParameterDict()
        self.kgd_tensors = torch.nn.ParameterDict()
        self.vgd_tensors = torch.nn.ParameterDict()

        # NOTE: Called before accessing grad, can be used for debugging or CPU offloading.
        self.grad_hook = None


    @torch.inference_mode()
    def remove_last_update(self):
        if len(self.last_update_pages) == 1:
            self.reset()
            return
        
        # TODO: ParameterList does not implement the __delitem__ method, so this is a workaround.
        self.last_update_token, update_token = self.last_update_token[:-1], self.last_update_token[-1]
        self.last_update_pages, update_pages = self.last_update_pages[:-1], self.last_update_pages[-1]
        self.num_kv -= update_token

        page_indicies = range(
            sum(self.last_update_pages), 
            sum(self.last_update_pages) + update_pages)

        for page_idx in page_indicies:
            page_idx = str(page_idx)
            del self.key_tensors[page_idx]
            del self.val_tensors[page_idx]
            del self.kgd_tensors[page_idx]
            del self.vgd_tensors[page_idx]


    @torch.inference_mode()
    def update(self, key, val, as_buffer=True):
        assert key.dtype == torch.bfloat16, 'only bfloat16 is supported'
        update_token = key.shape[1]

        # pad key and value
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
        kgd = torch.zeros_like(key, dtype=gd_dtype)
        vgd = torch.zeros_like(val, dtype=gd_dtype)
        if key.shape[1] > self.page_size:
            key = list(torch.chunk(key, chunks=key.shape[1] // self.page_size, dim=1))
            val = list(torch.chunk(val, chunks=val.shape[1] // self.page_size, dim=1))
            kgd = list(torch.chunk(kgd, chunks=kgd.shape[1] // self.page_size, dim=1))
            vgd = list(torch.chunk(vgd, chunks=vgd.shape[1] // self.page_size, dim=1))
        else:
            key, val, kgd, vgd = [key], [val], [kgd], [vgd]


        self.num_kv += update_token
        update_pages = len(key)

        if as_buffer:
            key_tensors = {
                str(sum(self.last_update_pages) + i): torch.nn.Buffer(x.data)
                for i, x in enumerate(key)}
            
            val_tensors = {
                str(sum(self.last_update_pages) + i): torch.nn.Buffer(x.data) 
                for i, x in enumerate(val)}
            
            kgd_tensors = {
                str(sum(self.last_update_pages) + i): torch.nn.Buffer(x.data) 
                for i, x in enumerate(kgd)}
            
            vgd_tensors = {
                str(sum(self.last_update_pages) + i): torch.nn.Buffer(x.data) 
                for i, x in enumerate(vgd)}
        else:
            key_tensors = {
                str(sum(self.last_update_pages) + i): x.data
                for i, x in enumerate(key)}
            
            val_tensors = {
                str(sum(self.last_update_pages) + i): x.data
                for i, x in enumerate(val)}
            
            kgd_tensors = {
                str(sum(self.last_update_pages) + i): x.data
                for i, x in enumerate(kgd)}
            
            vgd_tensors = {
                str(sum(self.last_update_pages) + i): x.data
                for i, x in enumerate(vgd)}
        
        self.key_tensors.update(key_tensors)
        self.val_tensors.update(val_tensors)
        self.kgd_tensors.update(kgd_tensors)
        self.vgd_tensors.update(vgd_tensors)

        self.last_update_token.append(update_token)
        self.last_update_pages.append(update_pages)

    @property
    @torch.inference_mode()
    def page_table(self):
        num_pages = sum(self.last_update_pages)
        assert num_pages == len(self.key_tensors)

        page_table = []
        for i in range(num_pages):
            page_table.append((
                self.key_tensors[str(i)].data_ptr(), 
                self.val_tensors[str(i)].data_ptr(), 
                self.kgd_tensors[str(i)].data_ptr(), 
                self.vgd_tensors[str(i)].data_ptr()))

        page_table = torch.tensor(
            page_table, 
            dtype=int, 
            device='cuda')

        return page_table
    
    @property
    @torch.inference_mode()
    def keys(self):
        page_indicies = range(sum(self.last_update_pages))
        ret = []
        for page_idx in page_indicies:
            page_idx = f"{page_idx}"
            ret.append(self.key_tensors[page_idx])
        return ret
    
    @property
    @torch.inference_mode()
    def values(self):
        page_indicies = range(sum(self.last_update_pages))
        ret = []
        for page_idx in page_indicies:
            page_idx = f"{page_idx}"
            ret.append(self.val_tensors[page_idx])
        return ret

    @property
    @torch.inference_mode()
    def grad(self):
        if self.grad_hook is not None:
            self.grad_hook()

        page_indicies = range(
            sum(self.last_update_pages[:-1]), 
            sum(self.last_update_pages))
        page_indices = [str(page_idx) for page_idx in page_indicies]
        last_update_kgd = [self.kgd_tensors[page_idx] for page_idx in page_indices]
        last_update_vgd = [self.vgd_tensors[page_idx] for page_idx in page_indices]

        kgd = torch.cat(last_update_kgd, dim=1)[:, :self.last_update_token[-1]]
        vgd = torch.cat(last_update_vgd, dim=1)[:, :self.last_update_token[-1]]

        if not IS_BF16_ATOM_ADD_SUPPORTED:
            kgd = kgd.to(torch.bfloat16)
            vgd = vgd.to(torch.bfloat16)

        return kgd, vgd


class LayerCache(torch.nn.Module):
    def __init__(self, batch_size, page_size, num_heads, head_dim):
        super().__init__()
        self.manager = CacheManager(
            batch_size=batch_size, 
            page_size=page_size, 
            num_kv_heads=num_heads, 
            head_dim=head_dim)
        self.reset()

    def reset(self):
        self.current_device = 'cuda'
        self.manager.reset()

    def move_to_cpu(self, stream): 
        if self.current_device != 'cpu':
            with torch.cuda.stream(stream):
                self.to('cpu', non_blocking=True)
        self.current_device = 'cpu'

    def move_to_cuda(self, stream):
        if self.current_device != 'cuda':
            if self.length() > 0:
                assert next(self.parameters()).is_pinned()
            with torch.cuda.stream(stream):
                self.to('cuda', non_blocking=True)
        self.current_device = 'cuda'

    def length(self):
        return self.manager.num_kv


class KVCache:
    def __init__(
        self, 
        num_layers: int = 28, 
        batch_size: int = 1, 
        page_size: int = 64,
        num_heads: int = 4,
        head_dim: int = 128,
        cpu_offload=None,
        num_streams: int = 4):
    
        self.num_layers = num_layers    
        self.cpu_offload = cpu_offload

        self.streams = [Stream() for _ in range(num_streams)]
        self.stream_idx = 0

        self.cache = [
            LayerCache(
                batch_size,
                page_size,
                num_heads,
                head_dim)
            for _ in range(num_layers)]

    def reset(self):
        for c in self.cache:
            c.reset()

    def visit(self, layer_idx, reverse=False):

        if self.cpu_offload is not None:
            factor = -1 if reverse else 1
            cuda_layers = [
                (layer_idx + self.num_layers + factor * i) % self.num_layers 
                for i in range(self.cpu_offload)]
            cpu_layers = filter(lambda x: x not in cuda_layers, range(self.num_layers))

            for lid in cpu_layers:
                stream = self.streams[self.stream_idx % len(self.streams)]
                self.cache[lid].move_to_cpu(stream)
                self.stream_idx += 1
            for lid in cuda_layers:
                stream = self.streams[self.stream_idx % len(self.streams)]
                self.cache[lid].move_to_cuda(stream)
                self.stream_idx += 1

    @property
    def device(self):
        d = []
        for c in self.cache:
            d.append(next(c.parameters()).device)
        return d
    
    def __getitem__(self, idx):
        self.visit(idx)
        return self.cache[idx].manager
    
    def length(self, layer_idx):
        return self.cache[layer_idx].length()
    
    def pre_process(self):
        for idx, c in enumerate(self.cache):
            c.manager.grad_hook = lambda idx=idx: self.visit(idx, True)

    def post_process(self):
        for c in self.cache:
            c.manager.remove_last_update()
