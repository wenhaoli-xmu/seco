from abc import abstractmethod
import torch
from torch.cuda import Stream
from ..ops.utils import IS_BF16_ATOM_ADD_SUPPORTED
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import torch.distributed as dist

class CPUOffload:
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kwargs)

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

    @abstractmethod
    def onload(self, stage):
        raise NotImplementedError

    @abstractmethod
    def offload(self, stage):
        raise NotImplementedError
