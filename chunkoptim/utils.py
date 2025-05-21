import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .modifiers import get_modifier
from torch import distributed as dist

from functools import partial
import os, math

import numpy as np
import matplotlib.pyplot as plt
import time 

from itertools import chain
from profiler import WallTime
from abc import abstractmethod


def build_mixed_float_mask(seq_len, chunk_indices):
    """
    Arguments
    ---------
    mask_4d: [1, 1, seqlen_q, seqlen_k]
    chunk_indices: [1, chunk_size]

    Return
    ------
    mixed_mask
    """
    window_size = chunk_indices.shape[-1]
    chunk_mask_4d = torch.where(
        torch.arange(seq_len, device=chunk_indices.device)[None, None, None, :] <= chunk_indices[:, None, :, None],
        0,
        float('-inf'))
    chunk_mask_4d.scatter_(
        dim=-1, 
        index=chunk_indices[:, None, None, :].expand(-1, -1, window_size, -1), 
        value=float('-inf'))
    pad_mask_4d = torch.full(
        size=(window_size, window_size), 
        fill_value=float('-inf'),
        dtype=chunk_mask_4d.dtype, 
        device=chunk_mask_4d.device)
    pad_mask_4d = pad_mask_4d.triu(1)[None, None, :, :].expand(chunk_mask_4d.shape[0], -1, -1, -1)
    mixed_mask_4d = torch.cat([chunk_mask_4d, pad_mask_4d], dim=-1)
    return mixed_mask_4d


def average_filter(x, window):
    y = []
    w = []

    for elem in x:
        w.append(elem)
        if len(w) == window:
            y.append(sum(w) / len(w))
            w.pop(0)

    return y


def chunkize(tensor, dim, chunk_size):
    if chunk_size is None:
        chunk_size = tensor.shape[dim]

    for i in range(0, tensor.shape[dim], chunk_size):
        j = min(tensor.shape[dim], i + chunk_size)
        s = [slice(None)] * tensor.ndim
        s[dim] = slice(i, j)
        yield tensor[tuple(s)]


def colored_text(text, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def gradient_color(string, x):
    if not (0 <= x <= 1):
        raise ValueError("Input must be between 0 and 1")
    if x <= 0.5:
        ratio = x / 0.5
        r = int(0 + (255 - 0) * ratio)
        g = 255
        b = 0
    else:
        ratio = (x - 0.5) / 0.5
        r = 255
        g = int(255 - (255 - 0) * ratio)
        b = 0
    return colored_text(string, r, g, b)


def get_torch_dtype(dtype: str):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'fp32':
        return torch.float32
    elif dtype == 'bf16':
        return torch.bfloat16
    elif dtype == 'fp64':
        return torch.float64
    else:
        raise RuntimeError(f"Unknown dtype '{dtype}'")


def get_env_conf(env_conf: str):
    import json
    with open(env_conf, 'r') as f:
        env_conf = json.load(f)
    return env_conf


def get_model_and_tokenizer(
        model_name, 
        model_dtype, 
        model_method, 
        model_structure, 
        save_ckp, 
        load_ckp, 
        config, 
        device_map, 
        **kwargs
    ):

    from accelerate import dispatch_model
    token = os.environ['HF_ACCESS_TOKEN']
    tokenizer = AutoTokenizer.from_pretrained(kwargs.get('tokenizer_name', model_name))

    student_dtype = get_torch_dtype(model_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=student_dtype, 
        token=token, 
        device_map="auto" if device_map is None else None,
        trust_remote_code=True)
    modifier = get_modifier(model_method, model_structure)

    if modifier is not None:
        model = modifier(
            model,
            save_ckp=save_ckp,
            load_ckp=load_ckp,
            config=config)

    if device_map is not None:
        model.model = dispatch_model(model.model, device_map=device_map)

    return model, tokenizer


def lr_scheduler(epoch, total_epochs, warmup, plateau, max_lr, min_lr, restart=20):
    total_epochs /= restart
    epoch = epoch % total_epochs

    if epoch / total_epochs < warmup:
        partial = epoch / int(total_epochs * warmup)
        return partial * (max_lr - min_lr) + min_lr
    
    elif epoch / total_epochs < warmup + plateau:
        return max_lr
    
    else:
        epoch -= int(total_epochs * (warmup + plateau))
        total_epochs -= int(total_epochs * (warmup + plateau))
        cos_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        lr = (max_lr - min_lr) * cos_decay + min_lr
    
    return lr


def adjust_lr(optim, step, total, max_lr, min_lr, restart, warmup, plateau):
    for param_group in optim.param_groups:
        param_group['lr'] = lr_scheduler(
            step, 
            total, 
            warmup=warmup, 
            plateau=plateau, 
            max_lr=max_lr, 
            min_lr=min_lr, 
            restart=restart)


def get_optimizer_and_lr_adjuster(max_lr, train_iters, warmup, weight_decay, beta1, beta2, params, **kwargs):
    optim = torch.optim.AdamW(params, lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster


def check_not_nest(func):
    def wrapper(self, *args, **kwargs):
        assert self.is_nest is False, "this operation could not be conducted under nest mode"
        return func(self, *args, **kwargs)
    return wrapper


def check_nest(func):
    def wrapper(self, *args, **kwargs):
        assert self.is_nest is True, "this operation must be conducted under nest mode"
        return func(self, *args, **kwargs)
    return wrapper


class LayerCache(torch.nn.Module):
    def __init__(self, seq_dim):
        self.seq_dim = seq_dim
        super().__init__()
        self.reset()

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def move_to_cpu(self):
        if self.current_device != 'cpu':
            self.to('cpu', non_blocking=True)
        self.current_device = 'cpu'
    
    def move_to_cuda(self):
        if self.current_device != 'cuda':
            self.to('cuda', non_blocking=True)
        self.current_device = 'cuda'


class ListLayerCache(LayerCache):
    def delete_rear(self):
        self.keys, key = self.keys[:-1], self.keys[-1]
        self.vals, val = self.vals[:-1], self.vals[-1]
        del key, val

    def get(self, idx):
        return self.keys[idx], self.vals[idx]

    def reset(self):
        self.current_device = 'cuda'
        self.key_bwd = None
        self.val_bwd = None
        self.visible_range = None
        self.keys = torch.nn.ParameterList()
        self.vals = torch.nn.ParameterList()

    def append(self, key, val):
        self.keys.append(torch.nn.Parameter(key, requires_grad=True))
        self.vals.append(torch.nn.Parameter(val, requires_grad=True))
    
    def length(self):
        if self.visible_range is not None:
            past_keys = self.keys[:self.visible_range]
        else:
            past_keys = self.keys
        return sum([x.shape[self.seq_dim] for x in past_keys])
    
    def gather(self):
        past_keys = self.keys if self.visible_range is None else self.keys[:self.visible_range]
        past_vals = self.vals if self.visible_range is None else self.vals[:self.visible_range]
        past_keys += (self.key_bwd,)
        past_vals += (self.val_bwd,)
        return (
            torch.cat([*past_keys], dim=self.seq_dim),
            torch.cat([*past_vals], dim=self.seq_dim))

    def update(self, key, val):
        try:
            past_keys = self.keys if self.visible_range is None else self.keys[:self.visible_range]
            past_vals = self.vals if self.visible_range is None else self.vals[:self.visible_range]
            ret_keys = torch.cat([*past_keys, key], dim=self.seq_dim)
            ret_vals = torch.cat([*past_vals, val], dim=self.seq_dim)
            return ret_keys, ret_vals
        finally:
            if not torch.is_grad_enabled():
                self.append(key, val)
            else:
                self.key_bwd = key
                self.val_bwd = val
    
    def get_bwd(self):
        return self.key_bwd, self.val_bwd
    
    def pre_recon(self, idx):
        self.visible_range = idx

    def after_backward(self):
        del self.key_bwd, self.val_bwd
        self.visible_range = None
        self.key_bwd = None
        self.val_bwd = None
        self.delete_rear()


class NestLayerCache(LayerCache):
    def reset(self):
        self.current_device = 'cuda'
        self.key_bwd = None
        self.val_bwd = None
        self.keys = None
        self.vals = None

    def append(self, key, val):
        if self.keys is None:
            self.keys = torch.nn.Parameter(key, requires_grad=True)
            self.vals = torch.nn.Parameter(val, requires_grad=True)
        else:
            self.keys.data = torch.cat((self.keys.data, key.data), dim=self.seq_dim)
            self.vals.data = torch.cat((self.vals.data, val.data), dim=self.seq_dim)

    def length(self):
        return self.keys.shape[self.seq_dim] if self.keys is not None else 0
    
    def get(self):
        return self.keys, self.vals
    
    def gather(self):
        return (
            torch.cat((self.keys, self.key_bwd), dim=self.seq_dim),
            torch.cat((self.vals, self.val_bwd), dim=self.seq_dim))

    def update(self, key, val):
        try:
            if self.keys is not None:
                ret_keys = torch.cat([self.keys, key], dim=self.seq_dim)
                ret_vals = torch.cat([self.vals, val], dim=self.seq_dim)
            else:
                ret_keys = key
                ret_vals = val
            return ret_keys, ret_vals
        finally:
            if not torch.is_grad_enabled():
                self.append(key, val)
            else:
                self.key_bwd = key
                self.val_bwd = val

    def get_bwd(self):
        return self.key_bwd, self.val_bwd

    def after_backward(self):
        del self.key_bwd, self.val_bwd
        self.key_bwd = None
        self.val_bwd = None


class KVCache:
    def __init__(self, num_layers, cpu_offload=None, seq_dim=-2):
        self.num_layers = num_layers    
        self.cpu_offload = cpu_offload
        self.seq_dim = seq_dim
        self.reset()

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def visit(self, layer_idx, reverse=False):
        if self.cpu_offload is not None:
            factor = -1 if reverse else 1
            cuda_layers = [
                (layer_idx + self.num_layers + factor * i) % self.num_layers 
                for i in range(self.cpu_offload)]
            cpu_layers = filter(lambda x: x not in cuda_layers, range(self.num_layers))
            for lid in cpu_layers:
                self.cache[lid].move_to_cpu()
            for lid in cuda_layers:
                self.cache[lid].move_to_cuda()

    @property
    def device(self):
        d = []
        for c in self.cache:
            d.append(next(c.parameters()).device)
        return d

    def update(self, layer_idx, key, val):
        self.visit(layer_idx)
        return self.cache[layer_idx].update(key, val)
    
    def length(self, layer_idx):
        return self.cache[layer_idx].length()
    
    def gather(self, layer_idx):
        self.visit(layer_idx)
        return self.cache[layer_idx].gather()


class ListCache(KVCache):
    def reset(self):
        if hasattr(self, 'cache'):
            del self.cache
        self.cache = [
            ListLayerCache(seq_dim=self.seq_dim) 
            for _ in range(self.num_layers)]

    def additive_hook(self, grad, base, layer_idx):
        self.visit(layer_idx, reverse=True)
        if base.grad is not None:
            return grad + base.grad
        return grad

    def pre_backward(self, idx):
        for layer_idx, c in enumerate(self.cache):
            key, val = c.get(idx)
            key_bwd, val_bwd = c.get_bwd()
            key_bwd.register_hook(partial(self.additive_hook, base=key, layer_idx=layer_idx))
            val_bwd.register_hook(partial(self.additive_hook, base=val, layer_idx=layer_idx))

    def pre_recon(self, idx):
        for c in self.cache:
            c.pre_recon(idx)

    def after_backward(self):
        for c in self.cache:
            c.after_backward()


class NestCache(KVCache):
    def reset(self):
        if hasattr(self, 'cache'):
            del self.cache
        self.cache = [
            NestLayerCache(seq_dim=self.seq_dim) 
            for _ in range(self.num_layers)]
    
    def additive_hook(self, grad, base, layer_idx, indices):
        self.visit(layer_idx, reverse=True)
        if base.grad is not None:
            if self.seq_dim == 1:
                indices = indices[:, :, None, None]
                indices = indices.expand(-1, -1, grad.shape[2], grad.shape[3])
            elif self.seq_dim == 2:
                indices = indices[:, None, :, None]
                indices = indices.expand(-1, grad.shape[1], -1, grad.shape[3])
            else:
                raise NotImplementedError
            delta = torch.gather(base.grad.data, dim=self.seq_dim, index=indices)
            return grad + delta
        return grad

    def pre_backward(self, indices):
        for layer_idx, c in enumerate(self.cache):
            keys, vals = c.get()
            key_bwd, val_bwd = c.get_bwd()
            key_bwd.register_hook(partial(self.additive_hook, base=keys, layer_idx=layer_idx, indices=indices))
            val_bwd.register_hook(partial(self.additive_hook, base=vals, layer_idx=layer_idx, indices=indices))

    def after_backward(self):
        for c in self.cache:
            c.after_backward()


def average_filter(x, window):
    y = []
    w = []

    for elem in x:
        w.append(elem)
        if len(w) == window:
            y.append(sum(w) / len(w))
            w.pop(0)

    return y


class History:
    def __init__(self, log_step=1):
        self.loss = []
        self.time = []
        self.seq_len = []
        self.memory = []
        self._step = 0
        self._log_step = log_step
        self._start = time.time()

        self.path = "history-{step}.jpg"
        self.template = "step-{step:<5d} | loss: {loss:.3f} | avg time: {time:.3f} | max memory: {memory:.3f} | seq len: {seq_len:>10d}"

    def init(self):
        self._start = time.time()


    def step(self, loss, seq_len):
        interval = time.time() - self._start

        self.loss.append(loss)
        self.time.append(interval)
        self.seq_len.append(seq_len)

        self.memory.append(torch.cuda.max_memory_allocated())
        self._step += 1

        if self._step % self._log_step == 0:
            self.summary()


    def summary(self, pr1nt=True):

        times = self.time
        min_time = min(times) if len(times) > 0 else 0.0
        min_memory = min(self.memory) / 1024 ** 2

        if dist.get_rank() == 0 and pr1nt:
            # plt.figure()
            # plt.subplot(131)
            # plt.title("loss")
            # plt.plot(self.loss)

            # plt.subplot(132)
            # plt.title(f"time, avg-{np.mean(self.time): .3f}")
            # plt.plot(self.time)

            # plt.subplot(133)
            # plt.title(f"max mem, avg-{np.mean(self.memory) // 1024 ** 2: .3f}")
            # plt.plot(self.memory)

            # plt.savefig(self.path.format(step=self._step))

            print(self.template.format(
                step=self._step,
                loss=self.loss[-1],
                time=min_time,
                memory=min_memory,
                seq_len=int(np.mean(self.seq_len))))

        dist.barrier()

        return min_time, min_memory


