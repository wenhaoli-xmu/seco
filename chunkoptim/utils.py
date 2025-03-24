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


def _reorganize_list(x, dim1, dim2):
    y = []
    for i in range(dim1):
        y.append([])
        for j in range(dim2):
            y[-1].append(x[i * dim2 + j])
    return y


class LayerCache(torch.nn.Module):
    def __init__(self, seq_dim=-2):
        super().__init__()
        self.seq_dim = seq_dim
        self.reset()

    def reset(self):
        self.keys = torch.nn.ParameterList()
        self.vals = torch.nn.ParameterList()
        self.current_device = 'cuda'
        self.key_bwd = None
        self.val_bwd = None
        self.visible_range = None

    def append(self, key, val):
        self.keys.append(torch.nn.Parameter(key, requires_grad=False))
        self.vals.append(torch.nn.Parameter(val, requires_grad=False))

    def move_to_cpu(self):
        if self.current_device != 'cpu':
            self.to('cpu', non_blocking=True)
        self.current_device = 'cpu'
    
    def move_to_cuda(self):
        if self.current_device != 'cuda':
            self.to('cuda', non_blocking=True)
        self.current_device = 'cuda'

    def length(self):
        if self.visible_range is not None:
            past_keys = self.keys[:self.visible_range]
        else:
            past_keys = self.keys
        return sum([x.shape[self.seq_dim] for x in past_keys])
    
    def gather(self):
        past_keys = self.keys if self.visible_range is None else self.keys[:self.visible_range]
        past_vals = self.vals if self.visible_range is None else self.vals[:self.visible_range]
        return (
            torch.cat([*past_keys], dim=self.seq_dim),
            torch.cat([*past_vals], dim=self.seq_dim))
    
    def delete_rear(self):
        self.keys, key = self.keys[:-1], self.keys[-1]
        self.vals, val = self.vals[:-1], self.vals[-1]
        del key, val
    
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

    def get(self, idx):
        return self.keys[idx], self.vals[idx]
    
    def get_bwd(self):
        return self.key_bwd, self.val_bwd

    def pre_reconstruction(self, idx):
        self.visible_range = idx

    def after_backward(self):
        del self.key_bwd, self.val_bwd
        self.visible_range = None
        self.key_bwd = None
        self.val_bwd = None
        self.delete_rear()


class SecoCache:
    def __init__(self, num_layers, cpu_offload=None, seq_dim=-2):
        self.num_layers = num_layers    
        self.cpu_offload = cpu_offload
        self.seq_dim = seq_dim
        self.reset()

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

    def update(self, layer_idx, key, val):
        self.visit(layer_idx)
        return self.cache[layer_idx].update(key, val)
    
    def length(self, layer_idx):
        return self.cache[layer_idx].length()
    
    def gather(self, layer_idx):
        self.visit(layer_idx)
        return self.cache[layer_idx].gather()

    def reset(self):
        if hasattr(self, 'cache'):
            del self.cache
        self.cache = [
            LayerCache(seq_dim=self.seq_dim) 
            for _ in range(self.num_layers)]

    def pre_reconstruction(self, idx):
        for c in self.cache:
            c.pre_reconstruction(idx)

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

    def after_backward(self):
        for c in self.cache:
            c.after_backward()
    

    def collect_sparse_grad(self, indices):
        num_chunks = len(self.k_cache[0])
        assert num_chunks == 1, "Sparse gradient collection does not support multiple chunks."

        grads = list(chain.from_iterable(chain.from_iterable(self.grad)))
        num_layers_times_2 = self.num_layers * 2

        _, num_heads, _, head_dim = grads[0].shape

        indices = indices[None, :, None, :, None].expand(
            self.num_layers * 2,
            -1,
            num_heads,
            -1,
            head_dim)

        sparse_grad_list = []
        for i in range(num_layers_times_2):
            grad_i = grads[i]
            index_i = indices[i]
            sparse_i = torch.gather(grad_i, dim=2, index=index_i)
            sparse_grad_list.append(sparse_i)

        sparse_gd = _reorganize_list(sparse_grad_list, dim1=num_layers_times_2, dim2=1)
        sparse_gd = _reorganize_list(sparse_gd, dim1=self.num_layers, dim2=2)

        return sparse_gd


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

        times = self.time[3:]
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

