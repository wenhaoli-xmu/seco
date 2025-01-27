import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .modifiers import get_modifier
from torch import distributed as dist

from functools import partial
import os, math

import numpy as np
import matplotlib.pyplot as plt
import time 


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

    if "tokenizer_name" in kwargs:
        tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get('tokenizer_name'), 
            use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True)

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


class SecoCache:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.reset()


    def update(self, layer_idx, keys, vals):
        try:
            ret_keys = torch.cat([*self.k_cache[layer_idx], keys], dim=-2)
            ret_vals = torch.cat([*self.v_cache[layer_idx], vals], dim=-2)
            return ret_keys, ret_vals

        finally:
            if torch.is_grad_enabled():
                self.k_cache[layer_idx].append(keys)
                self.v_cache[layer_idx].append(vals)

            else:
                k_detach, v_detach = keys.detach(), vals.detach()
                k_detach.requires_grad_(True)
                v_detach.requires_grad_(True)
                self.k_cache[layer_idx].append(k_detach)
                self.v_cache[layer_idx].append(v_detach)

    
    def length(self, layer_idx):
        return sum([x.shape[-2] for x in self.k_cache[layer_idx]])


    def reset(self):
        self.k_cache = [[] for _ in range(self.num_layers)]
        self.v_cache = [[] for _ in range(self.num_layers)]
        self.seq_len = [0 for _ in range(self.num_layers)]


    def range(self, i):
        view = SecoCache(self.num_layers)

        for layer_idx in range(view.num_layers):
            view.k_cache[layer_idx] = self.k_cache[layer_idx][:i]
            view.v_cache[layer_idx] = self.v_cache[layer_idx][:i]

        return view
    

    def index(self, i):
        view = SecoCache(self.num_layers)

        for layer_idx in range(view.num_layers):
            view.k_cache[layer_idx] = self.k_cache[layer_idx][i:i+1]
            view.v_cache[layer_idx] = self.v_cache[layer_idx][i:i+1]

        return view


    @property
    def grad(self):
        nest_gd = []
        for layer_idx in range(self.num_layers):
            nest_gd_k = [x.grad if x.grad is not None else torch.zeros_like(x) for x in self.k_cache[layer_idx]]
            nest_gd_v = [x.grad if x.grad is not None else torch.zeros_like(x) for x in self.v_cache[layer_idx]]
            nest_gd.append([nest_gd_k, nest_gd_v])
        return nest_gd
    

    def copy_scaled_grad(self, gd, scaler=None):
        def hook_fn(grad, base):
            if scaler is not None:
                return grad + base * scaler
            assert (base != 0).count_nonzero() == 0
            return grad + base

        for layer_idx in range(self.num_layers):
            layer_gd = gd[layer_idx]
            key_gd, val_gd = layer_gd
            for i in range(len(self.k_cache[layer_idx])):
                if self.k_cache[layer_idx][i].requires_grad:
                    self.k_cache[layer_idx][i].register_hook(partial(hook_fn, base=key_gd[i]))
                if self.v_cache[layer_idx][i].requires_grad:
                    self.v_cache[layer_idx][i].register_hook(partial(hook_fn, base=val_gd[i]))


    @grad.setter
    def grad(self, value):

        def hook_fn(grad, base):
            return grad + base

        for layer_idx in range(self.num_layers):
            layer_gd = value[layer_idx]
            key_gd, val_gd = layer_gd
            for i in range(len(self.k_cache[layer_idx])):
                if self.k_cache[layer_idx][i].requires_grad:
                    self.k_cache[layer_idx][i].register_hook(partial(hook_fn, base=key_gd[i]))
                if self.v_cache[layer_idx][i].requires_grad:
                    self.v_cache[layer_idx][i].register_hook(partial(hook_fn, base=val_gd[i]))



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

