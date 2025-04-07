![img](docs/main.png)

[Download Paper](https://github.com/wenhaoli-xmu/seco/raw/main/320.pdf)

## Multi-Model Instruction Tuning (based on LLaVA-HR)

This training is running on 8 * A100 GPUs, using Deepspeed zero2 optimization. The base model is LLaMA3-8B and we full finetune its parameters on the official instruct tuning dataset of LLaVA-HR.

![img](mllm.png)
![img](docs/timecost.png)

We used our new implementation, which supports kv cache offloading, to run this experiments. The core codes are listed as follows:

<details>
<summary>SecoCache</summary>
<code>
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
</code>
</details>

<details>
<summary>The new training step function</summary>
<code>
def _seco(self, model, inputs):
    fwd_chunk_size = 512
    bwd_chunk_size = 512
    valid_label_count = (inputs['labels'] != -100).sum()

    # prepare inputs
    attention_mask, inputs_embeds, labels, _ = _prepare_inputs(model, inputs)

    # align length across gpus
    inputs_embeds, labels, attention_mask, _ = _maybe_align_length_across_gpus(
        inputs_embeds, 
        labels, 
        attention_mask,
        None)

    inputs_embeds_detach = inputs_embeds.detach()
    inputs_embeds_detach.requires_grad_(True)
    labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)

    # chunkize inputs
    inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
        [inputs_embeds_detach, labels, attention_mask],
        chunk_size=fwd_chunk_size)

    # LLM forward prop
    accum_loss, seco_cache = _first_forward_prop_seco(
        model, 
        inputs_embeds_list, 
        labels_list, 
        masks_list, 
        valid_label_count,
        cpu_offload=1)

    # maybe change chunk size before backward prop
    if fwd_chunk_size != bwd_chunk_size:
        seco_cache.reorganize(bwd_chunk_size)
        inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
            [inputs_embeds_detach, labels, attention_mask],
            chunk_size=bwd_chunk_size,)

    # LLM backward prop
    generator = reversed(list(enumerate(zip(inputs_embeds_list, labels_list))))
    for i, (chunk_embeds, chunk_labels) in generator:
        tmp_cache = seco_cache.index(i)
        seco_cache.delete(i)
        outputs = model(
            input_ids=None,
            attention_mask=torch.cat(masks_list[:i+1], dim=1),
            inputs_embeds=chunk_embeds,
            labels=chunk_labels,
            past_key_values=seco_cache,
            shift_label=False,
            is_reduce=False)
        loss = outputs['loss'].sum() / valid_label_count
        seco_cache.link_grad(tmp_cache, i)
        _backward(loss, model)
        seco_cache.delete(i)
        del tmp_cache

    inputs_embeds.register_hook(partial(
        _set_to_incomming_grad, 
        incomming_grad=inputs_embeds_detach.grad))
    _backward(inputs_embeds.sum(), model)
    _step(model, self.optimizer)

    return accum_loss / self.args.gradient_accumulation_steps
</code>
</details>


<details>
<summary>Seco profiling tools</summary>
<code>
def _profile_seco(self, model, inputs):
    from pygments.console import colorize
    from profiler import WallTime

    fwd_chunk_size = 512
    bwd_chunk_size = 512
    context_length = 6144
    valid_label_count = (inputs['labels'] != -100).sum()

    if dist.is_initialized():
        gpu_device = dist.get_rank()
    else:
        gpu_device = 0
    
    t0 = WallTime("end to end time", gpu_device)
    t1 = WallTime("vision tower fwd", gpu_device)
    t2 = WallTime("forward prop", gpu_device)
    t3 = WallTime("reorganize time", gpu_device)
    t4 = WallTime("backward prop", gpu_device)
    t5 = WallTime("vision tower bwd", gpu_device)

    while inputs['input_ids'].shape[-1] < context_length:
        inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.full_like(inputs['input_ids'], fill_value=self.tokenizer.pad_token_id)], dim=-1)
        inputs['labels'] = torch.cat([inputs['labels'], torch.full_like(inputs['labels'], fill_value=-100)], dim=-1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.full_like(inputs['attention_mask'], fill_value=0)], dim=-1)
    inputs['input_ids'] = inputs['input_ids'][:, :context_length]
    inputs['labels'] = inputs['labels'][:, :context_length]
    inputs['attention_mask'] = inputs['attention_mask'][:, :context_length]

    for _ in range(3):
        with t0:
            with t1:
                attention_mask, inputs_embeds, labels, _ = _prepare_inputs(model, inputs)

            # align length across gpus
            inputs_embeds, labels, attention_mask, _ = _maybe_align_length_across_gpus(
                inputs_embeds, 
                labels, 
                attention_mask,
                None)
            
            inputs_embeds_detach = inputs_embeds.detach()
            inputs_embeds_detach.requires_grad_(True)
            labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], fill_value=-100)), dim=-1)

            # chunkize inputs
            inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
                [inputs_embeds_detach, labels, attention_mask],
                chunk_size=fwd_chunk_size)

            with t2:
                accum_loss, seco_cache = _first_forward_prop_seco(
                    model, 
                    inputs_embeds_list, 
                    labels_list, 
                    masks_list, 
                    valid_label_count,
                    cpu_offload=1)

            with t3:
                if fwd_chunk_size != bwd_chunk_size:
                    seco_cache.reorganize(bwd_chunk_size)
                    inputs_embeds_list, labels_list, masks_list = _chunkize_inputs(
                        [inputs_embeds_detach, labels, attention_mask],
                        chunk_size=bwd_chunk_size)

            with t4:
                generator = reversed(list(enumerate(zip(inputs_embeds_list, labels_list))))
                for i, (chunk_embeds, chunk_labels) in generator:
                    
                    # reconstruction
                    seco_cache.pre_reconstruction(i)
                    outputs = model(
                        input_ids=None,
                        attention_mask=torch.cat(masks_list[:i+1], dim=1),
                        inputs_embeds=chunk_embeds,
                        labels=chunk_labels,
                        past_key_values=seco_cache,
                        shift_label=False,
                        is_reduce=False)
                    loss = outputs['loss'].sum() / valid_label_count

                    # backward propagation
                    seco_cache.pre_backward(i)
                    _backward(loss, model) 
                    seco_cache.after_backward()

            with t5:
                inputs_embeds.register_hook(partial(
                    _set_to_incomming_grad, 
                    incomming_grad=inputs_embeds_detach.grad))
                _backward(inputs_embeds.sum(), model)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # output key information
        t0.result(detail=True)
        t1.result(detail=True)
        t2.result(detail=True)
        t3.result(detail=True)
        t4.result(detail=True)
        t5.result(detail=True)

        # compute memory allocation
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += param.data.numel()
            if param.requires_grad:
                grad_count += param.data.numel()
        param_memory = f"{param_count * 2 / 1024 ** 3: .1f}"
        grad_memory = f"{grad_count * 2 / 1024 ** 3: .1f}"

        memory_info = {
            "cur mem alloc": f"{torch.cuda.memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "max mem alloc": f"{torch.cuda.max_memory_allocated(gpu_device) / 1024 ** 3: .1f}",
            "parameters": param_memory,
            "gradients": grad_memory
        }
        print('=' * 10)

        for key, value in memory_info.items():
            key = colorize("green", key)
            value = colorize("yellow", value)
            print(f"{key}:\t{value}")

        import IPython
        IPython.embed(header=f"fwd: {fwd_chunk_size}, bwd: {bwd_chunk_size}, len: {inputs_embeds.shape[-2]}")

    if dist.is_initialized():
        dist.barrier()

    raise NotImplementedError
</code>
</details>

## 🤖Inst-Tuning Results

To more comprehensively validate the performance of SeCO and SpaCO, we further compared them with model parallel (running on 4 RTX 3090 GPUs, utilizing gradient checkpointing) in the instruction fine-tuning task. The results are shown in the figure below:

![img](docs/longbench.png)

## 👁️Overview

We propose SeCO and SpaCO for training LLMs under memory-constrained scenarios.

### Sequential Chunk-wise Optimization (SeCO)

* Employs a step-by-step strategy to execute forward propagation and localized backward propagation in chunks, with only one computational graph stored in GPU memory at any given time.

* Enables exact gradient computation, achieving gradient accuracy up to **12 decimal places** when using fp64 precision.

* Maintains near-native training speed when the chunk size is efficiently large, with no significant slowdown compared to conventional gradient checkpointing.

### Sparse Chunk-wise Optimization (SpaCO)

* Extends SeCO by introducing sparsification during backward propagation.

* Gradually aligns training costs with inference costs as context length increases.

* While unable to compute exact gradients, the resulting trade-offs remain practically acceptable for most applications.

Compared to mainstream training approaches, SeCO and SpaCO demonstrate substantial efficiency advantages:

![img](docs/efficiency.png)


## 🚀Quick Start


### Installation

```bash

$ git clone https://github.com/wenhaoli-xmu/seco.git
$ cd seco
$ pip install -e .
$ pip install -r requirements.txt
```

### Example-1: Single GPU

```python
from chunkoptim.utils import chunkize, SecoCache


chunk_size = 128


for batch in data_loader:
    
    input_ids = list(chunkize(batch.input_ids, -1, chunk_size))
    labels = list(chunkize(batch.labels, -1, chunk_size))
    kv_cache = SecoCache(model.num_layers)

    # forward prop
    with torch.no_grad():
        for chunk_input, chunk_target in zip(input_ids, labels):
            inputs = dict(
                input_ids=chunk_input,
                labels=chunk_target,
                kv_cache=kv_cache)
            model(**inputs)

    accum_loss = 0

    gen = reversed(list(enumerate(zip(input_ids, labels))))

    for i, (chunk_input, chunk_target) in gen:

        tmp_kv_cache = kv_cache.range(i)

        # graph reconstruction
        inputs = dict(
            input_ids=chunk_input,
            labels=chunk_target,
            kv_cache=tmp_kv_cache)

        loss = model(**inputs).sum() / batch['seq_len']
        accum_loss += loss.item()


        # localized backward prop
        tmp_kv_cache.index(i).copy_scaled_grad(gd=kv_cache.index(i).grad)
        loss.backward()

    optim.step()
    optim.zero_grad()
```

### Example-2: Multiple GPUs (Deepspeed)

⚠️If you use Deepspeed for distributed training, the gradient won't synchronize until the model_engine.step() function call, which meets the requirements of SeCO. Otherwise, you should cancel gradient synchronization manually to prevent redundant communication.

```python
from chunkoptim.utils import chunkize, SecoCache
chunk_size = 128

for batch in data_loader:
    input_ids = list(chunkize(batch.input_ids, -1, chunk_size))
    labels = list(chunkize(batch.labels, -1, chunk_size))
    kv_cache = SecoCache(model.num_layers)

    # forward prop
    with torch.no_grad():
        for chunk_input, chunk_target in zip(input_ids, labels):
            inputs = dict(
                input_ids=chunk_input,
                labels=chunk_target,
                kv_cache=kv_cache)
            model_engine(**inputs)

    accum_loss = 0

    gen = reversed(list(enumerate(zip(input_ids, labels))))

    for i, (chunk_input, chunk_target) in gen:

        tmp_kv_cache = kv_cache.range(i)

        # graph reconstruction
        inputs = dict(
            input_ids=chunk_input,
            labels=chunk_target,
            kv_cache=tmp_kv_cache)

        loss = model_engine(**inputs).sum() / batch['seq_len']
        accum_loss += loss.item()

        # localized backward prop
        tmp_kv_cache.index(i).copy_scaled_grad(gd=kv_cache.index(i).grad)
        model_engine.backward(loss)

    model_engine.step()
```

