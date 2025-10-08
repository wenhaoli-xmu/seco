# Out of the Memory Barrier: A Highly Memory-Efficient Training System for LLMs with Million-Token Contexts

## Foreword

**⚠️ This document is for anonymous review. For reproduction and standard documentation, please see [docs/README.md](docs/README.md).**

There are two ways to review the code:

* Click the links to view the code directly (not recommended).

* Click the triangle to expand and review the code in-place (recommended, as comments are more detailed).

We divide the code into two main parts. The first is the **User Interface**, which is the user-facing part of the system. The second is the **Core Modules**, which showcases the internal components that form the system's fundamental functionality.

## User Interface
### 1. Top-Level Pipeline Code
<table style="width:100%; border: none;">
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>A. Parallel Training</b> (<a href="test_efficiency/test.py#L155" target="_blank">test_efficiency/test.py:155</a>)</summary>

```python
def baseline_tensor_parallel(model, batch, grad_ckpt):

    # Forward pass
    loss = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        kv_cache=None, # chunk-wise training is not used, so no KVCache is passed.
        grad_ckpt=grad_ckpt).sum() / batch['seq_len']

    # Backward pass
    loss.backward()
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>B. Chunk-wise Training</b> (<a href="test_efficiency/test.py#L271" target="_blank">test_efficiency/test.py:271</a>)</summary>

```python
def blockwise_tensor_parallel(model, batch, grad_ckpt, block_size, page_size, cpu_offload):

    # First, segment `input_ids` and `labels` into equal-length chunks.
    my_chunkize = partial(chunkize, dim=-1, chunk_size=block_size)
    input_ids = list(my_chunkize(batch['input_ids']))
    labels = list(my_chunkize(batch['labels']))

    # Initialize KVCache and configure its parameters.
    from chunkoptim.cache.kv_cache import KVCache
    kv_cache = KVCache(
        num_layers=model.model.config.num_hidden_layers,
        batch_size=1, # Chunk-wise training supports batch size > 1.
        page_size=page_size, # `page_size` is the same as TILE_SIZE. Recommended: 64 for A100, 128 for H200.
        num_heads=model.model.config.num_key_value_heads // dist.get_world_size(), # Divide by device count for tensor parallelism.
        cpu_offload=cpu_offload)

    """
    1. First forward pass to populate the KV cache.
    The goal is to populate the entire KV cache. Therefore, activations and gradients 
    are not needed, and this is done within a `no_grad` context. The chunk size here 
    can differ from the one used in the backward pass; it can even be done in parallel 
    in a single pass, as long as all KV values for all tokens are computed.

    This step does introduce overhead. However, when gradient checkpointing is enabled 
    (as is typical), a standard forward pass is also executed twice. Thus, our method 
    is not less efficient.
    """
    with torch.no_grad():

        for chunk_input, chunk_target in zip(input_ids, labels):

            # forward pass
            inputs = dict(
                input_ids=chunk_input,
                labels=chunk_target,
                kv_cache=kv_cache,
                grad_ckpt=False)
            model(**inputs)

    """
    2. Gradient Calculation.
    Compute gradients in reverse order. For each chunk, perform a forward pass 
    followed by a backward pass for local gradient computation.
    """
    for chunk_input, chunk_target in reversed(list(zip(input_ids, labels))):

        # forward prop
        inputs = dict(
            input_ids=chunk_input,
            labels=chunk_target,
            kv_cache=kv_cache,
            grad_ckpt=grad_ckpt)
        loss = model(**inputs).sum() / batch['seq_len']

        # `pre_process` attaches gradient hooks to all layers to enable 
        # prefetching of KV cache values and their gradients.
        kv_cache.pre_process()
        
        # Our code is also compatible with the DeepSpeed framework.
        # Simply run `model_engine.backward(loss)`.
        loss.backward()

        # `post_process` evicts KV cache entries for which computation is complete 
        # and asynchronously writes back their gradients.
        kv_cache.post_process()
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>C. Chunk-wise + Sparse Training</b> (<a href="test_efficiency/test.py#L164" target="_blank">test_efficiency/test.py:164</a>)</summary>

```python
"""
This code is largely identical to B. Chunk-wise Training, with the main 
difference being the instantiation of `SparseKVCache`. Please review the 
comments in section B before proceeding.
"""

def blockwise_tensor_parallel_sparse(model, batch, grad_ckpt, block_size, page_size, cpu_offload, page_budget):
    my_chunkize = partial(chunkize, dim=-1, chunk_size=block_size)
    input_ids = list(my_chunkize(batch['input_ids']))
    labels = list(my_chunkize(batch['labels']))

    from chunkoptim.cache.topk_cache import SparseKVCache

    world_size = dist.get_world_size()

    """
    The `KVCache` from section B is replaced with `SparseKVCache`; other parts remain the same.
    An additional parameter, `page_budget`, is introduced. The total retrieval budget is 
    `page_budget * page_size`, and the total number of tokens involved in the computation 
    is <= retrieval budget + chunk size.
    """
    kv_cache = SparseKVCache(
        num_layers=model.model.config.num_hidden_layers,
        batch_size=1,
        page_size=page_size,
        num_heads=model.model.config.num_key_value_heads // world_size,
        cpu_offload=cpu_offload,
        page_budget=page_budget)

    with torch.no_grad():
        for chunk_input, chunk_target in zip(input_ids, labels):

# input_ids: [bsz, seq_len]
# labels: [bsz, seq_len]

input_ids_chunks = list(my_chunkize(input_ids))
labels_chunks = list(my_chunkize(labels))
```

</details>
</td>
</tr>
</table>

### 2. Inner Attention Layer Code
<table style="width:100%; border: none;">
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>A. Parallel Training</b> (<a href="chunkoptim/modifiers/train_baseline_tp.py#L134" target="_blank">chunkoptim/modifiers/train_baseline_tp.py:134</a>)</summary>

```python
"""
Apart from the tensor parallel support, the logic in this section 
is consistent with the 🤗HuggingFace implementation.
"""
def self_attn_forward(self, hidden_states, kv_cache):
    world_size = get_tensor_parallel_world_size()
    
    # Dividing by `world_size` here is necessary for tensor parallelism.
    num_heads = self.config.num_attention_heads // world_size
    num_kv_heads = self.config.num_key_value_heads // world_size
    embed_dim = self.config.hidden_size
    head_dim = embed_dim // self.config.num_attention_heads
    
    # query & key & value projection
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim, head_first=False)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, head_first=False)

    # position embedding
    pos = torch.arange(0, keys.shape[1])
    pos = pos[None, :].to(keys.device)
    cos, sin = self.rotary_emb(keys, pos)
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    # Here, we make a standard call to the FlashAttention interface.
    attn_output = flash_attn_func(ques, keys, vals, is_causal=True)

    attn_output = attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>B. Chunk-wise Training</b> (<a href="chunkoptim/modifiers/train_blockwise_tp.py#L109" target="_blank">chunkoptim/modifiers/train_blockwise_tp.py:109</a>)</summary>

```python
"""
As explained in the pipeline section, there are two forward passes:
* Stage 1 (in a `no_grad` context) to populate the KV cache.
* Stage 2 (with gradients enabled) for computation.
Prefetching, offloading, and KV cache update strategies differ slightly 
between these stages. Therefore, a `stage` parameter must be passed to two key interfaces:
* kv_cache.visit(layer_idx: int, stage: int)
* kv_cache.update(key: Tensor, value: Tensor, stage: int)

Code wrapped in `===` highlights the main modifications; other parts are 
similar to the baseline.
"""
def self_attn_forward(self, hidden_states, kv_cache):
    # =========================================
    # Main Change 1
    stage = 2 if torch.is_grad_enabled() else 1
    kv_cache.visit(self.layer_idx, stage)
    # =========================================

    world_size = get_tensor_parallel_world_size()
    
    num_heads = self.config.num_attention_heads // world_size
    num_kv_heads = self.config.num_key_value_heads // world_size
    embed_dim = self.config.hidden_size
    head_dim = embed_dim // self.config.num_attention_heads

    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim, head_first=False)
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, head_first=False)

    # ===========================================
    # Main Change 2
    # In Stage 2, `num_kv` includes the length of the current chunk's KV cache, 
    # so it must be subtracted to get the true past KV cache length.
    past_length = kv_cache[self.layer_idx].num_kv
    if stage == 2:
        past_length -= ques.shape[1]
    # ===========================================

    pos = torch.arange(past_length, past_length + keys.shape[1])
    pos = pos[None, :].to(keys.device)
    cos, sin = self.rotary_emb(keys, pos)
    ques, keys = check_and_apply_qk_rope(ques, keys, cos, sin)

    # ================================================
    # Main Change 3
    kv_cache[self.layer_idx].update(keys, vals, stage)
    # ================================================

    # Here, we call our custom Triton kernel, passing the KV cache manager 
    # to get the page table.
    attn_output = flash_paged_attn_func(
        ques,
        keys,
        vals,
        kv_cache[self.layer_idx])

    attn_output = attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>C. Chunk-wise + Sparse Training</b> (<a href="chunkoptim/modifiers/train_blockwise_tp_sparse.py#L134" target="_blank">chunkoptim/modifiers/train_blockwise_tp_sparse.py:134</a>)</summary>

```python
"""
This code is mostly identical to code B. However, after the query projection 
is complete, sparse KV cache retrieval is immediately triggered to maximize the 
overlap between data transfer and computation.

Retrieval is performed using `kv_cache.select(query: Tensor, stage: int)`, 
an interface exclusive to the `SparseKVCache` class.
"""
def self_attn_forward(self, hidden_states, kv_cache):
    world_size = get_tensor_parallel_world_size()

    # =========================================
    stage = 2 if torch.is_grad_enabled() else 1
    kv_cache.visit(self.layer_idx, stage)
    # =========================================

    num_heads = self.config.num_attention_heads // world_size
    num_kv_heads = self.config.num_key_value_heads // world_size
    embed_dim = self.config.hidden_size
    head_dim = embed_dim // self.config.num_attention_heads

    # ===========================================
    past_length = kv_cache[self.layer_idx].num_kv
    if stage == 2:
        past_length -= hidden_states.shape[1]
    # ===========================================
        
    pos = torch.arange(past_length, past_length + hidden_states.shape[1])
    pos = pos[None, :].to(hidden_states.device)
    cos, sin = self.rotary_emb(hidden_states, pos)

    # Here, we only project the query first.
    ques = do_projection(self.q_proj, hidden_states, num_heads, head_dim, head_first=False)
    ques = apply_rope(ques, cos, sin)

    # ==========================================
    # Once the query projection is done, immediately trigger retrieval + KV cache onload.
    kv_cache[self.layer_idx].select(ques, stage)
    # ==========================================

    # The sparse KV cache onload happens concurrently with the Key and Value projections.
    keys = do_projection(self.k_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    vals = do_projection(self.v_proj, hidden_states, num_kv_heads, head_dim, head_first=False)
    keys = apply_rope(keys, cos, sin)

    # ================================================
    kv_cache[self.layer_idx].update(keys, vals, stage)
    # ================================================

    attn_output = flash_paged_sparse_attn_func(
        ques,
        keys,
        vals,
        kv_cache[self.layer_idx])

    attn_output = attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output
```

</details>
</td>
</tr>
</table>

## Core Modules

### Module Functionality Overview

* `SimpleCacheManager`: Implements basic paged memory management.
* `CacheManger`: Adds asynchronous CPU offloading to `SimpleCacheManager`.
* `SimpleSparseCacheManager`: Implements paged memory management with a retrieval mechanism and history tracking.
* `SparseCacheManager`: Adds CPU offloading to `SimpleSparseCacheManager`.
* `SparseKVCache`, `KVCache`: Manages KV cache prefetching and the overall chunk-wise training logic.
* `ops.flash_paged_topk.py`: Implements the attention computation for sparse paged KV cache. Offloading is transparent to this kernel.
* `ops.flash_paged_attn.py`: Implements the attention computation for dense paged KV cache. Offloading is transparent to this kernel.

---

<img src="docs/framework.png" width=600/>

### 1. Triton Kernel Code
<table style="width:100%; border: none;">
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>A. FlashAttention Kernel</b> (<a href="chunkoptim/ops/flash_attn.py" target="_blank">chunkoptim/ops/flash_attn.py</a>)</summary>

```python
"""
This code is consistent with the official Triton implementation of FlashAttention.
It does not include a paging mechanism and only supports parallel training.
"""

@triton.jit
def _bwd_kernel(
    Q, K, V, DO, DQ, DK, DV,
    LSE, D,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kvb, stride_kvh, stride_kvn,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    nheads, seqlen_q, q_start_idx, headdim,
    seqlen_q_rounded, seqlen_k, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    v_ptrs = V + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + (offs_m[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    dk_ptrs = DK + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    dv_ptrs = DV + off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    
    lse_ptrs = LSE + off_hb * seqlen_q_rounded + offs_m
    d_ptrs = D + off_hb * seqlen_q_rounded + offs_m

    mask_m = offs_m < seqlen_q
    if EVEN_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        lse_i = tl.load(lse_ptrs)
        Di = tl.load(d_ptrs)
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse_i = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        Di = tl.load(d_ptrs, mask=mask_m, other=0.0)
        
    dq_block = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    q_idx = q_start_idx + offs_m

    """
    A minor optimization: only the lower-triangular part of the attention 
    matrix is computed. The upper-triangular part is masked by default 
    to reduce latency.
    """
    for kv_block_idx in tl.range(0, tl.cdiv(q_start_idx, BLOCK_N) + start_m_block + 1):
        
        k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = k_idx[:, None] < seqlen_k

        k = tl.load(k_ptrs, mask=kv_mask)
        v = tl.load(v_ptrs, mask=kv_mask)

        qk = tl.dot(q, k.T)
        
        cond = q_idx[:,None] >= k_idx[None,:]
        qk = tl.where(cond, qk, float("-inf"))

        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        """
        Since our code supports Grouped-Query Attention (GQA), `atomic_add` is 
        used when calculating DK. The semantic is set to 'relaxed' to 
        maximize throughput.
        """
        dv_block = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_ptrs, dv_block, mask=kv_mask, sem='relaxed')

        dp = tl.dot(do, v.T)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        """
        DV calculation is the same as DK.
        """
        dk_block = tl.dot(ds.T, q)
        tl.atomic_add(dk_ptrs, dk_block, mask=kv_mask, sem='relaxed')

        dq_block += tl.dot(ds, k)

        k_ptrs += stride_kvn * BLOCK_N
        v_ptrs += stride_kvn * BLOCK_N
        dk_ptrs += stride_kvn * BLOCK_N
        dv_ptrs += stride_kvn * BLOCK_N

    if EVEN_M:
        tl.store(dq_ptrs, dq_block)
    else:
        tl.store(dq_ptrs, dq_block, mask=mask_m[:, None])
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>B. Flash Paged Attention Kernel</b> (<a href="chunkoptim/ops/flash_paged_attn.py" target="_blank">chunkoptim/ops/flash_paged_attn.py</a>)</summary>

```python
"""
The forward and backward pass logic is largely the same. Therefore, only the 
backward pass is commented here for your review.
"""
@triton.jit
def _bwd_kernel(
    Q, DO, DQ, T,
    LSE, D,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kvb, stride_kvh, stride_kvn,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    nheads, seqlen_q, q_start_idx, headdim,
    seqlen_q_rounded, seqlen_k, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP32_ATOMIC_ADD: tl.constexpr,
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    kv_offs = off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + (offs_m[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    
    lse_ptrs = LSE + off_hb * seqlen_q_rounded + offs_m
    d_ptrs = D + off_hb * seqlen_q_rounded + offs_m

    mask_m = offs_m < seqlen_q
    if EVEN_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        lse_i = tl.load(lse_ptrs)
        Di = tl.load(d_ptrs)
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse_i = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        Di = tl.load(d_ptrs, mask=mask_m, other=0.0)
        
    dq_block = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    q_idx = q_start_idx + offs_m
    start_m_block += 1

    for kv_block_idx in range(0, tl.cdiv(q_start_idx, BLOCK_N) + start_m_block):
        
        k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = k_idx[:, None] < seqlen_k

        """
        Here, we first read the page table and then use page offsets to read 
        content within the page frames. This is the main difference from 
        `flash_attn.py`; other parts are nearly identical.
        """
        k_page_ptr = tl.load(T)
        v_page_ptr = tl.load(T + 1)
        dk_page_ptr = tl.load(T + 2)
        dv_page_ptr = tl.load(T + 3)
        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))

        """
        Triton < 3.4 does not support bfloat16 atomic addition, and the 
        implementation in Triton 3.4 is slow. As a compromise, we use 
        float32 to store DK and DV.
        """
        if FP32_ATOMIC_ADD:
            dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.float32))
            dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.float32))
        else:
            dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.bfloat16))
            dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.bfloat16))
        
        k = tl.load(k_page_ptr + kv_offs, mask=kv_mask)
        v = tl.load(v_page_ptr + kv_offs, mask=kv_mask)
        
        qk = tl.dot(q, k.T)

        cond = q_idx[:,None] >= k_idx[None,:]
        qk = tl.where(cond, qk, float("-inf"))

        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        dv_block = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_page_ptr + kv_offs, dv_block, mask=kv_mask, sem='relaxed')

        dp = tl.dot(do, v.T)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        dk_block = tl.dot(ds.T, q)
        tl.atomic_add(dk_page_ptr + kv_offs, dk_block, mask=kv_mask, sem='relaxed')

        dq_block += tl.dot(ds, k)

        T += 4

    if EVEN_M:
        tl.store(dq_ptrs, dq_block)
    else:
        tl.store(dq_ptrs, dq_block, mask=mask_m[:, None])
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>C. Flash Paged Sparse Attention Kernel</b> (<a href="chunkoptim/ops/flash_paged_topk.py" target="_blank">chunkoptim/ops/flash_paged_topk.py</a>)</summary>

```python
@triton.jit
def _bwd_kernel(
    Q, DO, DQ, T, ST,
    LSE, D,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kvb, stride_kvh, stride_kvn,
    stride_sel_b, stride_sel_q,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    nheads, seqlen_q, q_start_idx, headdim,
    seqlen_q_rounded, seqlen_k, num_kv_heads,
    recent_offset, recent_pages,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP32_ATOMIC_ADD: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr
):
    start_m_block = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    off_kv_h = off_h // GROUP_SIZE

    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    kv_offs = off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + (offs_m[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    
    lse_ptrs = LSE + off_hb * seqlen_q_rounded + offs_m
    d_ptrs = D + off_hb * seqlen_q_rounded + offs_m

    mask_m = offs_m < seqlen_q
    if EVEN_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        lse_i = tl.load(lse_ptrs)
        Di = tl.load(d_ptrs)
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse_i = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        Di = tl.load(d_ptrs, mask=mask_m, other=0.0)
        
    dq_block = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    sel_ptrs = ST + off_b * stride_sel_b + start_m_block * stride_sel_q
    q_idx = q_start_idx + offs_m
    """
    Sparse attention is handled in two loops. 
    * The first loop processes the retrieved portions.
    * The second loop handles the local attention window.

    Below is the first loop.
    """
    for sel_idx in tl.range(NUM_SEL_KV_BLOCKS):

        """
        Read the offset, which is the position of the current block to be 
        computed within the dense KV cache.
        """
        kv_block_idx = tl.load(sel_ptrs + sel_idx)
        sel_k_idx = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = sel_k_idx[:, None] < seqlen_k

        """
        Use this offset to find the address of the KV cache page.
        """
        k_page_ptr = tl.load(T + kv_block_idx * 4)
        v_page_ptr = tl.load(T + kv_block_idx * 4 + 1)
        dk_page_ptr = tl.load(T + kv_block_idx * 4 + 2)
        dv_page_ptr = tl.load(T + kv_block_idx * 4 + 3)

        """
        Cast the address to a pointer type.
        """
        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))

        if FP32_ATOMIC_ADD:
            dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.float32))
            dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.float32))
        else:
            dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.bfloat16))
            dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.bfloat16))
        
        k = tl.load(k_page_ptr + kv_offs, mask=kv_mask)
        v = tl.load(v_page_ptr + kv_offs, mask=kv_mask)

        qk = tl.dot(q, k.T)
        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        dv_block = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_page_ptr + kv_offs, dv_block, mask=kv_mask, sem='relaxed')

        dp = tl.dot(do, v.T)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        dk_block = tl.dot(ds.T, q)
        tl.atomic_add(dk_page_ptr + kv_offs, dk_block, mask=kv_mask, sem='relaxed')

        dq_block += tl.dot(ds, k)

    """
    This is the second loop.
    """
    for recent_idx in tl.range(recent_offset, recent_offset + recent_pages):

        recent_k_idx = recent_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = recent_k_idx[:, None] < seqlen_k

        k_page_ptr = tl.load(T + recent_idx * 4)
        v_page_ptr = tl.load(T + recent_idx * 4 + 1)
        dk_page_ptr = tl.load(T + recent_idx * 4 + 2)
        dv_page_ptr = tl.load(T + recent_idx * 4 + 3)

        k_page_ptr = tl.cast(k_page_ptr, tl.pointer_type(tl.bfloat16))
        v_page_ptr = tl.cast(v_page_ptr, tl.pointer_type(tl.bfloat16))

        if FP32_ATOMIC_ADD:
            dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.float32))
            dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.float32))
        else:
            dk_page_ptr = tl.cast(dk_page_ptr, tl.pointer_type(tl.bfloat16))
            dv_page_ptr = tl.cast(dv_page_ptr, tl.pointer_type(tl.bfloat16))
        
        k = tl.load(k_page_ptr + kv_offs, mask=kv_mask)
        v = tl.load(v_page_ptr + kv_offs, mask=kv_mask)

        qk = tl.dot(q, k.T)

        cond = q_idx[:,None] >= recent_k_idx[None,:]
        qk = tl.where(cond, qk, float("-inf"))

        p = tl.exp(qk * softmax_scale - lse_i[:, None])

        dv_block = tl.dot(p.to(do.dtype).T, do)
        tl.atomic_add(dv_page_ptr + kv_offs, dv_block, mask=kv_mask, sem='relaxed')

        dp = tl.dot(do, v.T)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)

        dk_block = tl.dot(ds.T, q)
        tl.atomic_add(dk_page_ptr + kv_offs, dk_block, mask=kv_mask, sem='relaxed')

        dq_block += tl.dot(ds, k)

    if EVEN_M:
        tl.store(dq_ptrs, dq_block)
    else:
        tl.store(dq_ptrs, dq_block, mask=mask_m[:, None])
```

</details>
</td>
</tr>
</table>

### 2. KV Cache Manager Implementations
<table style="width:100%; border: none;">
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>A. SimpleCacheManager Code</b> (<a href="chunkoptim/cache/kv_cache.py#L8" target="_blank">chunkoptim/cache/kv_cache.py:8</a>)</summary>

```python
class SimpleCacheManager:
    def __init__(self, batch_size, page_size, num_kv_heads, head_dim, local_rank):
        super().__init__()

        """
        Basic KV cache metadata.
        """
        self.batch_size = batch_size
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = 'cuda'
        self.cuda = f'cuda:{dist.get_rank() if local_rank is None else local_rank}'
        self.reset()

    def reset(self):

        """
        State tracking variables.
        """
        self.num_kv = 0
        self.last_update_token = []
        self.last_update_pages = []

        """
        `grad_hook` is a backward hook related to offloading and prefetching.
        """
        self.grad_hook = None

        """
        Tensors within `self.xxx_gpu` point to data blocks on the GPU.
        """
        self.key_gpu = []
        self.val_gpu = []
        self.kgd_gpu = []
        self.vgd_gpu = []

    @property
    @torch.inference_mode()
    def page_table(self):
        """
        The page table is computed on-the-fly because the memory locations of the 
        KV cache can change with each onload. The returned tensor has a shape 
        of (P, 4), where P is the number of pages, and 4 corresponds to key, 
        value, key gradient, and value gradient.
        """
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
        """
        Returns the gradients of the KV cache corresponding to the chunk 
        currently being processed.
        """
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
        """
        Used in chunk-wise training. This removes a KV cache chunk from right to left, 
        indicating that all computations related to it are finished and it can be evicted.
        """
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
        """
        Stage 1 already saves the entire KV cache, so Stage 2 does not need to 
        update it again. See 'User Interface -> Inner Attention Layer Code -> B. Chunk-wise 
        Training' for an explanation of stages 1 & 2.
        """
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
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>B. CacheManager Code</b> (<a href="chunkoptim/cache/kv_cache.py#L147" target="_blank">chunkoptim/cache/kv_cache.py:147</a>)</summary>

```python
"""
This class extends SimpleCacheManager by adding offload functionality.
"""
class CacheManager(SimpleCacheManager):
    def __init__(self, ...):
        super().__init__(...)

        """
        Handles for the offload and onload worker futures.
        """
        self.offload_future = None
        self.onload_future = None

        """
        Onload and offload operations are performed on a separate CUDA stream 
        to overlap with computation.
        """
        self.stream = Stream()
        self.pool = ThreadPoolExecutor(1)
        self.fake = torch.zeros((1,), device=self.cuda)

    def reset(self):
        super().reset()
        """
        `self.xxx_cpu` stores tensors that point to CPU memory.
        """
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
        """
        After the parent object removes the GPU tensors, the corresponding 
        data in CPU memory is also removed.
        """
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

        """
        First, update the data on the GPU, then allocate corresponding pinned 
        memory on the CPU. The data is not immediately copied from GPU to CPU; 
        this is deferred until the next offload cycle.
        """
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
            """
            For dense attention, all pages must be onloaded. The process differs 
            between stages. In Stage 1, only keys and values are needed. In 
            Stage 2, gradients are also transferred.
            """
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
            """
            The offload logic is similar to onload.
            """
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

    """
    `wait_onload` and `wait_offload` are used to synchronize the main thread 
    with the data transfer threads.
    """
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
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>C. KVCache Code</b> (<a href="chunkoptim/cache/kv_cache.py#L300" target="_blank">chunkoptim/cache/kv_cache.py:300</a>)</summary>

```python
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

        """
        Each layer has its own KV cache manager.
        """
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
        """
        The `visit(layer_idx)` interface informs the KVCache that `layer-idx` is 
        currently being accessed, triggering the corresponding prefetch and 
        offload operations.
        """
        if self.cpu_offload is not None:
            factor = -1 if reverse else 1
            """
            Determine which layers need to be prefetched and which need to be offloaded.
            """
            cuda_layers = [
                (layer_idx + self.num_layers + factor * i) % self.num_layers 
                for i in range(self.cpu_offload)]
            cpu_layers = list(filter(lambda i: i not in cuda_layers, range(self.num_layers)))
            
            """
            Invoke the corresponding asynchronous transfer operations.
            """
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
        """
        During the backward pass, attach gradient hooks to all KV cache managers. 
        When the gradient is computed, the hook is triggered, calling the `visit` 
        function and initiating prefetch/offload operations.
        """
        for idx, m in enumerate(self.managers):
            m.grad_hook = lambda idx=idx: self.visit(idx, 2, True)

    def post_process(self):
        """
        After processing each chunk, this function is called to evict the 
        corresponding KV cache.
        """ 
        for m in self.managers:
            m.remove_last_update()
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>D. SimpleSparseCacheManager Code</b> (<a href="chunkoptim/cache/topk_cache.py#L21" target="_blank">chunkoptim/cache/topk_cache.py:21</a>)</summary>

```python
"""
Extends SimpleCacheManager by overriding key methods to support sparse attention.
"""
class SimpleSparseCacheManager(SimpleCacheManager):
    def __init__(self, ...):
        super().__init__(...)
        self.page_budget = page_budget

    def reset(self):
        super().reset()
        """
        `self.avg` stores the historical page averages of keys. 
        `self.idx` stores historical retrieval results to ensure alignment 
        between the forward and backward passes.
        """
        self.avg = None
        self.idx = []
        
    @torch.inference_mode()
    def remove_last_update(self):
        del self.idx[-1]
        return super().remove_last_update

    @torch.inference_mode()
    def select(self, query, stage):
        if stage == 2:
            return
        end = sum(self.last_update_pages)
        if end <= self.page_budget:
            self.idx.append(None)
            return
        """
        This is the core retrieval logic. It computes new key averages, votes for 
        the most important pages, and saves their indices.
        """
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
        """
        After the parent object completes the update, it also computes the 
        average for the current key and updates the key average list.
        """
        avg = torch.cat([k.mean(dim=1, keepdim=True) for k in key], dim=1)
        avg = avg.permute(0, 2, 3, 1).contiguous()
        self.avg = avg if self.avg is None else torch.cat([self.avg, avg], dim=-1)
        return key, val
```


</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>E. SparseCacheManager Code</b> (<a href="chunkoptim/cache/topk_cache.py#L61" target="_blank">chunkoptim/cache/topk_cache.py:61</a>)</summary>

```python
"""
Extends SimpleSparseCacheManager by adding a KV cache offload mechanism.
"""
class SparseCacheManager(SimpleSparseCacheManager):
    def __init__(self, ...):
        super().__init__(...)
        """
        Handlers for data transfer thread futures.
        """
        self.offload_future = None
        self.onload_future = None

        self.stream = Stream()
        self.pool = ThreadPoolExecutor(1)
        self.fake = torch.zeros(1, device='cuda')

    def reset(self):
        super().reset()
        """
        Tensors in `self.xxx_cpu` point to data in CPU memory.
        """
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

        """
        Pre-allocate empty memory regions on the CPU for future offloads.
        """
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
        """
        First, execute the parent's `select` function to update `self.idx`.
        """
        super().select(query, stage)
        
        sps_index = self.idx[-1]
        num_pages = len(self.key_cpu)
        rng_onload = num_pages if stage == 1 else (num_pages - self.last_update_pages[-1])
        
        @torch.inference_mode()
        def worker():
            """
            If the retrieval result is `None`, it means either the number of 
            historical KV cache pages is below the budget, or there is no KV 
            cache yet (i.e., processing the first chunk). In these cases, all 
            available KV cache pages are prefetched.
            """
            if sps_index is None:
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

                """
                In Stage 1, `select` is called before `update`. The key/value for the 
                current chunk have not been added yet, but they will be on the GPU 
                when they are. Therefore, the following code only onloads the 
                retrieved KV cache pages.
                """
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

                """
                In Stage 2, the current chunk's KV cache was already computed and 
                stored in Stage 1. However, KV cache pages from the local window might 
                have been offloaded to the CPU. Therefore, we must onload both the 
                retrieved KV cache pages and the local window's KV cache pages.
                """
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
        """
        The logic for offload is broadly similar to onload.
        """
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
```

</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>F. SparseKVCache Code</b> (<a href="chunkoptim/cache/topk_cache.py#L261" target="_blank">chunkoptim/cache/topk_cache.py:261</a>)</summary>

```python
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
        """
        The `visit(layer_idx)` interface is called to mark the currently accessed 
        layer, which in turn triggers the corresponding onload and offload 
        operations. `reverse` is a flag indicating that the process is in the 
        backward pass.
        """
        if self.cpu_offload is not None:
            factor = -1 if reverse else 1

            """
            Determine which layers should reside on the GPU and which should be on the CPU.
            """
            cuda_layers = [
                (layer_idx + self.num_layers + factor * i) % self.num_layers 
                for i in range(self.cpu_offload)]
            cpu_layers = list(filter(lambda i: i not in cuda_layers, range(self.num_layers)))

            """
            Trigger `offload` or `select` (which handles onload).
            """
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
        """
        Attach hook functions to all layers. When a layer's gradient is accessed, 
        the hook is triggered, executing the corresponding `visit` method to 
        perform onload and offload.
        """
        for idx, m in enumerate(self.managers):
            m.grad_hook = lambda idx=idx: self.visit(idx, 2, True)

    def post_process(self):
        """
        After each block's computation is complete, the corresponding KV cache 
        must be removed to maximize memory utilization.
        """
        for m in self.managers:
            m.remove_last_update()
```

</details>
</td>
</tr>
</table>

### 3. Tensor Parallel and Context Parallel Implementation
<table style="width:100%; border: none;">
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>A. Tensor Parallel Code</b> (<a href="chunkoptim/modifiers/train_blockwise_tp.py#L21" target="_blank">chunkoptim/modifiers/train_blockwise_tp.py:21</a>)</summary>

```python
def get_tensor_parallel_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def get_tensor_parallel_rank():
    return dist.get_rank() if dist.is_initialized() else 0

class ColumnParallelLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.world_size = get_tensor_parallel_world_size()
        self.rank = get_tensor_parallel_rank()

        output_size_per_partition = linear_layer.out_features // self.world_size
        self.weight = nn.Parameter(
            linear_layer.weight.data[
                self.rank * output_size_per_partition: (self.rank + 1) * output_size_per_partition, :
            ].clone()
        )
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(
                linear_layer.bias.data[
                    self.rank * output_size_per_partition: (self.rank + 1) * output_size_per_partition
                ].clone()
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output_parallel = F.linear(x, self.weight, self.bias)
        return output_parallel

class RowParallelLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.world_size = get_tensor_parallel_world_size()
        self.rank = get_tensor_parallel_rank()

        input_size_per_partition = linear_layer.in_features // self.world_size
        self.weight = nn.Parameter(
            linear_layer.weight.data[
                :, self.rank * input_size_per_partition: (self.rank + 1) * input_size_per_partition
            ].clone()
        )
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output_parallel = F.linear(x, self.weight)
        
        if self.world_size > 1:
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
    
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        return output_parallel

class ModelForTraining:
    # ...
    def _parallelize_model(self, model):
        world_size = get_tensor_parallel_world_size()
        if world_size <= 1:
            return

        for layer in model.model.layers:
            # Attention
            layer.self_attn.q_proj = ColumnParallelLinear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = ColumnParallelLinear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = ColumnParallelLinear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = RowParallelLinear(layer.self_attn.o_proj)
            # MLP
            layer.mlp.gate_proj = ColumnParallelLinear(layer.mlp.gate_proj)
            layer.mlp.up_proj = ColumnParallelLinear(layer.mlp.up_proj)
            layer.mlp.down_proj = RowParallelLinear(layer.mlp.down_proj)

        model.lm_head = ColumnParallelLinear(model.lm_head)
```
</details>
</td>
</tr>
<tr style="vertical-align: top;">
<td style="padding: 5px; border: none;">
<details>
<summary><b>B. Context Parallel Implementation</b> (<a href="chunkoptim/modifiers/train_ringflash.py" target="_blank">chunkoptim/modifiers/train_ringflash.py</a>)</summary>

```python
from ring_flash_attn import update_ring_flash_attn_params
from ring_flash_attn.adapters.hf_adapter import create_ring_flash_attention_forward, check_params

def self_attn_forward(self, hidden_states, attention_mask):

    # ... (projection and rope) ...

    attn_output = self.ring_attention(
        query_states=ques,
        key_states=keys,
        value_states=vals,
        attention_mask=attention_mask,
        query_length=ques.shape[1],
        is_causal=True)

    attn_output = attn_output.flatten(2)
    attn_output = self.o_proj(attn_output)

    return attn_output


class ModelForTraining(Modifier):
    def __init__(self, model, save_ckp: str, load_ckp: str, config: str):
        # ...
        for layer in model.model.layers:
            # ...
            ring_attention = create_ring_flash_attention_forward(None, 1)[0]
            layer.self_attn.ring_attention = lambda *args, **kwargs: ring_attention(*args, **kwargs)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

    def forward(self, input_ids, labels):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        cu_seqlens = torch.tensor([0, input_ids.shape[-1]], dtype=torch.int32, device=rank)
        update_ring_flash_attn_params(cu_seqlens, None)

        input_ids_chunk = torch.chunk(input_ids, world_size, dim=1)[rank]
        logits_chunk = self.model(input_ids=input_ids_chunk)
        labels_chunk = torch.chunk(labels, world_size, dim=1)[rank]

        loss = F.cross_entropy(
            logits_chunk.view(-1, logits_chunk.shape[-1]), 
            labels_chunk.view(-1),
            reduction='mean')
        
        return loss
```

</details>
</td>
</tr>
</table>
