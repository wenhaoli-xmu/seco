# OOMB

## Overview

OOMB incorporates numerous optimization techniques, including:

  * **Paged KV Cache & Gradient Management:** Improves memory scaling performance.
  * **Independent KV Cache Management:** A more efficient system that operates independently of the Torch autograd engine.
  * **CPU Offloading:** Full support for offloading to CPU memory.

**Tensor Parallelism (TP) Support**

  * Source code is available in the `chunkoptim/modifiers` directory.
  * Reduces single-GPU memory usage by nearly 50%.
  * Training time decreases almost linearly as the number of parallel GPUs increases.

**Top-K Sparse Attention Support**

  * Allows training time to grow **nearly linearly** with context length.
  * In sparse attention mode, the communication overhead from CPU offloading does not increase with context length.

## Quick Start

### 1\. Create a `KVCache` or `SparseKVCache` object

  * `SparseKVCache` includes an additional `page_budget` parameter, which specifies the number of pages for the Top-K attention computation.
  * The `cpu_offload` parameter can be set to `2` (enabled) or `None` (disabled).
  * It is recommended to load the kernel on `gpu:0` first. This allows processes on other GPUs to load it directly from the cache, avoiding concurrent loading conflicts.
  * While a `batch_size` \> 1 is supported, it offers little practical benefit for extremely long sequences.

<!-- end list -->

```python
from chunkoptim.cache.kv_cache import KVCache
from chunkoptim.cache.topk_cache import SparseKVCache
import torch.distributed as dist

# It's recommended to initialize on rank 0 first to prevent race conditions
if dist.get_rank() == 0:
    kv_cache = KVCache(
        num_layers=model.model.config.num_hidden_layers,
        batch_size=1,
        page_size=page_size,
        num_heads=model.model.config.num_key_value_heads // dist.get_world_size(),
        cpu_offload=cpu_offload
    )
dist.barrier()
if dist.get_rank() != 0:
    kv_cache = KVCache(
        num_layers=model.model.config.num_hidden_layers,
        batch_size=1,
        page_size=page_size,
        num_heads=model.model.config.num_key_value_heads // dist.get_world_size(),
        cpu_offload=cpu_offload
    )
dist.barrier()
```

### 2\. Chunk the input data

  * You can use the convenient chunking utility provided in our `utils`.
  * Here, `4096` is the chunk size, which acts as the processing unit for the context. For GPUs with larger memory, you can increase this value to reduce the total number of training steps.

<!-- end list -->

```python
from chunkoptim.utils import chunkize
from functools import partial

my_chunkize = partial(chunkize, dim=-1, chunk_size=4096)

# input_ids: [bsz, seq_len]
# labels: [bsz, seq_len]

input_ids_chunks = list(my_chunkize(input_ids))
labels_chunks = list(my_chunkize(labels))
```

### 3\. Implement the block-wise training pipeline

  * This code structure works for any combination of techniques, such as using TP, sparse attention, or both simultaneously.
  * The `pre_process` function is primarily related to CPU offloading.
  * The `post_process` function is mainly for handling backpropagation.

<!-- end list -->

```python
# First forward pass without gradients to populate the KV cache
with torch.no_grad():
    for chunk_input, chunk_target in zip(input_ids_chunks, labels_chunks):
        inputs = dict(
            input_ids=chunk_input,
            labels=chunk_target,
            kv_cache=kv_cache,
            grad_ckpt=False
        )
        model(**inputs)

# Second forward pass and backward pass, chunk by chunk in reverse
for chunk_input, chunk_target in reversed(list(zip(input_ids_chunks, labels_chunks))):
    
    # Forward propagation
    inputs = dict(
        input_ids=chunk_input,
        labels=chunk_target,
        kv_cache=kv_cache,
        grad_ckpt=grad_ckpt # Enable gradient checkpointing here
    )
    loss = model(**inputs).sum() / total_seq_len

    # Backward propagation
    kv_cache.pre_process()
    loss.backward()
    kv_cache.post_process()
```

For **DeepSpeed** integration, please refer to the pipeline in `test_efficiency/test_ds.py`, which requires only minor modifications to the code above.
