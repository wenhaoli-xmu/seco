# SeCO v2: 让LLM在单卡上训练4M上下文

## Overview

* 在SeCO v1的基础上，加入了大量的优化技术，包括：
    * paged kv cache & its gradients 管理，提高内存scale表现
    * 独立于torch autograd system的kv cache管理，更加高效
    * 支持cpu offload

* 支持tensor parallel
    * 源代码在`chunkoptim/modifiers`文件夹下
    * 能够将单卡内存减少接近一半
    * 训练时间随着并行卡数增多近线性减少

* 支持topk sparse attention
    * 允许训练时间随着context length **近线性增长**
    * 在sparse attention模式下，cpu offload的通信量不随着上下文增长而增长

## 快速入门

1. 创建`KVCache`或者`SparseKVCache`对象

    * `SparseKVCache`比`KVCache`多一个参数`page_budget`，表示attention计算采用的topk的page数
    * `cpu_offload`参数可选 `2` 或者 `None`，分别表示启动offload或者关闭
    * 要尽量在gpu:0加载kernel，这样其他gpu中的进程可以在gpu0加载好之后直接从缓存调用，从而避免一起加载导致的冲突
    * 支持batch size > 1，但是在超长文本中 >1 的batch size没有什么实际意义

    ```python
    from chunkoptim.cache.kv_cache import KVCache
    from chunkoptim.cache.topk_cache import SparseKVCache

    if dist.get_rank() == 0:
        if dist.get_rank() == 0:
            kv_cache = KVCache(
                num_layers=model.model.config.num_hidden_layers,
                batch_size=1,
                page_size=page_size,
                num_heads=model.model.config.num_key_value_heads // dist.get_world_size(),
                cpu_offload=cpu_offload)
        dist.barrier()
        if dist.get_rank() != 0:
            kv_cache = KVCache(
                num_layers=model.model.config.num_hidden_layers,
                batch_size=1,
                page_size=page_size,
                num_heads=model.model.config.num_key_value_heads // dist.get_world_size(),
                cpu_offload=cpu_offload)
        dist.barrier()
    ```

2. 将输入切分成块

    * 可以直接使用我们在utils中提供了便捷的切分工具
    * 这里的4096就是最终处理上下文的单位，对于内存更大的GPU，可以尽量调高此值，从而减少回合数

    ```python
    from chunkoptim.utils import chunkize
    from functools import partial

    my_chunkize = partial(chunkize, dim=-1, chunk_size=4096)

    # input_ids: [bsz, n]
    # labels: [bsz, n]

    input_ids = list(my_chunkize(input_ids))
    labels = list(my_chunkize(labels))
    ```

3. 编写block-wise training pipeline

    * 对于任意technique的组合，例如 +TP, +sparse attention，或者同时使用两者，都可以凭这段代码实现
    * `pre_process`函数的作用主要和cpu-offload有关
    * `post_process`函数则主要与反向传播有关

    ```python
    with torch.no_grad():
        for chunk_input, chunk_target in zip(input_ids, labels):

            # forward pass
            inputs = dict(
                input_ids=chunk_input,
                labels=chunk_target,
                kv_cache=kv_cache,
                grad_ckpt=False)
            model(**inputs)

    for chunk_input, chunk_target in reversed(list(zip(input_ids, labels))):

        # forward prop
        inputs = dict(
            input_ids=chunk_input,
            labels=chunk_target,
            kv_cache=kv_cache,
            grad_ckpt=grad_ckpt)
        loss = model(**inputs).sum() / seq_len

        # backward prop
        kv_cache.pre_process()
        loss.backward()
        kv_cache.post_process()
    ```

    如果要支持deepspeed，则可以参考`test_efficiency/test_ds.py`中的pipeline，相比上面的代码只有少量更改
