```python
def blockwise(model, batch, grad_ckpt, block_size, page_size, cpu_offload):
    my_chunkize = partial(chunkize, dim=-1, chunk_size=block_size)
    input_ids = list(my_chunkize(batch['input_ids']))
    labels = list(my_chunkize(batch['labels']))

    kv_cache = KVCache(
        num_layers=model.model.config.num_hidden_layers,
        batch_size=1,
        page_size=page_size,
        num_heads=model.model.config.num_key_value_heads,
        cpu_offload=cpu_offload)

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
        loss = model(**inputs).sum() / batch['seq_len']

        # backward prop
        kv_cache.pre_process()
        loss.backward()
        kv_cache.post_process()
```