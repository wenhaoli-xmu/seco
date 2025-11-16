from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import torch
import json

from corpus import get_processor, LazyRandomSampleCorpus
from chunkoptim.utils import (
    get_model_and_tokenizer, 
    get_env_conf, 
    chunkize,
    History)
from functools import partial
from pathlib import Path
import argparse, random, numpy, os
from pygments.console import colorize


def zero_grad(params):
    for param in params:
        param.grad = None


def build_dataset(env_conf, tokenizer):
    sum_partition = 0

    num_iters = env_conf['train']['train_iters']
    corpus = []
    for info in env_conf['train']['corpus']:
        sum_partition += info['partition']
        num_instance = int(info['partition'] * num_iters)

        proc = get_processor(info['conf'], tokenizer)
        corp = LazyRandomSampleCorpus(info['data'], proc, max_instance=num_instance, use_cache=False)
        corpus.append(corp)

    assert sum_partition == 1
    return ConcatDataset(corpus)


def collate_fn(batch):
    input_ids = batch[0]['input_ids']
    labels = input_ids[1:] + [-100]

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
    labels = torch.tensor(labels, dtype=torch.int64, device='cuda')

    input_ids = input_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)

    seq_len = input_ids.shape[-1]

    return dict(
        input_ids=input_ids,
        labels=labels,
        seq_len=seq_len)


def seed_everything(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def backend_setup():
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def backend_cleanup():
    dist.destroy_process_group()


def launch_test(args, pipeline):
    backend_setup()
    
    env_conf = args.env_conf
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    args.context = eval(args.context)


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    model.train()

    """
    NOTE: Rank0 dataset loading is ahead of other ranks. This is because data buffer is saved after rank0 finishes,
    thus others can utilize this buffer to avoid redundant processing and ensure consistency across ranks.
    """
    if dist.get_rank() == 0:
        corpus = build_dataset(env_conf, tokenizer)
    dist.barrier()
    if dist.get_rank() != 0:
        corpus = build_dataset(env_conf, tokenizer)
    dist.barrier()

    loader = DataLoader(
        corpus, 
        batch_size=1, 
        collate_fn=collate_fn)

    base_memory_allocated = torch.cuda.max_memory_allocated()
    print(colorize("yellow", "Base GPU memory allocated:") + colorize("green", f"{base_memory_allocated // 1024 ** 2} MB"))
    
    batch = next(iter(loader))

    for context in args.context:

        while batch['input_ids'].shape[-1] < context:
            batch['input_ids'] = torch.cat([batch['input_ids'], batch['input_ids']], dim=-1)
            batch['labels'] = torch.cat([batch['labels'], batch['labels']], dim=-1)
        batch['input_ids'] = batch['input_ids'][..., :context]
        batch['labels'] = batch['labels'][..., :context]
        batch['seq_len'] = context

        history = History(1_000_000)

        for _ in range(2 if context < 65536 else 1):
            
            history.init()

            pipeline(
                model=model,
                batch=batch)

            history.step(0, batch['seq_len'])

        mean_time, mean_memory = history.summary(False)
        template = colorize("yellow", f"{context:<5d}\t|") + "{mean_time:<3.3f}\t| {mean_memory:.3f}"
        print(template.format(mean_time=mean_time, mean_memory=mean_memory), flush=True)

    backend_cleanup()


def get_grad(params):
    grads = []
    for param in params:
        if param.grad is not None and param.grad.data is not None:
            grads.append(param.grad.data.ravel())
        else:
            grads.append(torch.zeros_like(param).ravel())
    return torch.cat(grads, dim=0)


def print_grad(params):
    for param in params:
        if param.grad is not None and param.grad.data is not None:
            print(param.abs().sum().item())
        else:
            print("none")


def baseline_tensor_parallel(model, batch, grad_ckpt):
    loss = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        kv_cache=None,
        grad_ckpt=grad_ckpt).sum() / batch['seq_len']
    loss.backward()


def blockwise_tensor_parallel_sparse(model, batch, grad_ckpt, block_size, page_size, cpu_offload, page_budget, visualize):
    my_chunkize = partial(chunkize, dim=-1, chunk_size=block_size)
    input_ids = list(my_chunkize(batch['input_ids']))
    labels = list(my_chunkize(batch['labels']))

    from chunkoptim.cache.topk_cache import SparseKVCache

    world_size = dist.get_world_size()

    if dist.get_rank() == 0:
        kv_cache = SparseKVCache(
            num_layers=model.model.config.num_hidden_layers,
            batch_size=1,
            page_size=page_size,
            num_heads=model.model.config.num_key_value_heads// world_size,
            cpu_offload=cpu_offload,
            page_budget=page_budget)
    dist.barrier()
    if dist.get_rank() != 0:
        kv_cache = SparseKVCache(
            num_layers=model.model.config.num_hidden_layers,
            batch_size=1,
            page_size=page_size,
            num_heads=model.model.config.num_key_value_heads // world_size,
            cpu_offload=cpu_offload,
            page_budget=page_budget)
    dist.barrier()

    with torch.no_grad():
        for chunk_input, chunk_target in zip(input_ids, labels):

            # forward pass
            inputs = dict(
                input_ids=chunk_input,
                labels=chunk_target,
                kv_cache=kv_cache,
                grad_ckpt=False)
            model(**inputs)

    if visualize is not None:
        import os
        import math
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        num_pages = int(math.ceil(batch['input_ids'].shape[-1] / page_size))
        mask = torch.zeros((num_pages, num_pages), dtype=torch.bool)
        num_pages_per_chunk = block_size // page_size
        manager = kv_cache.managers[visualize]

        for chunk_idx, (pages, table) in enumerate(zip(manager.last_update_pages, manager.idx)):
            start = num_pages_per_chunk * chunk_idx
            end = start + pages
            if table is None:
                mask[start: end, :] = True
            else:
                table = table[0]
                for query_page_idx, table_per_query in enumerate(table):
                    for page_idx in table_per_query:
                        if (start + query_page_idx) < num_pages and page_idx < num_pages:
                            mask[start + query_page_idx, page_idx] = True
            
            valid_start = start
            valid_end = min(end, num_pages)
            if valid_start < valid_end:
                mask[valid_start: valid_end, valid_start: valid_end] = True

        rng = torch.arange(num_pages)
        causal_mask = rng[:, None] >= rng[None, :]
        mask = mask & causal_mask

        os.makedirs("visualize", exist_ok=True)
        file_name = f"visualize/ctx{batch['input_ids'].shape[-1]}-pgs{page_size}-cks{block_size}-bgt{page_budget}-lyr{visualize}.jpg"

        dark_red = [x / 255 for x in [21, 56, 89]]
        light_red_almost_white = [x / 255 for x in [254, 245, 220]]

        cmap_colors = [
            (0.0, light_red_almost_white),
            (1.0, dark_red)
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom_red_cmap", cmap_colors)

        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap=custom_cmap, vmin=0, vmax=1)
        
        plt.axis('off') 
        
        plt.savefig(file_name, dpi=640, bbox_inches='tight', pad_inches=0)
        plt.close()

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


def blockwise_tensor_parallel(model, batch, grad_ckpt, block_size, page_size, cpu_offload):
    my_chunkize = partial(chunkize, dim=-1, chunk_size=block_size)
    input_ids = list(my_chunkize(batch['input_ids']))
    labels = list(my_chunkize(batch['labels']))

    from chunkoptim.cache.kv_cache import KVCache

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, default="[10240 * (i + 1) for i in range(10)]")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    args.config = json.load(open(args.config, 'r'))
    args.env_conf = Path(os.path.dirname(__file__)) / 'model.json'
    args.env_conf = get_env_conf(args.env_conf)

    method = args.config.pop('method')
    args.env_conf['model']['model_method'] = method
    kwargs = args.config

    if method == 'baseline-tp':
        pipe = baseline_tensor_parallel

    elif method == 'blockwise-tp':
        pipe = blockwise_tensor_parallel

    elif method == 'blockwise-tp-sparse':
        pipe = blockwise_tensor_parallel_sparse

    pipe = partial(pipe, **kwargs)
    launch_test(args, pipe)