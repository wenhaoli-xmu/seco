from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torch.distributed as dist

import torch
import json

from corpus import get_processor, LazyRandomSampleCorpus
from chunkoptim.utils import (
    get_optimizer_and_lr_adjuster,
    get_model_and_tokenizer, 
    get_env_conf, 
    chunkize,
    History)
from functools import partial
from pathlib import Path
from chunkoptim.cache.kv_cache import KVCache
import argparse, random, numpy, os
from pygments.console import colorize


class PackedTokenDataset(Dataset):
    def __init__(self, file_path: str):
        super().__init__()
        self.data = torch.load(file_path)            
        self.num_chunks, self.chunk_size = self.data.shape
        print(f"✅ data loaded")
        print(f"  - each dataset contains {self.num_chunks} data blocks.")
        print(f"  - each block consists of {self.chunk_size} tokens.")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx: int):
        chunk = self.data[idx].tolist()
        input_ids = chunk[:-1]
        labels = chunk[1:]
        
        return {
            "input_ids": input_ids,
            "labels": labels}


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

        corp = PackedTokenDataset(info['data'])
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


def launch_train(args, pipeline):
    backend_setup()
    
    env_conf = args.env_conf
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    train_iters = env_conf['train']["train_iters"]

    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    model.train()

    # load optimizer and lr adjuster
    params = model.ft_params()
    optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(**env_conf['train'], params=params)

    # build dataset
    corpus = build_dataset(env_conf, tokenizer)
    loader = DataLoader(
        corpus, 
        batch_size=1, 
        collate_fn=collate_fn)

    base_memory_allocated = torch.cuda.max_memory_allocated()
    print(colorize("yellow", "Base GPU memory allocated:") + colorize("green", f"{base_memory_allocated // 1024 ** 2} MB"))
    history = History(1)

    for step, batch in enumerate(loader):
        lr_adjuster(step=step)

        history.init()

        loss = pipeline(
            model=model,
            batch=batch)
        history.step(loss, batch['seq_len'])

        if (step + 1) % args.accum_grad == 0:
            optimizer.step()
            zero_grad(params)

        if (step + 1) >= train_iters:
            break

    output = json.dumps(history.loss)
    print(output)
    backend_cleanup()


def baseline_tensor_parallel(model, batch, grad_ckpt):
    loss = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        kv_cache=None,
        grad_ckpt=grad_ckpt).sum() / batch['seq_len']
    loss.backward()
    return loss.item()


def blockwise_tensor_parallel_sparse(model, batch, grad_ckpt, block_size, page_size, cpu_offload, page_budget):
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

    accum_loss = 0

    with torch.no_grad():
        for chunk_input, chunk_target in zip(input_ids, labels):

            # forward pass
            inputs = dict(
                input_ids=chunk_input,
                labels=chunk_target,
                kv_cache=kv_cache,
                grad_ckpt=False)
            accum_loss += model(**inputs).sum().item() / batch['seq_len']

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

    return accum_loss


def blockwise_tensor_parallel(model, batch, grad_ckpt, block_size, page_size, cpu_offload):
    my_chunkize = partial(chunkize, dim=-1, chunk_size=block_size)
    input_ids = list(my_chunkize(batch['input_ids']))
    labels = list(my_chunkize(batch['labels']))

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

    accum_loss = 0

    with torch.no_grad():
        for chunk_input, chunk_target in zip(input_ids, labels):

            # forward pass
            inputs = dict(
                input_ids=chunk_input,
                labels=chunk_target,
                kv_cache=kv_cache,
                grad_ckpt=False)
            accum_loss += model(**inputs).sum().item() / batch['seq_len']

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

    return accum_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--accum-grad", type=int, default=1)
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
    launch_train(args, pipe)