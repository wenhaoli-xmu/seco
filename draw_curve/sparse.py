from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import torch


from corpus import get_processor, LazyRandomSampleCorpus
from chunkoptim.utils import (
    get_model_and_tokenizer, 
    get_env_conf, 
    get_torch_dtype,
    get_optimizer_and_lr_adjuster, 
    chunkize,
    NestCache,
    build_mixed_float_mask,
    History)

import argparse, random, numpy, os, json
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
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
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


def prepare_for_sparse_backward(input_ids, labels, chunk_size, chunk_budget):
    assert input_ids.shape[0] == 1
    seq_len = input_ids.shape[-1]

    token_budget = chunk_size * chunk_budget
    token_budget = min(token_budget, seq_len)

    indices = torch.randperm(seq_len, device='cuda')[:token_budget].sort().values
    indices = indices.unsqueeze(0)

    input_ids = torch.gather(input_ids, 1, indices)
    labels = torch.gather(labels, -1, indices)

    assert input_ids.ndim == 2

    indices = chunkize(indices, -1, args.chunk_size)
    input_ids = chunkize(input_ids, -1, args.chunk_size)
    labels = chunkize(labels, -1, args.chunk_size)

    return indices, input_ids, labels, token_budget
    


def backend_cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':


    backend_setup()


    parser = argparse.ArgumentParser()
    parser.add_argument("--env-conf", type=str, required=True)
    
    # algorithm related arguments
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-budget", type=int, default=1)

    # others
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--accum-grad", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    
    env_conf = get_env_conf(args.env_conf)
    env_conf['train']['max_lr'] = args.lr
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(args.seed)

    model.eval()

    params = model.ft_params()
    optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(**env_conf['train'], params=params)


    # build dataset
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
    history = History(args.log_step)

    for step, batch in enumerate(loader):
        lr_adjuster(step=step)

        input_ids, labels = batch['input_ids'], batch['labels']
        input_ids_fwd = list(chunkize(input_ids, -1, args.chunk_size))
        labels_fwd = list(chunkize(labels, -1, args.chunk_size))
        kv_cache = NestCache(model.num_layers, cpu_offload=1, seq_dim=1)

        history.init()
        accum_loss = 0
        
        with torch.no_grad():
            for chunk_input, chunk_target in zip(input_ids_fwd, labels_fwd):
                inputs = dict(
                    input_ids=chunk_input,
                    labels=chunk_target,
                    kv_cache=kv_cache)
                loss = model(**inputs).sum() / batch['seq_len']
                accum_loss += loss.item()

        # prepare for sparse backward
        indices, input_ids_bwd, labels_bwd, budget = prepare_for_sparse_backward(
            batch['input_ids'], 
            batch['labels'], 
            args.chunk_size, 
            args.chunk_budget)

        for chunk_indices, chunk_input, chunk_target in reversed(list(zip(indices, input_ids_bwd, labels_bwd))):

            mixed_mask = build_mixed_float_mask(batch['seq_len'], chunk_indices)

            inputs = dict(
                input_ids=chunk_input,
                labels=chunk_target,
                kv_cache=kv_cache,
                attention_mask=mixed_mask)

            loss = model(**inputs).sum() / budget

            kv_cache.pre_backward(chunk_indices)
            loss.backward()
            kv_cache.after_backward()

        history.step(accum_loss, batch['seq_len'])

        if (step + 1) % args.accum_grad == 0:
            mag = 0
            for param in params:
                mag += param.grad.abs().mean()
            print(f'magnitude: {mag}')
            optimizer.step()
            zero_grad(params)

    output = json.dumps(history.loss)
    print(output)

    backend_cleanup()
