from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import torch


from corpus import get_processor, LazyRandomSampleCorpus
from chunkoptim.utils import (
    get_model_and_tokenizer, 
    get_env_conf, 
    get_torch_dtype,
    get_optimizer_and_lr_adjuster, 
    SecoCache,
    chunkize,
    History)

import argparse, random, numpy, os
from functools import partial
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


if __name__ == '__main__':


    backend_setup()


    parser = argparse.ArgumentParser()
    parser.add_argument("--env-conf", type=str, required=True)
    
    # algorithm related arguments
    parser.add_argument("--chunk-budget", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--context", type=str, default=None)

    # others
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--accum-grad", type=int, default=1)

    args = parser.parse_args()

    
    env_conf = get_env_conf(args.env_conf)
    env_conf['model']['device_map'] = {"": dist.get_rank()}
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    import json
    if args.context is None:
        args.context = [512] + [1024 * i for i in range(1,1025)]
    else:
        args.context = json.loads(args.context)


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(dist.get_rank())


    model.train()


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

    my_chunkize = partial(chunkize, dim=-1, chunk_size=args.chunk_size)


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

        for _ in range(10):
            input_ids = list(my_chunkize(batch['input_ids']))
            labels = list(my_chunkize(batch['labels']))
            kv_cache = SecoCache(model.num_layers)
            loss_accum = 0

            history.init()
            
            with torch.no_grad():
                for chunk_input, chunk_target in zip(input_ids, labels):

                    # forward pass
                    inputs = dict(
                        input_ids=chunk_input,
                        labels=chunk_target,
                        kv_cache=kv_cache)
                    model(**inputs)

            
            I = torch.randperm(len(input_ids))[:args.chunk_budget]


            for i, (chunk_input, chunk_target) in reversed(list(enumerate(zip(input_ids, labels)))):


                if i in I:

                    tmp_kv_cache = kv_cache.range(i)

                    # forward prop
                    inputs = dict(
                        input_ids=chunk_input,
                        labels=chunk_target,
                        kv_cache=tmp_kv_cache)
                    loss = model(**inputs).sum() / batch['seq_len']

                    # copy kv cache grad
                    tmp_kv_cache.index(i).copy_scaled_grad(
                        gd=kv_cache.index(i).grad, 
                        scaler=len(input_ids) / args.chunk_budget)

                    # backward prop
                    loss.backward()

            history.step(loss_accum, batch['seq_len'])
            del tmp_kv_cache, kv_cache, loss
            torch.cuda.empty_cache()

        mean_time, mean_memory = history.summary(False)
        template = colorize("yellow", f"{context:<5d}") + "{mean_time:<3.3f} | {mean_memory:.3f}"
        print(template.format(mean_time=mean_time, mean_memory=mean_memory))

    backend_cleanup()
