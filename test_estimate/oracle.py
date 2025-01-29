from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import torch


from corpus import get_processor, LazyRandomSampleCorpus
from solos.utils import (
    get_model_and_tokenizer, 
    get_env_conf, 
    get_torch_dtype,
    get_optimizer_and_lr_adjuster, 
    SecoCache,
    chunkize,
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

    # others
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--accum-grad", type=int, default=1)

    args = parser.parse_args()

    
    env_conf = get_env_conf(args.env_conf)
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(0)

    model.train()


    params = model.ft_params()
    optimizer, lr_adjuster = get_optimizer_and_lr_adjuster(**env_conf['train'], params=params)

    for param in params:
        if param.sum().item() == 0:
            param.data = torch.randn_like(param) * 0.001


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

    batch = next(iter(loader))
    grads = []

    for _ in range(args.sample):
        history.init()

        kv_cache = SecoCache(model.num_layers)
        accum_loss = 0

        input_ids = list(chunkize(batch['input_ids'], -1, args.chunk_size))
        labels = list(chunkize(batch['labels'], -1, args.chunk_size))

        for i, (chunk_input_ids, chunk_labels) in enumerate(zip(input_ids, labels)):
            loss = model(
                input_ids=chunk_input_ids,
                labels=chunk_labels,
                kv_cache=kv_cache,)
            accum_loss += loss.sum() / batch['seq_len']

        accum_loss.backward()

        grad = [param.grad.data.clone() if param.grad is not None else torch.zeros_like(param) for param in params]
        grads.append(grad)
        zero_grad(params)

    grads = [torch.cat([y.ravel() for y in x]) for x in grads]
    grads = torch.stack(grads)

    if not os.path.exists("grads"):
        os.mkdir("grads")
    
    torch.save(grads, "grads/oracle.pth")


    backend_cleanup()
