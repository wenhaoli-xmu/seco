from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist

import torch


from corpus import get_processor, LazyRandomSampleCorpus
from chunkoptim.utils import (
    get_model_and_tokenizer, 
    get_env_conf, 
    get_torch_dtype,
    get_optimizer_and_lr_adjuster,
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
    batch[0]['labels'] = batch[0]['labels'][1:] + [-100]
    batch[0]['assistant_masks'] = batch[0]['assistant_masks'][1:] + [False]

    input_ids = batch[0]['input_ids']
    labels = batch[0]['labels']
    assistant_masks = batch[0]['assistant_masks']

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda')
    labels = torch.tensor(labels, dtype=torch.int64, device='cuda')

    input_ids = input_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)

    seq_len = sum(assistant_masks)

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
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--accum-grad", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--load-ckpt", type=str, default=None)
    parser.add_argument("--save-ckpt", type=str, default=None)

    args = parser.parse_args()

    
    env_conf = get_env_conf(args.env_conf)
    env_conf['train']['max_lr'] = args.lr
    dtype = get_torch_dtype(env_conf['model']['model_dtype'])


    # load model
    seed_everything(0)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    seed_everything(args.seed)

    model.eval()

    if args.load_ckpt is not None:
        model.load_checkpoint(args.load_ckpt)


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
        collate_fn=collate_fn,
        shuffle=True)


    base_memory_allocated = torch.cuda.max_memory_allocated()
    print(colorize("yellow", "Base GPU memory allocated:") + colorize("green", f"{base_memory_allocated // 1024 ** 2} MB"))
    history = History(args.log_step)

    for step, batch in enumerate(loader):
        lr_adjuster(step=step)

        input_ids = batch['input_ids']
        labels = batch['labels']

        # initialize history
        history.init()
        
        # forward pass & backward pass
        loss = model(
            input_ids=batch['input_ids'],
            labels=batch['labels'])
        loss = loss.sum() / batch['seq_len']
        loss.backward()

        # log history
        history.step(loss.item(), batch['seq_len'])

        if (step + 1) % args.accum_grad == 0:
            optimizer.step()
            zero_grad(params)

    output = json.dumps(history.loss)
    print(output)


    # save model
    if args.save_ckpt is not None:
        model.save_checkpoint(args.save_ckpt)


    backend_cleanup()
