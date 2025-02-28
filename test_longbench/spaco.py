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
    SecoCache,
    History)

import argparse, random, numpy, os, json
from pygments.console import colorize
from functools import partial


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


def collate_fn(batch, budget=16, chunk_size=128):
    batch[0]['labels'] = batch[0]['labels'][1:] + [-100]
    batch[0]['assistant_masks'] = batch[0]['assistant_masks'][1:] + [False]

    input_ids = torch.tensor(batch[0]['input_ids'], dtype=torch.int64, device='cuda').unsqueeze(0)
    labels = torch.tensor(batch[0]['labels'], dtype=torch.int64, device='cuda').unsqueeze(0)
    assistant_masks = torch.tensor(batch[0]['assistant_masks'], dtype=torch.int64, device='cuda')

    # 计算有多少chunks包含assistant回答
    chunk_masks = list(chunkize(assistant_masks, dim=0, chunk_size=chunk_size))
    has_assistant = [x.sum().item() > 0 for x in chunk_masks]
    num_assistant_chunks = sum(has_assistant)
    num_question_chunks = len(has_assistant) - num_assistant_chunks

    # 初始化两部分的budget
    assistant_budget = budget // 2
    question_budget = budget // 2

    # 将assistant部分匀出来？
    if assistant_budget > num_assistant_chunks:
        question_budget += assistant_budget - num_assistant_chunks
        assistant_budget = num_assistant_chunks

    # 是否将question部分的匀出来？
    if question_budget > num_question_chunks:
        assistant_budget += question_budget - num_question_chunks
        question_budget = num_question_chunks

    # 提取对应部分的indices
    has_assistant = torch.tensor(has_assistant, dtype=torch.bool, device='cuda')
    indices = torch.arange(has_assistant.numel(), device='cuda')
    question_indices = indices[~has_assistant]
    assistant_indices = indices[has_assistant]

    question_indices = torch.gather(question_indices, dim=0, index=torch.randperm(question_indices.numel(), device='cuda')[:question_budget])
    assistant_indices = torch.gather(assistant_indices, dim=0, index=torch.randperm(assistant_indices.numel(), device='cuda')[:assistant_budget])
    I = torch.cat([question_indices, assistant_indices], dim=0).tolist()
    I = sorted(I)

    return dict(
        input_ids=input_ids,
        labels=labels,
        seq_len=assistant_masks.sum().item(),
        indices=I)


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
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-budget", type=int, default=16)

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
    env_conf['model']['device_map'] = {"": dist.get_rank()}
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
        collate_fn=partial(collate_fn, budget=args.chunk_budget, chunk_size=args.chunk_size),
        shuffle=True)


    base_memory_allocated = torch.cuda.max_memory_allocated()
    print(colorize("yellow", "Base GPU memory allocated:") + colorize("green", f"{base_memory_allocated // 1024 ** 2} MB"))
    history = History(args.log_step)


    for step, batch in enumerate(loader):
        lr_adjuster(step=step)

        # preprocess, segmentation
        input_ids = list(chunkize(batch['input_ids'], -1, args.chunk_size))
        labels = list(chunkize(batch['labels'], -1, args.chunk_size))

        # initialization
        kv_cache = SecoCache(model.num_layers)
        history.init()

        loss_accum = 0

        with torch.no_grad():
            for chunk_input, chunk_target in zip(input_ids, labels):

                # forward pass
                inputs = dict(
                    input_ids=chunk_input,
                    labels=chunk_target,
                    kv_cache=kv_cache)
                loss = model(**inputs).sum() / batch['seq_len']
                loss_accum += loss.item()

        I = batch['indices']

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
                tmp_kv_cache.index(i).copy_scaled_grad(gd=kv_cache.index(i).grad)

                # backward prop
                loss.backward()


        history.step(loss_accum, batch['seq_len'])

        if (step + 1) % args.accum_grad == 0:
            optimizer.step()
            zero_grad(params)

    output = json.dumps(history.loss)
    print(output)


    # save model
    if args.save_ckpt is not None:
        model.save_checkpoint(args.save_ckpt)


    backend_cleanup()
