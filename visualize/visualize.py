from myabc.utils import get_model_and_tokenizer, get_env_conf, gradient_color
from corpus import get_processor, LazyRandomSampleCorpus
import argparse
import torch
import random
import numpy as np
import json


def build_dataset(env_conf, tokenizer):
    proc = get_processor(env_conf['eval']['config'], tokenizer)
    corp = LazyRandomSampleCorpus(env_conf['eval']['file_path'], proc, max_instance=env_conf['eval']['max_instance'])
    return corp


def seed_everything(seed):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class History:
    def __init__(self):
        self.loss = []
        self.baseline = []
        self.ratio = []

    def update(self, outs, outs_baseline):
        self.loss.append(outs['loss'].item())
        self.baseline.append(outs_baseline['loss'].item())
        self.ratio.append(outs['ratio'])

    def summary(self):
        results = dict(
            loss=np.mean(self.loss),
            loss_baseline=np.mean(self.baseline),
            ratio=np.mean(self.ratio))
        print(json.dumps(results, indent=4))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str)
    args = parser.parse_args()

    seed_everything(42)

    env_conf = get_env_conf(args.env_conf)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])
    
    corpus = build_dataset(env_conf, tokenizer)

    history = History()


    for data in corpus:
        input_ids = data['input_ids']
        labels = input_ids[1:] + [-100]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cuda').unsqueeze(0)
        labels = torch.tensor(labels, dtype=torch.int64, device='cuda').unsqueeze(0)

        outs = model(input_ids=input_ids, labels=labels)
        outs_bl = model(input_ids=input_ids, labels=labels, enable_prune=False)

        history.update(outs, outs_bl)

        print(f"loss:{outs['loss'].item():.3f}, loss(baseline): {outs_bl['loss'].item():.3f}")
        for m, tok in zip(outs['mask'], data['input_ids'][1:]):
            print(gradient_color(tokenizer.decode(tok) + ' ' if tok != 13 else '\\n', int(m)), end='')
        print()
        print('=' * 40)

    history.summary()
