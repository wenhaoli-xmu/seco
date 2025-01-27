from myabc.utils import get_model_and_tokenizer, get_env_conf, show_masked_string
from corpus import get_processor, LazyRandomSampleCorpus
import argparse
import torch
import random
import numpy as np
import json
from corpus.processor.conversations import get_conv_template
from rouge_score.rouge_scorer import RougeScorer
from pygments.console import colorize



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
        self.scores = []

    def update(self, score):
        self.scores.append(score)

    def summary(self):
        results = dict(score=np.mean(self.scores))
        print(json.dumps(results, indent=4))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, required=True)
    parser.add_argument("--chat_template", type=str, required=True)
    parser.add_argument("--enable_prune", action='store_true')
    args = parser.parse_args()

    seed_everything(42)

    env_conf = get_env_conf(args.env_conf)
    model, tokenizer = get_model_and_tokenizer(**env_conf['model'])

    scorer = RougeScorer(['rougeL'], use_stemmer=True)

    prompt = """
        Copy the following paragraph, do not change any single word: {text}.
    """

    texts = []
    with open("copy/texts.jsonl", 'r') as f:
        for line in f:
            texts.append(json.loads(line)['text'])

    history = History()

    for text in texts:

        prompted_text = prompt.format(text=text)
        conv = get_conv_template(args.chat_template)
        conv.append_message(conv.roles[0], prompted_text)
        conv.append_message(conv.roles[1], None)
        prompted_text = conv.get_prompt()

        input_ids = tokenizer(prompted_text, return_tensors='pt').input_ids
        output_ids, mask = model.generate(input_ids, enable_prune=args.enable_prune)

        length = input_ids.shape[-1]
        output_ids = output_ids[:,length:].ravel().tolist()

        pred = tokenizer.decode(output_ids)

        print(colorize("yellow", "### origin"))
        print(text)

        if args.enable_prune:
            print(colorize("yellow", "### mask"))
            print(show_masked_string(input_ids.ravel().tolist(), tokenizer, mask))

        print(colorize("yellow", "### copy"))
        print(pred)
        print(colorize("yellow", "=" * 80))

        score = scorer.score(target=text, prediction=pred)['rougeL'].precision
        history.update(score)

    history.summary()
