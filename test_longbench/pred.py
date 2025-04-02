import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from chunkoptim.utils import get_model_and_tokenizer
from corpus import get_processor


# modified: 将参数max_length去掉
def get_pred(
        tokenizer, 
        model,
        data, 
        max_gen, 
        prompt_format, 
        dataset, 
        device, 
        out_path, 
        model_max_length,
        processor):

    
    for json_obj in tqdm(data):

        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        
        # =====================================================================================================================================================
        # NOTE: 注释掉
        # tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model_name:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        # if len(tokenized_prompt) > max_length:
        #     half = int(max_length/2)
        #     prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # =====================================================================================================================================================

        # 设置 safe-gap
        safe_gap = 256
        max_prompt_tokens = model_max_length - max_gen - safe_gap
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]
            prompt = tokenizer.decode(prompt_ids)


        # tokenization
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            conversation = [{"role": "user", "content": prompt}]
            prompt = processor(conversation, add_generation_prompt=True)
            input_ids = torch.tensor(prompt['input_ids'], dtype=torch.int64).to(device)
            context_length = input_ids.numel()

        else:
            input_ids = tokenizer(prompt, truncation=False, return_tensors='pt').input_ids.to(device)
            context_length = input_ids.numel()


        # =================================================================================================
        # NOTE: 注释掉
        # if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
        #     output = model.generate(
        #         **input,
        #         max_new_tokens=max_gen,
        #         num_beams=1,
        #         do_sample=False,
        #         temperature=1.0,
        #         min_length=context_length+1,
        #         eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
        #     )[0]
        # else:
        # =================================================================================================


        # ============================================================================
        # NOTE: 新增加
        output = model.generate(
            input_ids=input_ids,
            tokenizer=tokenizer,
            max_new_tokens=max_gen
        ).ravel().tolist()

        # NOTE: 新增加
        # if tokenizer.eos_token_id in output:
        #     index = output.index(tokenizer.eos_token_id)
        #     output = output[:index]
        torch.cuda.empty_cache()

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        # ============================================================================


        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--config", type=str, default="config/longbench.json")
    parser.add_argument("--max_gen", type=int, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--load-ckpt", type=str, default=None)
    args = parser.parse_args()

    import json, os
    with open(args.env_conf, "r") as f:
        env_conf = json.load(f)
    with open("test_longbench/pred.json", 'r') as f:
        pred_conf = json.load(f)

    seed_everything(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    for dataset in pred_conf:
        assert dataset in datasets
    datasets = pred_conf

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    model, tokenizer = get_model_and_tokenizer(**env_conf["model"])
    processor = get_processor(path=args.config, tokenizer=tokenizer)

    if args.load_ckpt:
        model.load_checkpoint(args.load_ckpt)

    model_name = args.load_ckpt.split('/')[-1].replace(".pth", "")

    for dataset in datasets:

        if args.e:
            data = load_dataset('LongBench/LongBench.py', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"

        else:
            data = load_dataset('LongBench/LongBench.py', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        get_pred(
            tokenizer,
            model,
            data_all, 
            max_gen if args.max_gen is None else args.max_gen, 
            prompt_format, 
            dataset, 
            device, 
            out_path, 
            args.model_max_length,
            processor)
