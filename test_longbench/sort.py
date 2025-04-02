import argparse
import os
from rouge_score.rouge_scorer import RougeScorer
import json
import numpy as np
import tqdm


ALL_DATA_LIST = [
    "narrativeqa",
    "qasper", 
    "multifieldqa_en", 
    "multifieldqa_zh", 
    "hotpotqa", 
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report", 
    "qmsum", 
    "multi_news", 
    "vcsum", 
    "trec", 
    "triviaqa", 
    "samsum", 
    "lsht",
    "passage_count", 
    "passage_retrieval_en", 
    "passage_retrieval_zh", 
    "lcc", 
    "repobench-p"
]


def get_file_list(dir):
    assert os.path.exists(dir), f"{dir} dost not exists!"
    lst = list(filter(lambda x: isinstance(x, str) and x.endswith(""), os.listdir(dir)))
    if 'result' in lst:
        lst.remove('result')
    return lst


def sort_lst(lst):
    data_order = {}
    for i, data in enumerate(ALL_DATA_LIST):
        data_order[data] = i
    return sorted(lst, key=lambda x: data_order[x])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str)
    args = parser.parse_args()

    with open(os.path.join('pred', args.script, 'result.json'), 'r') as f:
        result = json.loads(f.read())
    
    for data in ALL_DATA_LIST:
        if data in result.keys():
            print(f"{data:20} {result[data]}")
