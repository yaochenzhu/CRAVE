import re
import os
import sys
import time
import json
import pickle
import threading
import multiprocessing

from pprint import pprint
from copy import deepcopy
from functools import partial
from collections import defaultdict

import nflx_copilot as ncp
from tqdm.autonotebook import tqdm
from editdistance import eval as distance

import torch
import transformers
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append('../lib')
from utils import del_parentheses, del_space, del_numbering, remove_duplicates
from utils import remove_quotes, extract_movie_name, nearest, TitleInfo
from metrics import evaluate_direct_match
from model import get_response

K_list = [0,2,5,10,15]
temperature = 0.0
max_tokens = 8192
n_threads = 500
n_print = 100
n_samples = -1
llm_model = "gpt-4o"

debater_0_prompt = (
    "Pretend you are a movie recommender system. Here is a user's query {user_query}.\n"
    "Specifically, after writing down your reasoning, "
    "write #### to mark the beginning of your recommendation list."
    "Then, list EXACTLY 20 movie recommendations, each on a new line with no extra sentences." 
)

debater_0_prompt_with_exp = (
    "Pretend you are a movie recommender system. Here is a user's query {user_query}.\n"
    "When making recommendations, consider the following guidelines: {train_exp}."
    "Specifically, after writing down your reasoning, "
    "write #### to mark the beginning of your recommendation list."
    "Then, list EXACTLY 20 movie recommendations, each on a new line with no extra sentences." 
)

def retrieve_raw_exps(test_item, 
                      train_data, 
                      K):
    train_sim_idxes = test_item["train_sim_idxes"][:K]
    train_exps = [train_data[idx]["debater_0_bwd_from_llm"]['resp']['choices'][0]['message']['content'].split("####")[-1].strip()
                  for idx in train_sim_idxes]
    return train_exps


def debater0_with_exp(test_data, train_data):
    for K in K_list:
        EXSTING = {}
        threads, results = [], []

        for i, item in enumerate(tqdm(test_data, 
                                total=len(test_data), 
                                desc="zero-shot recommendation with exp on CIKM test data...")):   
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
        
            ### retrieve the training experience 
            if K != 0:
                train_exps = "\n".join(retrieve_raw_exps(item, train_data, K=K))
                input_text = {
                    "user_query" : context,
                    "train_exp"  : train_exps
                }
                prompt = debater_0_prompt_with_exp
            else:
                input_text = {
                    "user_query" : context,
                }          
                prompt = debater_0_prompt
                        
            execute_thread = threading.Thread(
                target=get_response,
                args=(i, input_text, prompt, llm_model, temperature, max_tokens, results, EXSTING)
            )
            
            time.sleep(0.03 + max(K/5-1,0)*0.12)
            threads.append(execute_thread)
            execute_thread.start()
            if len(threads) == n_threads:
                for execute_thread in threads:
                    execute_thread.join()
        
                for res in results:
                    index = res["index"]
                    test_data[index][f"debater0_with_exp_from_llm_{K}"] = res
                    
                threads = []
                results = []
                time.sleep(0)
        
        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()
        
        for res in results:
            index = res["index"]
            test_data[index][f"debater0_with_exp_from_llm_{K}"] = res
        
        
def process_item_debater0(item, K):
    raw_rec = item[f'debater0_with_exp_from_llm_{K}']['resp']['choices'][0]['message']['content']
    raw_rec = raw_rec.split("####")[-1].strip()   
    ### Remove extra lines
    raw_rec = re.sub(r'\n+', '\n', raw_rec)
    ### Standardize movie names
    raw_rec_list = [del_numbering(del_space(del_parentheses(remove_quotes(i).strip()))) for i in raw_rec.split('\n')]
    ### Remove the remaining quotes
    raw_rec_list = remove_duplicates([remove_quotes(name) for name in raw_rec_list])
    item[f"debater0_rec_{K}"] = raw_rec_list
    return item


if __name__ == '__main__':
    ### Obtain the dataset
    train_pkl = f"data/train.pkl"
    test_pkl = f"data/test.pkl"
    
    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    with open(test_pkl, "rb") as f:
        test_data = pickle.load(f)
        
    debater0_with_exp(test_data, train_data)

    for K in K_list:
        test_data = [process_item_debater0(item, K) for item in test_data]

    k_list = [1,2,5,10,15,20]
    metrics_vague = {}
    avg_metrics_vague = {}

    errors = set()
    results = {k: {K: [] for K in K_list} for k in k_list}

    for k in k_list:
        print(f"Processing top {k}")
        for K in K_list:
            for i, item in enumerate(tqdm(test_data, total=len(test_data), desc=f"Evaluating debater0_rec_{K}...")):
                try:
                    results[k][K].append(evaluate_direct_match(item, k, rec_field=f"debater0_rec_{K}", gt_field="groundtruth"))
                except:
                    errors.add(i)

    recalls_dir = {k: {K: [res[0] for res in results[k][K]] for K in K_list} for k in k_list}
    ndcgs_dir = {k: {K: [res[1] for res in results[k][K]] for K in K_list} for k in k_list}

    avg_recalls_dir = {k: {K: np.mean(recalls_dir[k][K]) for K in K_list} for k in k_list}
    avg_ndcgs_dir = {k: {K: np.mean(ndcgs_dir[k][K]) for K in K_list} for k in k_list}

    print(f"number of errors: {len(errors)}")

    metrics_vague["dir"] = {"recall": recalls_dir, "ndcg": ndcgs_dir}
    avg_metrics_vague["dir"] = {"recall": avg_recalls_dir, "ndcg": avg_ndcgs_dir}

    print(avg_metrics_vague)