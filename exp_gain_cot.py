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


debater_0_reflect_prompt = (
    "You are evaluating a movie recommender system.\n"
    "Assess the reasoning and recommended movies based on the user query: {user_query}.\n"
    "Here is the reasoning and recommended movies from the system: {debater_0}.\n"
    "Here are the ground truth movies the user wants to watch: {gt_items}.\n"
    "Determine if the reasoning and recommendations are successful by checking the consistency with the ground truth movies and the overlap with recommended movies.\n"
    "First, provide your judgment: success/failure, followed by '####'.\n"
    "Next, analyze why the reasoning/recommendations succeed or fail based on the user query and ground truth movies, followed by '####'.\n"
    "Finally, summarize general guidelines for making movie recommendations for similar user queries, based on your analysis."
)


def forward_pass_cot(train_sim_idxes, train_data):
    EXSTING = {}
    threads, results = [], []

    for _, i in enumerate(tqdm(train_sim_idxes, 
                            total=len(train_sim_idxes), 
                            desc="forward pass for debater 0...")):   
        item = train_data[i]
        context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])

        input_text = {
            "user_query" : context
        }
        prompt = debater_0_prompt
                    
        execute_thread = threading.Thread(
            target=get_response,
            args=(i, input_text, prompt, llm_model, temperature, max_tokens, results, EXSTING)
        )
        
        time.sleep(0.02)
        threads.append(execute_thread)
        execute_thread.start()
        if len(threads) == n_threads:
            for execute_thread in threads:
                execute_thread.join()

            for res in results:
                index = res["index"]
                train_data[index][f"debater_0_fwd_from_llm"] = res
                
            threads = []
            results = []
            time.sleep(0)

    if len(threads) > 0:
        for execute_thread in threads:
            execute_thread.join()

    for res in results:
        index = res["index"]
        train_data[index][f"debater_0_fwd_from_llm"] = res
    

def backward_pass_cot(train_sim_idxes, train_data):
    EXSTING = {}
    threads, results = [], []

    for _, i in enumerate(tqdm(train_sim_idxes, 
                            total=len(train_sim_idxes), 
                            desc="backward pass for debater 0...")):   
        item = train_data[i]
        context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
        debater_0 = item["debater_0_fwd_from_llm"]['resp']['choices'][0]['message']['content']
        gt_items = "\n".join(item['groundtruth'])

        input_text = {
            "user_query" : context,
            "debater_0"  : debater_0,
            "gt_items"   : gt_items
        }
        prompt = debater_0_reflect_prompt
                    
        execute_thread = threading.Thread(
            target=get_response,
            args=(i, input_text, prompt, llm_model, temperature, max_tokens, results, EXSTING)
        )
        
        time.sleep(0.02)
        threads.append(execute_thread)
        execute_thread.start()
        if len(threads) == n_threads:
            for execute_thread in threads:
                execute_thread.join()

            for res in results:
                index = res["index"]
                train_data[index][f"debater_0_bwd_from_llm"] = res
                
            threads = []
            results = []
            time.sleep(0)

    if len(threads) > 0:
        for execute_thread in threads:
            execute_thread.join()

    for res in results:
        index = res["index"]
        train_data[index][f"debater_0_bwd_from_llm"] = res


if __name__ == '__main__':
    ### Obtain the dataset
    train_pkl = f"data/train.pkl"
    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    
    save_dir = os.path.join("data", "processed")
    train_sim_dix_file = os.path.join(save_dir, "train_sim_idx.pkl")
    with open(train_sim_dix_file, "wb") as f:
        train_sim_idxes = pickle.load(f)
    
    forward_pass_cot(train_data, train_sim_idxes)
    backward_pass_cot(train_data, train_sim_idxes)
    
    ### Save the gained experience
    with open(train_pkl, "wb") as f:
        pickle.dump(train_data, f)