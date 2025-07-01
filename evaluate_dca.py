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

debater_1_prompt = (
    "Pretend you are a movie recommender system. Here is a user's query: {user_query}.\n"
    "Below is the reasoning and recommendation list from another movie recommender system: {debater_0}.\n"
    "Evaluate the reasoning for any potential issues. Even if the reasoning is sound, the recommendations may not align well with it.\n"
    "Analyze these aspects and provide your corrected reasoning and recommendations.\n"
    "After your reasoning, write '####' to indicate the start of your recommendation list.\n"
    "Then, list EXACTLY 20 movie recommendations, each on a new line with no extra sentences." 
)

critic_prompt = (
    "You are a judge for a debate on movie recommendations for the user query: {user_query}.\n"
    "The debate between two movie recommender systems is as follows:\n"
    "Movie Recommender System 0: {debater_0}\n\n"
    "Movie Recommender System 1: {debater_1}\n\n"
    "Your task is to reflect on both movie recommender systems and comprehensively critique the reasoning and recommendations from each side.\n"
    "After providing your analysis, generate scores for each movie from both recommender systems to indicate the quality of the recommendation.\n"
    "Use the following scale: -2 for very bad, -1 for bad, 0 for neutral, 1 for good, and 2 for very good.\n"
    "Write '####' to mark the beginning of your judgment on the recommendation list.\n"
    "Then, list the 40 movies from both sides with their scores in the format: movie_name####score, each on a new line with no extra sentences." 
)


debater_0_prompt_with_exp = (
    "Pretend you are a movie recommender system. Here is a user's query {user_query}.\n"
    "When making recommendations, consider the following guidelines: {train_exp}."
    "Specifically, after writing down your reasoning, "
    "write #### to mark the beginning of your recommendation list."
    "Then, list EXACTLY 20 movie recommendations, each on a new line with no extra sentences." 
)

debater_1_prompt_with_exp = (
    "Pretend you are a movie recommender system. Here is a user's query: {user_query}.\n"
    "Below is the reasoning and recommendation list from another movie recommender system: {debater_0}.\n"
    "Evaluate the reasoning for any potential issues. Even if the reasoning is sound, the recommendations may not align well with it.\n"
    "Analyze these aspects and provide your corrected reasoning and recommendations.\n"
    "When doing evaluation and making recommendations, consider the following guidelines: {train_exp}."
    "After completing your reasoning, write '####' to indicate the start of your recommendation list.\n"
    "Then, list EXACTLY 20 movie recommendations, each on a new line with no extra sentences." 
)

critic_prompt_with_exp = (
    "You are a judge for a debate on movie recommendations for the user query: {user_query}.\n"
    "The debate between two movie recommender systems is as follows:\n"
    "Movie Recommender System 0: {debater_0}\n\n"
    "Movie Recommender System 1: {debater_1}\n\n"
    "Your task is to reflect on both movie recommender systems and comprehensively critique the reasoning and recommendations from each side.\n"
    "After providing your analysis, generate scores for each movie from both recommender systems to indicate the quality of the recommendation.\n"
    "Use the following scale: -2 for very bad, -1 for bad, 0 for neutral, 1 for good, and 2 for very good.\n"
    "Consider the following rules when you make the judgment: {train_exp}\n\n"
    "Write '####' to mark the beginning of your judgment on the recommendation list.\n"
    "Then, list the 40 movies from both sides with their scores in the format: movie_name####score, each on a new line with no extra sentences." 
)


def retrieve_raw_exps(test_item, 
                      train_data, 
                      K):
    train_sim_idxes = test_item["train_sim_idxes"][:K]
    train_exps = [train_data[idx]["debater_0_bwd_from_llm"]['resp']['choices'][0]['message']['content'].split("####")[-1].strip()
                  for idx in train_sim_idxes]
    return train_exps


def debater1_with_exp(test_data, train_data):
    for K in K_list:
        EXSTING = {}
        threads, results = [], []
        
        for i, item in enumerate(tqdm(test_data, 
                                total=len(test_data), 
                                desc="debater 1 with exp on CIKM test data...")):  
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
            debater_0 = item[f"debater0_with_exp_from_llm_{K}"]['resp']['choices'][0]['message']['content']

            ### retrieve the training experience 
            if K != 0:
                train_exps = "\n".join(retrieve_raw_exps(item, train_data, "debater_1_bwd_from_llm", K=K))
                input_text = {
                    "user_query" : context,
                    "debater_0" : debater_0,
                    "train_exp" : train_exps
                }
                prompt = debater_1_prompt_with_exp
            else:
                input_text = {
                    "user_query" : context,
                    "debater_0"  : debater_0
                }
                prompt = debater_1_prompt
        
                        
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
                    test_data[index][f"debater1_with_exp_from_llm_{K}"] = res
                    
                threads = []
                results = []
                time.sleep(0)
        
        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()
        
        for res in results:
            index = res["index"]
            test_data[index][f"debater1_with_exp_from_llm_{K}"] = res
            

def critic_with_exp(test_data, train_data):
    for K in K_list:
        EXSTING = {}
        threads, results = [], []
        
        for i, item in enumerate(tqdm(test_data, 
                                total=len(test_data), 
                                desc="critic with exp on CIKM test data...")):  
            context = "\n".join([": ".join(rnd) for rnd in item['context_raw']])
            debater_0 = item[f"debater0_with_exp_from_llm_{K}"]['resp']['choices'][0]['message']['content']
            debater_1 = item[f"debater1_with_exp_from_llm_{K}"]['resp']['choices'][0]['message']['content']
            
            ### retrieve the training experience 
            if K != 0:
                train_exps = "\n".join(retrieve_raw_exps(item, train_data, "critic_bwd_from_llm", K=K))
                input_text = {
                    "user_query" : context,
                    "debater_0" : debater_0,
                    "debater_1" : debater_1,
                    "train_exp" : train_exps
                }
                prompt = critic_prompt_with_exp
            else:
                input_text = {
                    "user_query" : context,
                    "debater_0"  : debater_0,
                    "debater_1"  : debater_1
                }
                prompt = critic_prompt
                        
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
                    test_data[index][f"critic_with_exp_from_llm_{K}"] = res
                    
                threads = []
                results = []
                time.sleep(0)
        
        if len(threads) > 0:
            for execute_thread in threads:
                execute_thread.join()
        
        for res in results:
            index = res["index"]
            test_data[index][f"critic_with_exp_from_llm_{K}"] = res
        
        
def rating_key(movie):
    rating = movie[1]
    # Create a mapping for the ratings
    rating_map = {"2": 5, "1": 4, "0": 3, "-1": 2, "-2": 1}
    # Return the mapped value or 0 for other strings
    return rating_map.get(rating, 0)


def process_rec_critic(item, K):
    raw_rec_list = item[f'critic_with_exp_from_llm_{K}']['resp']['choices'][0]['message']['content']
    raw_rec_list = raw_rec_list.split('####', 1)[-1]
    raw_rec_list = re.sub(r'\n+', '\n', raw_rec_list)
    try:
        raw_rec_list = [reflect.strip().split("####") for reflect in raw_rec_list.split('\n')]
        raw_rec_list = [item for item in raw_rec_list if len(item) == 2]
        # Sort the list using the custom key function, maintaining original order for same ratings
        raw_rec_list = sorted(raw_rec_list, key=rating_key, reverse=True)
        # Extract the movie names from the sorted list
        raw_rec_list = [movie[0] for movie in raw_rec_list]     
        raw_rec_list = [del_numbering(del_space(del_parentheses(remove_quotes(i).strip()))) for i in raw_rec_list]
        raw_rec_list = remove_duplicates([remove_quotes(name) for name in raw_rec_list])
        item[f"critic_rec_{K}"] = raw_rec_list
        error = False
    except:
        item[f"critic_rec_{K}"] = []
        error = True
    return error, item


if __name__ == '__main__':
    ### Obtain the dataset
    train_pkl = f"data/train.pkl"
    test_pkl = f"data/test.pkl"
    
    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    with open(test_pkl, "rb") as f:
        test_data = pickle.load(f)
        
    debater1_with_exp(test_data, train_data)
    critic_with_exp(test_data, train_data)
    
    for K in K_list:
        test_data = [process_rec_critic(item, K) for item in test_data]
        errors = [item[0] for item in test_data if item[0]]
        test_data = [item[1] for item in test_data]
        print(f"# errors: {len(errors)}")

    k_list = [1,2,5,10,15,20]
    metrics_vague = {}
    avg_metrics_vague = {}

    errors = set()
    results = {k: {K: [] for K in K_list} for k in k_list}

    for k in k_list:
        print(f"Processing top {k}")
        for K in K_list:
            for i, item in enumerate(tqdm(test_data, total=len(test_data), desc=f"Evaluating critic_rec_{K}...")):
                try:
                    results[k][K].append(evaluate_direct_match(item, k, rec_field=f"critic_rec_{K}", gt_field="groundtruth"))
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
