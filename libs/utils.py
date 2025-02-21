import os
import re
import json
import regex
import pickle

import datetime
from time import sleep

import hashlib
import numpy as np

from collections import defaultdict
from editdistance import eval as distance

from rank_bm25 import BM25Okapi

def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)

def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()

def del_numbering(text):
    pattern = r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?"
    return re.sub(pattern, "", text)

def is_in(text, items, threshold):
    for i in items:
        if (distance(i.lower(), text.lower()) <= threshold):
            return True
    return False

def nearest(text, items):
    """ given the raw text name and all candidates, 
        return {movie_name:, min_edit_distance: , nearest_movie: }
    """
    # calculate the edit distance
    items = list(set(items))
    dists = [distance(text.lower(), i.lower()) for i in items]
    # find the nearest movie
    nearest_idx = np.argmin(dists)
    nearest_movie = items[nearest_idx]
    return {
        'movie_name': text, 
        'min_edit_distance': dists[nearest_idx], 
        'nearest_movie': nearest_movie
    }

def nearest_thres(text, items, thres):
    """ given the raw text name and all candidates, 
        return {movie_name:, min_edit_distance: , nearest_movie: }
    """
    # calculate the edit distance
    items = list(set(items))
    dists = [distance(text.lower(), i.lower()) for i in items]
    # find the nearest movie
    nearest_idx = np.argmin(dists)
    nearest_movie = items[nearest_idx]

    if dists[nearest_idx] <= thres:
        return {
            'movie_name': text, 
            'min_edit_distance': dists[nearest_idx], 
            'nearest_movie': nearest_movie
        }
    else:
        return None

def extract_movie_name(text):
    text = text.split('/')[-1]
    text = text.replace('_', ' ').replace('-', ' ').replace('>', ' ')
    return del_space(del_parentheses(text))

def remove_quotes(s):
    if (s.startswith('"') and s.endswith('"')) \
        or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def recall_score(gt_list, pred_list, ks, threshold, verbose=False):
    hits = defaultdict(list)
    for gt, preds in zip(gt_list, pred_list):
        for k in ks:
            hits[k].append(int(is_in(gt, preds[:k], threshold)))
    if verbose:
        for k in ks:
            print("Recall@{}: {:.4f}".format(k, np.mean(hits[k])))
    return hits
    

def mrr_score(gt_list, pred_list, ks, threshold, verbose=False):
    mrrs = defaultdict(list)
    for gt, preds in zip(gt_list, pred_list):
        for k in ks:
            for i, p in enumerate(preds[:k]):
                if is_in(gt, [p], threshold):
                    mrrs[k].append(1 / (i + 1))
                else:
                    mrrs[k].append(0)
    if verbose:
        for k in ks:
            print("MRR@{}: {:.4f}".format(k, np.mean(mrrs[k])))
    return mrrs

def ndcg_score(gt_list, pred_list, ks, threshold, verbose=False):
    ndcgs = defaultdict(list)
    for gt, preds in zip(gt_list, pred_list):
        for k in ks:
            for i, p in enumerate(preds[:k]):
                if is_in(gt, [p], threshold):
                    ndcgs[k].append(1 / np.log2(i + 2))
                    break
    if verbose:
        for k in ks:
            print("NDCG@{}: {:.4f}".format(k, np.mean(ndcgs[k])))
    return ndcgs


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


class TitleInfo():
    def __init__(self,title_info):
        self.title_info=title_info
        self.title_names = title_info[ "title_name"].to_numpy()
        self.title_importance = title_info[ "importance_tier"].to_numpy()
        self.title_imdb_id = title_info[ "imdb_id"].to_numpy()
        self.tokenized_title_names = [  self.preprocess_phrase(doc).split(" ") for doc in self.title_names  ]
        self.bm25 = BM25Okapi(self.tokenized_title_names)

        self.imdb_id2ix_dict={}
        for ix,iid in  enumerate(self.title_imdb_id):
            self.imdb_id2ix_dict[iid]=ix

    def imdb_id_2_title_name(self, iid):
        if iid in self.imdb_id2ix_dict.keys(): 
            return self.title_names[  self.imdb_id2ix_dict[iid] ]
        return ""
        
    def imdb_id_2_title_importance(self, iid):
        if iid in self.imdb_id2ix_dict.keys(): 
            return self.title_importance[  self.imdb_id2ix_dict[iid] ]
        return -1        
    
    def preprocess_phrase(self,phrase):
        co1=re.sub(r"\W", " ", phrase)
        co2=re.sub(r"\s+", " ", co1)
        co3=co2.strip().lower()
        return co3

    def find_best_title_matches(self, query):
        preprocessed_query= self.preprocess_phrase(query)
        tokenized_query = preprocessed_query.split(" ")
        
        ## find good matches using BM25
        scores = self.bm25.get_scores(tokenized_query)
        kk=np.argsort(-scores)
        kk=kk[scores[kk]>0.0]
        ii=kk[:100]
        #### most words are matching
        ii_matches=[0]*len(ii)

        if not ii_matches:
            return [0, []]
            
        for c,i in enumerate(ii):
            for tt in tokenized_query:
                if tt in self.tokenized_title_names[i]: ii_matches[c]+=1
        jj=ii[np.array(ii_matches) == max(ii_matches)]
        query_match_fraction = max(ii_matches) /1.00/len(tokenized_query )
        #### re-sort best matches by title-importance
        kk = np.argsort(self.title_importance[jj])
        jj=jj[kk]
        ### put exact matches of title-names to the front
        exact=[]
        for j in jj:
            if  preprocessed_query ==  " ".join(self.tokenized_title_names[j]):
                exact.append(j)
        ii=[]
        for j in jj:
            if j not in exact:
                ii.append(j)
        jj= exact+ii
        return [query_match_fraction,  self.title_imdb_id[jj]]
    
    
def retrieve_raw_exps(test_item, 
                      train_data, 
                      K):
    train_sim_idxes = test_item["train_sim_idxes"][:K]
    train_exps = [train_data[idx]["debater_0_bwd_from_llm"]['resp']['choices'][0]['message']['content'].split("####")[-1].strip()
                  for idx in train_sim_idxes]
    return train_exps