import os
import pickle

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

sys.path.append("lib")
from retriever import extract_context, ModelWithClassificationHead
from retriever import build_datastore_embedding, calculate_hit_rate


# Define a prompt for the training data
train_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "


if __name__ == '__main__':
    ### Obtain the dataset
    train_pkl = f"data/train.pkl"
    val_pkl = f"data/val.pkl"
    test_pkl = f"data/test.pkl"

    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    print(f"Number of train samples: {len(train_data)}")

    with open(val_pkl, "rb") as f:
        val_data = pickle.load(f)
    with open(test_pkl, "rb") as f:
        test_data = pickle.load(f)
    print(f"Number of val/test samples: {len(test_data)}")

    # Create a global ID mapping for all unique groundtruth items
    all_groundtruth_items = set()
    for item in train_data + val_data + test_data:
        all_groundtruth_items.update(item['groundtruth'])

    groundtruth_id_map = {item: idx for idx, item in enumerate(all_groundtruth_items)}
    num_classes = len(groundtruth_id_map)

    # Prepare training dataset with prompts
    train_contexts = []
    train_labels = []
    for item in train_data:
        context = extract_context(item)
        # Prepend the prompt to the context
        context_with_prompt = train_prompt + context
        train_contexts.append(context_with_prompt)
        # Create a multi-hot encoded label vector
        label_vector = torch.zeros(len(groundtruth_id_map))
        for gt in item['groundtruth']:
            label_vector[groundtruth_id_map[gt]] = 1
        train_labels.append(label_vector)

    val_contexts = []
    for item in val_data:
        context = extract_context(item)
        val_contexts.append(context)

    test_contexts = []
    for item in test_data:
        context = extract_context(item)
        test_contexts.append(context)

    # Load model and tokenizer from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('stella_en_400M_v5', trust_remote_code=True)
    base_model = AutoModel.from_pretrained('stella_en_400M_v5', trust_remote_code=True).to("cuda")

    # Define model with head
    model_with_head = ModelWithClassificationHead(base_model, num_classes, groundtruth_id_map, tokenizer)
        
    # Load the best model weights
    if torch.cuda.device_count() > 1:
        model_with_head = nn.DataParallel(model_with_head)
        
    # Load the state dictionary
    best_model_path = "models/best_model.pth"

    if isinstance(model_with_head, nn.DataParallel):
        model_with_head.module.load_state_dict(torch.load(best_model_path))
    else:
        model_with_head.load_state_dict(torch.load(best_model_path))

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_with_head.to(device)

    # Evaluate the model at the end of each epoch
    hit_rate = calculate_hit_rate(val_data, train_data, train_contexts, val_contexts, model_with_head, groundtruth_id_map, top_k=5)
    print(f"Hit Rate - Val: {hit_rate:.4f}")
    

    # Calculate embeddings
    if isinstance(model_with_head, nn.DataParallel):
        base_model = model_with_head.module.base_model
        vector_linear = model_with_head.module.vector_linear
    else:
        base_model = model_with_head.base_model
        vector_linear = model_with_head.vector_linear
    train_embeddings = build_datastore_embedding(train_contexts, [base_model, vector_linear], tokenizer, device)


    # Extract contexts from test_data_vague
    test_contexts = ["\n".join([": ".join(rnd) for rnd in item['context_raw']]) for item in test_data]
    # Calculate embeddings for test_data_vague
    test_embeddings = build_datastore_embedding(test_contexts, [base_model, vector_linear], tokenizer, device)
    
    # Function to find the top N most similar training samples
    def top_n_similar_indices(query_embedding, post_embeddings, top_n=30):
        similarities = (query_embedding @ post_embeddings.T).squeeze().cpu().numpy()
        sorted_indices = np.argsort(similarities)[::-1]
        return sorted_indices[:top_n]

    print("Finding relevant training samples...")
    # Calculate and add top 30 similar indices for each test item
    for i, test_embedding in tqdm(enumerate(test_embeddings), total=len(test_embeddings)):
        test_embedding = test_embedding.reshape(1, -1)
        top_indices = top_n_similar_indices(test_embedding, train_embeddings, top_n=15)
        test_data[i]["train_sim_idxes"] = top_indices.tolist()

    train_sim_idxes = set()
    for item in test_data:
        train_sim_idxes.update(item["train_sim_idxes"])
    train_sim_idxes = sorted(train_sim_idxes)
    print(f"#Unique train idxes: {len(train_sim_idxes)}")
        
    # Save the results
    save_dir = os.path.join("data", "processed")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    test_file = os.path.join(save_dir, "test.pkl")
    with open(test_file, "wb") as f:
        pickle.dump(test_data, f)
        
    train_sim_dix_file = os.path.join(save_dir, "train_sim_idx.pkl")
    with open(train_sim_dix_file, "wb") as f:
        pickle.dump(train_sim_idxes, f)
        