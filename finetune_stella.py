import os
import sys
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
from retriever import CustomDataset, ModelWithClassificationHead
from retriever import MultinomialLikelihoodLoss, calculate_hit_rate

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


    # Load Stella-v5 model and tokenizer
    model_dir = "model/huggingface/stella_en_400M_v5/"  # Replace with your model path
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
    

    # Load Stella model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('stella_en_400M_v5', trust_remote_code=True)
    base_model = AutoModel.from_pretrained('tella_en_400M_v5', trust_remote_code=True).to("cuda")

    # Define a prompt for the training data
    train_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "

    # Function to extract context
    def extract_context(item):
        return "\n".join([": ".join(rnd) for rnd in item['context_raw']])

    # Create a global ID mapping for all unique groundtruth items
    all_groundtruth_items = set()
    for item in train_data + val_data + test_data:
        all_groundtruth_items.update(item['groundtruth'])

    groundtruth_id_map = {item: idx for idx, item in enumerate(all_groundtruth_items)}

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

    # Create a Dataset object
    train_dataset = CustomDataset(train_contexts, train_labels, tokenizer)

    # Initialize the model with the classification head
    num_classes = len(groundtruth_id_map)
    model_with_head = ModelWithClassificationHead(base_model, num_classes, groundtruth_id_map, tokenizer)
    if torch.cuda.device_count() > 1:
        model_with_head = nn.DataParallel(model_with_head)

    # Initialize the loss
    loss_fn = MultinomialLikelihoodLoss()

    # Training parameters
    num_epochs = 8
    batch_size = 64
    learning_rate = 1e-5

    # Create a DataLoader with the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    # Optimizer
    optimizer = torch.optim.Adam(model_with_head.parameters(), lr=learning_rate, weight_decay=0)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_with_head.to(device)

    # Initialize the best hit rate
    best_hit_rate = 0.0
    best_model_path = "models/best_model.pth"

    for epoch in range(num_epochs):
        # Evaluate the model at the end of each epoch
        hit_rate = calculate_hit_rate(val_data, train_data, train_contexts, val_contexts, model_with_head, groundtruth_id_map, batch_size=512, top_k=5)
        print(f"Epoch {epoch}, Hit Rate: {hit_rate:.4f}")
        
        # Check if the current hit rate is the best we've seen
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            # Save the best model
            if isinstance(model_with_head, nn.DataParallel):
                torch.save(model_with_head.module.state_dict(), best_model_path)  # Use .module to save the underlying model
            else:
                torch.save(model_with_head.state_dict(), best_model_path)
            print(f"New best model saved with hit rate: {best_hit_rate:.4f}")

        model_with_head.train()
        total_loss = 0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model_with_head(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
