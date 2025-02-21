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


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, contexts, labels, tokenizer, max_length=512):
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        label = self.labels[idx]
        return context, label

    def collate_fn(self, batch):
        contexts, labels = zip(*batch)
        # Tokenize and pad the contexts
        encoded_input = self.tokenizer(list(contexts), padding="longest", truncation=True, max_length=self.max_length, return_tensors='pt')
        labels = torch.stack(labels)
        return encoded_input['input_ids'], encoded_input['attention_mask'], labels
    

# Define a model with a classification head and a dense layer
class ModelWithClassificationHead(nn.Module):
    def __init__(self, model_dir, base_model, num_classes, groundtruth_id_map, tokenizer, vector_dim=768):
        super(ModelWithClassificationHead, self).__init__()
        self.base_model = base_model
        self.vector_linear = nn.Linear(base_model.config.hidden_size, vector_dim).to(base_model.device)
        self.classification_head = nn.Linear(vector_dim, num_classes, bias=False).to(base_model.device)  # No bias

        # Load the dense layer weights
        vector_linear_directory = f"2_Dense_{vector_dim}"
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)

        # Initialize weights with normalized embeddings of movie names
        movie_names = list(groundtruth_id_map.keys())
        with torch.no_grad():
            encoded_input = tokenizer(movie_names, padding="longest", truncation=True, max_length=512, return_tensors='pt')
            encoded_input = {k:v.to(base_model.device) for k, v in encoded_input.items()}
            model_output = self.base_model(**encoded_input)
            movie_embeddings = self.stella_pooling(model_output, encoded_input['attention_mask'])
            movie_embeddings = self.vector_linear(movie_embeddings)
            movie_embeddings = F.normalize(movie_embeddings, p=2, dim=1)  # Normalize embeddings
            self.classification_head.weight.copy_(movie_embeddings)

        # Disable gradient updates for the classification head
        self.classification_head.weight.requires_grad = False

    def stella_pooling(self, model_output, attention_mask):
        last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return vectors

    def forward(self, input_ids, attention_mask):
        model_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Perform pooling
        sentence_embeddings = self.stella_pooling(model_output, attention_mask)
        # Transform embeddings using the dense layer
        transformed_embeddings = self.vector_linear(sentence_embeddings)
        # Normalize embeddings
        sentence_embeddings = F.normalize(transformed_embeddings, p=2, dim=1)
        logits = self.classification_head(sentence_embeddings)
        return logits, sentence_embeddings
    

# Define the multinomial likelihood loss as a PyTorch module
class MultinomialLikelihoodLoss(nn.Module):
    def __init__(self):
        super(MultinomialLikelihoodLoss, self).__init__()

    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(labels * log_probs) / logits.size(0)
        return loss

# Function to calculate hit rate
def calculate_hit_rate(test_data, train_data, train_contexts, test_contexts, model, tokenizer, groundtruth_id_map, top_k=5, batch_size=32):
    model.eval()

    # Get train embeddings in batches
    train_embeddings = []
    num_train_batches = (len(train_contexts) + batch_size - 1) // batch_size  # Calculate number of batches

    with torch.no_grad():
        for batch_idx in tqdm(range(num_train_batches), desc="Evaluation I..."):
            batch_contexts = train_contexts[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            encoded_input = tokenizer(batch_contexts, padding="longest", max_length=512, truncation=True, return_tensors='pt').to(device)
            _, batch_embeddings = model(encoded_input['input_ids'], encoded_input['attention_mask'])
            train_embeddings.append(batch_embeddings)

    train_embeddings = torch.cat(train_embeddings)

    total_hit_rate = 0
    num_test_batches = (len(test_contexts) + batch_size - 1) // batch_size  # Calculate number of batches

    # Use tqdm to add a progress bar for the test batch loop
    for batch_idx in tqdm(range(num_test_batches), desc="Evaluating II..."):
        batch_contexts = test_contexts[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_test_items = test_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        with torch.no_grad():
            encoded_input = tokenizer(batch_contexts, padding="longest", max_length=512, truncation=True, return_tensors='pt').to(device)
            _, test_embeddings = model(encoded_input['input_ids'], encoded_input['attention_mask'])

            for i, test_embedding in enumerate(test_embeddings):
                similarities = (test_embedding.unsqueeze(0) @ train_embeddings.T).squeeze()
                top_k_indices = torch.topk(similarities, k=top_k).indices
                top_k_groundtruth = {groundtruth_id_map[gt] for idx in top_k_indices for gt in train_data[idx.item()]['groundtruth']}

                # Check if any of the test groundtruth is in the top K groundtruths
                test_gt_ids = {groundtruth_id_map[gt] for gt in batch_test_items[i]['groundtruth']}
                hits = test_gt_ids & top_k_groundtruth  # Intersection
                hit_rate = len(hits) / len(test_gt_ids) if len(test_gt_ids) > 0 else 0
                total_hit_rate += hit_rate

    average_hit_rate = total_hit_rate / len(test_data)
    return average_hit_rate

# Function to extract context
def extract_context(item):
    return "\n".join([": ".join(rnd) for rnd in item['context_raw']])


# Function to build datastore embeddings using Stella
def build_datastore_embedding(sentences, model, tokenizer, device):
    print('Building datastore embeddings')
    [model, vector_linear] = model
    
    model.eval()
    vector_linear.eval()
    
    # Initialize an empty list to hold the sentence embeddings
    all_sentence_embeddings = []
    
    # Process sentences in batches
    batch_size = 64
    
    for i in tqdm(range(0, len(sentences), batch_size), total=(len(sentences) + batch_size - 1) // batch_size):
        batch_sentences = sentences[i:i+batch_size]        
        # Tokenize sentences
        encoded_input = tokenizer(batch_sentences, padding="longest", max_length=512, truncation=True, return_tensors='pt').to(device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = stella_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = vector_linear(sentence_embeddings)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
        # Use the last hidden state
        all_sentence_embeddings.append(sentence_embeddings.cpu())
    
    # Concatenate all batched embeddings
    all_sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0).cpu()
    
    return all_sentence_embeddings

# Stella-specific pooling function
def stella_pooling(model_output, attention_mask):
    last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return vectors
