'''
Training multiple models with different training datasets, choosing which one to use in inference using MMD

https://www.notion.so/Simultaneous-Agents-Learning-to-Defer-with-KDE-13a02ba0ca0780c0a10cc060c9d180a2?pvs=4
'''

import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMultiSource
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import mmd_tools
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_squared_error
import mauve
from featurization_tools import BOW, Transformer
from source_recovery import get_mmds, get_mauves

def get_roc_auc_score(rfc, embeddings, outcome):
    probs = rfc.predict_proba(embeddings)
    if probs.shape[1] != 2 or len(outcome.unique()) == 1:
        print(probs.shape)
        print(len(outcome.unique()))
        print("----")
        return np.nan
    return roc_auc_score(outcome, probs[:,1])

def mse(rfc, embeddings, outcome):
    predictions = rfc.predict(embeddings)
    return mean_squared_error(predictions, outcome)

def sparse_weight_vector(N, alpha = 1, drop = 0.8):
    num_zeros = int(drop * N)
    num_ones = N - num_zeros
    distr = np.random.dirichlet(alpha * np.ones(num_ones))
    vector = np.concatenate((distr, np.zeros(num_zeros)))
    np.random.shuffle(vector)
    return vector

def get_all_notes_metrics_aucs(sources, eval_source, N, model_name, summary, max_length, task):
    reference_dfs = [source.obtain_samples(N) for source in sources]
    eval_df = eval_source.obtain_samples(N)
    
    models = [RandomForestClassifier() for df in reference_dfs]

    all_ref_embeddings = [mmd_tools.get_doc_embeddings(list(reference_df['text']), model_name = model_name, summary = summary, max_length = max_length) for reference_df in reference_dfs]
    eval_embedding = mmd_tools.get_doc_embeddings(list(eval_df['text']), model_name = model_name, summary = summary, max_length = max_length)
    
    for i, model in enumerate(models):
        model.fit(all_ref_embeddings[i], reference_dfs[i][task])
        
    mmds = get_mmds(all_ref_embeddings, eval_embedding)
    mauves = get_mauves(all_ref_embeddings, eval_embedding)
    
    aucs = [get_roc_auc_score(model, eval_embedding, eval_df[task]) for model in models]
    
    return mmds, mauves, aucs

def get_all_notes_metrics_mses(sources, eval_source, N, model_name, summary, max_length, task):
    reference_dfs = [source.obtain_samples(N) for source in sources]
    eval_df = eval_source.obtain_samples(N)
    
    models = [RandomForestRegressor(max_depth = 5) for df in reference_dfs]

    all_ref_embeddings = [mmd_tools.get_doc_embeddings(list(reference_df['text']), model_name = model_name, summary = summary, max_length = max_length) for reference_df in reference_dfs]
    eval_embedding = mmd_tools.get_doc_embeddings(list(eval_df['text']), model_name = model_name, summary = summary, max_length = max_length)
    
    for i, model in enumerate(models):
        model.fit(all_ref_embeddings[i], reference_dfs[i][task])
        
    mmds = get_mmds(all_ref_embeddings, eval_embedding)
    mauves = get_mauves(all_ref_embeddings, eval_embedding)
    
    mses = [mse(model, eval_embedding, eval_df[task]) for model in models]
    
    return mmds, mauves, mses