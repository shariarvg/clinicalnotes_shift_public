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

def mmd_metric(embeddings1, embeddings2):
    return mmd_tools.mmd_permutation_test(embeddings1, embeddings2, ret_sd = True)
    #return mmd_tools.mmd_permutation_test(embeddings1, embeddings2, ret = True)
def mauve_metric(embeddings1, embeddings2):
    return mauve.compute_mauve(p_features = embeddings1, q_features = embeddings2).mauve

def get_roc_auc_score(feat, rfc, texts, outcome, featurization = None):
    if featurization is None:
        probs = rfc.predict_proba(feat.transform(texts))
    else:
        probs = rfc.predict_proba(featurization)
    if probs.shape[1] != 2 or len(outcome.unique()) == 1:
        print(probs.shape)
        print(len(outcome.unique()))
        print("----")
        return np.nan
    return roc_auc_score(outcome, probs[:,1])

def mse(feat, rfc, texts, outcome, featurization = None):
    if featurization is None:
        predictions = rfc.predict(feat.transform(texts))
    else:
        predictions = rfc.predict(featurization)
    return mean_squared_error(predictions, outcome)

def sparse_weight_vector(N, alpha = 1, drop = 0.8):
    num_zeros = int(drop * N)
    num_ones = N - num_zeros
    distr = np.random.dirichlet(alpha * np.ones(num_ones))
    vector = np.concatenate((distr, np.zeros(num_zeros)))
    np.random.shuffle(vector)
    return vector


ep = MIMICEndpoint()

VERSION = "9"
TRAIN_SIZE = 100
EVAL_SIZE = 75
N_RUNS = 200

#task = 'death'
pred = RandomForestClassifier
METRIC = mmd_metric
SCORE = get_roc_auc_score

codes = ['Z66',\
 'E874',\
 'J95851',\
 'Z515',\
 'J9602',\
 'R6521',\
 'K7200',\
 'R578',\
 'R570',\
 'J9600',\
 'G935',\
 'N170']
model_base = ""
model_name = "sentence"
model = model_base + model_name
summary = "mean"
max_length = 300
CODE = "Z515"

source_top = MIMICSource(ep, "get_notes_diagnosis", CODE, 10)

out = []

for run in range(N_RUNS):
    eval_year = np.random.randint(2014, 2021)
    source_eval_year = MIMICSource(ep, "get_notes_start_year", eval_year)
    source_eval = MIMICMultiSource([source_top, source_eval_year])
    eval_set = source_eval.obtain_samples(EVAL_SIZE)
    eval_embeddings = mmd_tools.get_doc_embeddings(list(eval_set['text']), model_name = model, summary = summary)
    
    mmds = []
    
    for year in np.arange(2014, 2021, 1):
        source_year = MIMICSource(ep, "get_notes_start_year", year)
        source = MIMICMultiSource([source_top, source_year])

        train_set = source.obtain_samples(TRAIN_SIZE)
        train_embeddings = mmd_tools.get_doc_embeddings(list(train_set['text']), model_name = model, summary = summary)
        
        mmds.append(mmd_metric(train_embeddings, eval_embeddings))
        
    out.append([eval_year] + mmds)
    
save_name = "../../validation_year_source" + CODE + "_" + model_name + "_" + summary + "_" + str(max_length)

columns = ["EvalYear"] + ["MMD_"+str(year) for year in np.arange(2014, 2021, 1)]
pd.DataFrame(data = out, columns = columns).to_csv(save_name)


print(f"Finished Running {os.path.basename(__file__)} V{VERSION}")
        
                                                      
                                                      
            