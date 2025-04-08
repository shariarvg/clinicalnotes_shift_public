'''
For a list of sources S_1...S_N, obtain an NxN grid where the ij-th row corresponds to using the jth source as a training set and the ith source as an eval (averaged over many iterations)

https://www.notion.so/Grid-of-Performance-by-Train-vs-Test-Source-14302ba0ca0780d48b02f17c03f5402b?pvs=4
'''
import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import mmd_tools
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import mauve
from featurization_tools import BOW, Transformer

VERSION = "6"
ep1 = MIMICEndpoint()
#ep1.get_notes_diagnosis(["I10", '4019'], [10,9], inplace = True)
start_years = np.arange(2009, 2020, 1)
sources = [MIMICSource(ep1, "get_notes_start_year", syear) for syear in start_years]

task = 'admission_in_30_days'

model_name = "UFNLP/gatortron-base"
summary = "mean"
max_length = 300

def get_trained_model(train_set, featurizer = BOW(), model = RandomForestClassifier(max_depth = 5)):
    features = featurizer.transform(train_set['text'])
    model.fit(features, train_set[task])
    return featurizer, model

def eval_score(test_set, featurizer, model, sklearn_metric):
    if sklearn_metric == roc_auc_score:
        probs = model.predict_proba(featurizer.transform(test_set['text']))
        return roc_auc_score(test_set[task], probs[:,1])
    return sklearn_metric(test_set[task], model.predict(featurizer.transform(test_set['text'])))

def do_train_get_test_score(test_source, train_source, N_train, N_eval, N_runs, sklearn_metric = accuracy_score):
    avg_score = 0.0
    for count in range(N_runs):
        train_set = train_source.obtain_samples(N_train)
        test_set = test_source.obtain_samples(N_eval)
        feat, mod = get_trained_model(train_set, featurizer = Transformer(model_name, summary, max_length))
        score = eval_score(test_set, feat, mod, sklearn_metric)
        avg_score += score
    return avg_score/N_runs
        
grid = np.zeros((len(sources), len(sources)))
for i, test_source in enumerate(sources):
    for j, train_source in enumerate(sources):
        print(i)
        print(j)
        grid[i][j] = do_train_get_test_score(test_source, train_source, 100, 100, 200, roc_auc_score)
        print("----")
np.save(f"../../grid_v{VERSION}.npy", grid)