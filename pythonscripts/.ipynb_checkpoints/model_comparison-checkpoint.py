'''
For a list of sources S_1...S_N, build one model for each type of source except the last one, test all of them on the last one, and output the prediction probabilities

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

VERSION = "1"
ep1 = MIMICEndpoint()
start_years = np.arange(2009, 2018, 1)
sources = [MIMICSource(ep1, "get_notes_start_year", syear) for syear in start_years]
task = 'admission_in_30_days'

def get_trained_model(train_set, featurizer, model = RandomForestClassifier(max_depth = 5)):
    features = featurizer.transform(train_set['text'])
    model.fit(features, train_set[task])
    return featurizer, model

def prediction_by_individual(test_set, featurizer, model):
    return model.predict_proba(featurizer.transform(test_set['text']))

def do_train_get_test_pred(test_source, train_sources, N_train, N_eval):
    test_set = test_source.obtain_samples(N_eval)
    all_preds = []
    for train_source in train_sources:
        train_set = train_source.obtain_samples(N_train)
        feat, mod = get_trained_model(train_set, BOW(0.1, 0.95))
        preds = prediction_by_individual(test_set, feat, mod)
        all_preds.append(preds)
    return all_preds
        
all_preds = do_train_get_test_pred(sources[-1], sources[:-1], 750, 250)

np.save(f"../../allpreds_v{VERSION}.npy", all_preds)