import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import mmd_tools
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_squared_error
import mauve
from featurization_tools import BOW, Transformer
from sklearn.metrics import log_loss


def get_trained_model(train_df, featurizer, task, classification = True):
    embeddings = featurizer.transform(list(train_df['text']))
    if classification:
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    model.fit(embeddings, train_df[task])
    return model

def eval_score(test_df, featurizer, model, task, sklearn_metric):
    if sklearn_metric == roc_auc_score:
        probs = model.predict_proba(featurizer.transform(test_df['text']))
        return roc_auc_score(test_df[task], probs[:,1])
    return sklearn_metric(test_df[task], model.predict(featurizer.transform(test_df['text'])))

def iteration(source1, source2, N, featurizer, task, classification, sklearn_metric):
    train_df = source1.obtain_samples(N)
    test_df = source2.obtain_samples(N)
    featurizer, model = get_trained_model(train_df, featurizer, task, classification)
    score = eval_score(test_df, featurizer, model, task, sklearn_metric)
    return score