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
from mimic_source import MIMICSource
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

VERSION = "7"
NUM_TRAIN = 15
TRAIN_SIZE = 100
EVAL_SIZE = 75
N_EVAL_PER_TRAIN = 20

task = 'death'
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

model = "fine_tuned_gatortron_V2"
summary = "mean"
max_length = 300

def get_correct_assignment(model, summary, max_length, save_name, alpha = 1, drop = 0.8, NUM_TRAIN = 15, TRAIN_SIZE = 100, EVAL_SIZE = 75, N_EVAL_PER_TRAIN = 20, best_score = "Min"):
    
    all_weight_vecs = [sparse_weight_vector(len(codes), alpha = 1, drop = 0.8) for count in range(NUM_TRAIN)]
    sources = [MIMICSource(ep, "get_mixture", codes, [10 for code in codes], weights) for weights in all_weight_vecs]

    train_df_embeddings = []
    train_positivity_rate = []
    feat_models = []

    for source in sources:
        '''
        Generate a train dataframe
        Embed it with gatortron base
        Fit a CV and RFC for death prediction
        '''
        train_df = source.obtain_samples(TOTAL_SIZE = TRAIN_SIZE)
        train_df_embedding = mmd_tools.get_doc_embeddings(list(train_df['text']), model_name = model, summary = summary, max_length = max_length)
        train_df_embeddings.append(train_df_embedding)

        rfc = pred(max_depth = 5)
        feat = None #BOW()
        rfc.fit(train_df_embedding, train_df[task])
        #rfc.fit(feat.transform(train_df['text']), train_df[task])
        feat_models.append((feat, rfc))
        train_positivity_rate.append(train_df[task].mean())

        metadata = np.hstack((np.array(train_positivity_rate).reshape(-1,1), np.array(all_weight_vecs)))
        meta_columns = ["Positivity"] + [f"Weight{code}" for code in codes]
        pd.DataFrame(metadata, columns = meta_columns).to_csv(f"../../{save_name}_metadata.csv")

        assignments = []
        all_metrics = []
        all_accuracies = []

    def get_acc(cv, rfc, notes, balanced = False):
        if balanced:
            return balanced_accuracy_score(notes[task], rfc.predict(cv.transform(notes['text'])))
        return accuracy_score(rfc.predict(cv.transform(notes['text'])), notes[task])

    for i, source in enumerate(sources):
        for count in range(N_EVAL_PER_TRAIN):
            '''
            Generate an eval df from this weight vector
            Embed its notes with gatortron
            Find the MMD of this embedding with each training set's embedding
            Find the accuracy of each training set's (CV, RFC) models
            Store the MMD's, store the accuracies
            '''
            assignments.append(i)
            eval_df = source.obtain_samples(TOTAL_SIZE = EVAL_SIZE) #source.ep.get_mixture(codes, 10*np.ones(len(codes)), weight_vec, EVAL_SIZE)
            eval_embedding = mmd_tools.get_doc_embeddings(list(eval_df['text']), model_name = model, summary = summary, max_length = max_length)
            metrics = [METRIC(train_embedding, eval_embedding) for train_embedding in train_df_embeddings]
            accuracies = [SCORE(feat, rfc, eval_df['text'], eval_df[task], eval_embedding) for (feat, rfc) in feat_models]
            all_metrics.append(metrics)
            all_accuracies.append(accuracies) ## actually an AUC

    data = np.hstack((np.array(assignments).reshape(-1,1), np.array(all_metrics), np.array(all_accuracies)))

    columns = ["Assignment"] + [f"Metric{i}" for i in range(NUM_TRAIN)] + [f"Score{i}" for i in range(NUM_TRAIN)]

    df = pd.DataFrame(data = data, columns = columns)

    df.to_csv(f"../../{save_name}.csv")
    
    df["Assignment"] = df["Assignment"].astype(int)
    df["Min"] = df[[f"Metric{i}" for i in range(10)]].apply(lambda row: row.values.argmin(), axis=1)
    df["Max"] = df[[f"Metric{i}" for i in range(10)]].apply(lambda row: row.values.argmax(), axis=1)
    print(df[(df["Assignment"] == df[best_score])].shape[0]/results.shape[0])

print(f"Finished Running {os.path.basename(__file__)} V{VERSION}")
        
                                                      
                                                      
            