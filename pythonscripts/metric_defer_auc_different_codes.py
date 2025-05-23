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

def get_roc_auc_score(rfc, embeddings, outcome):
    probs = rfc.predict_proba(embeddings)
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
TRAIN_SIZE = 10
EVAL_SIZE = 5
N_EVAL_PER_TRAIN = 2


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
#model = "sentence"
summary = "mean"
max_length = 300
task = 'admission_in_30_days'

def get_correct_assignment(model, summary, max_length, save_name, NUM_TRAIN = 15, TRAIN_SIZE = 100, EVAL_SIZE = 75, N_EVAL_PER_TRAIN = 20, best_score = "Min"):
    
    train_df_embeddings = []
    train_positivity_rate = []
    pred_models = {}
    train_df_hadm_id_sets = []
    
    #source = MIMICSource(ep, "get_notes_diagnosis", code, 10)

    for code in codes:
        #year_source = MIMICSource(ep, "get_notes_start_year", year)
        train_source = MIMICSource(ep, "get_notes_diagnosis", code, 10)
        
        '''
        Generate a train dataframe
        Embed it with gatortron base
        Fit a CV and RFC for task prediction
        '''
        train_df = train_source.obtain_samples(TOTAL_SIZE = TRAIN_SIZE)
        train_df_embedding = mmd_tools.get_doc_embeddings(list(train_df['text']), model_name = model, summary = summary, max_length = max_length)
        train_df_embeddings.append(train_df_embedding)
        train_df_hadm_id_sets.append(set(train_df['hadm_id']))
        
        pred = RandomForestClassifier(max_depth = 5)
        pred.fit(train_df_embedding, train_df[task])
        
        pred_models[code] = pred

    assignments = []
    all_metrics = []
    all_aucs = []

    for code in codes:
        for count in range(N_EVAL_PER_TRAIN):
            '''
            Generate an eval df from this weight vector
            Embed its notes with gatortron
            Find the MMD of this embedding with each training set's embedding
            Find the accuracy of each training set's (CV, RFC) models
            Store the MMD's, store the accuracies
            '''
            #year_source = MIMICSource(ep, "get_notes_start_year", year)
            eval_source = MIMICSource(ep, "get_notes_diagnosis", code, 10)
            assignments.append(code)
            eval_df = eval_source.obtain_samples(TOTAL_SIZE = EVAL_SIZE) #source.ep.get_mixture(codes, 10*np.ones(len(codes)), weight_vec, EVAL_SIZE)
            eval_embedding = mmd_tools.get_doc_embeddings(list(eval_df['text']), model_name = model, summary = summary, max_length = max_length)
            metrics = [METRIC(train_embedding, eval_embedding) for train_embedding in train_df_embeddings]
            all_metrics.append(metrics)
            
            aucs = [get_roc_auc_score(pred_models[code], eval_embedding, eval_df[task]) for code in codes]
            all_aucs.append(aucs)
            
    data = np.hstack((np.array(assignments).reshape(-1,1), np.array(all_metrics), np.array(all_aucs)))


    columns = ["Assignment"] + [f"Metric_{code}" for code in codes] + [f"AUC_{code}" for code in codes]

    df = pd.DataFrame(data = data, columns = columns)

    df.to_csv(f"../../{save_name}.csv")
    
    #df["Assignment"] = df["Assignment"].astype(int)
    #df["Min"] = df[[f"Metric{i}" for i in range(10)]].apply(lambda row: row.values.argmin(), axis=1)
    #df["Max"] = df[[f"Metric{i}" for i in range(10)]].apply(lambda row: row.values.argmax(), axis=1)
    #print(df[(df["Assignment"] == df[best_score])].shape[0]/results.shape[0])
    
save_name = "mmd_auc_" + task+ "_" + model + "_" + summary + "_" + str(max_length) + "_" + "multi_code"
    
get_correct_assignment(model, summary, max_length, save_name)

print(f"Finished Running {os.path.basename(__file__)} V{VERSION}")
        
                                                      
                                                      
            