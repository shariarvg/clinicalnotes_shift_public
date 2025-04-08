import nltk
import pandas as pd
import numpy
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
import mmd_tools
import sys

ep = MIMICEndpoint()
model = "UFNLP/gatortron-base"
N_SAMPLES_BUCKET = 100

codes = ep.get_top_K_codes_icd10(500)
mmds = []
mmd_quantiles = []
auc1s = []
auc2s = []
codes_used = []

def split_buckets_sy(notes_df, sy1, sy2):
    return notes_df[(notes_df['start_year']==sy1)], notes_df[(notes_df['start_year']==sy2)]

def downsample(df, N):
    return df.sample(min(N, df.shape[0]))

def split_train_eval(notes_df):
    return notes_df.iloc[:int(notes_df.shape[0]/2)], notes_df.iloc[int(notes_df.shape[0]/2):]

def cv_rfc():
    cv1 = CountVectorizer(min_df = 0.1, max_df = 0.95)
    cv2 = CountVectorizer(min_df = 0.1, max_df = 0.95)

    rfc1 = RandomForestClassifier(max_depth = 5)
    rfc2 = RandomForestClassifier(max_depth = 5)
    
    return cv1, cv2, rfc1, rfc2

def mix_train(n1, n2train):
    return pd.concat([n1.iloc[:int(n1.shape[0]/2)], n2train])

def featurize(cv1, cv2, n1, n1n2mix):
    #print(n1['text'].iloc[0])
    #print(n1n2mix['text'].iloc[0])
    f1 = cv1.fit_transform(list(n1['text']))
    f2 = cv2.fit_transform(list(n1n2mix['text']))
    return f1, f2

def featurize_and_train(cv1, cv2, n1, n2train, rfc1, rfc2):
    n1n2mix = mix_train(n1, n2train)
    feats1, feats2 = featurize(cv1, cv2, n1, n1n2mix)
    rfc1.fit(feats1, n1['death'])
    rfc2.fit(feats2, n1n2mix['death'])
    
def get_roc_auc_scores(cv1, cv2, rfc1, rfc2, texts, death):
    probs1 = rfc1.predict_proba(cv1.transform(texts))
    probs2 = rfc2.predict_proba(cv2.transform(texts))
    if probs1.shape[1] != 2 or probs2.shape[1] != 2:
        return np.nan, np.nan
    return roc_auc_score(death, probs1[:,1]), roc_auc_score(death, probs2[:,1])
                    
for code in codes:
    notes = ep.get_notes_diagnosis(code, 10)
    n_first_bucket, n_second_bucket = split_buckets_sy(notes, 2015, 2017)
    if n_first_bucket.shape[0] > N_SAMPLES_BUCKET and n_second_bucket.shape[0] > N_SAMPLES_BUCKET:
        
        n_first_bucket_ds = downsample(n_first_bucket, N_SAMPLES_BUCKET)
        n_second_bucket_ds = downsample(n_second_bucket, N_SAMPLES_BUCKET)
        
        n_second_bucket_train, n_second_bucket_eval = split_train_eval(n_second_bucket_ds)
        
        if len(n_second_bucket_eval['death'].unique())> 1:
            
            codes_used.append(code)
        
            n_first_bucket_embeddings = mmd_tools.get_doc_embeddings(list(n_first_bucket_ds['text']), model_name = model)
            n_second_bucket_embeddings = mmd_tools.get_doc_embeddings(list(n_second_bucket_ds['text']), model_name = model)

            mmd, quantile = mmd_tools.mmd_permutation_test(n_first_bucket_embeddings, n_second_bucket_embeddings, ret = True, ret_quantile = True)
            mmds.append(mmd)
            mmd_quantiles.append(quantile)

            cv1, cv2, rfc1, rfc2 = cv_rfc()
            featurize_and_train(cv1, cv2, n_first_bucket_ds, n_second_bucket_train, rfc1, rfc2)


            auc1, auc2 = get_roc_auc_scores(cv1, cv2, rfc1, rfc2, n_second_bucket_eval['text'], n_second_bucket_eval['death'])

            auc1s.append(auc1)
            auc2s.append(auc2)
        
df = pd.DataFrame({"Code": codes_used, "MMD": mmds, "Quantile": mmd_quantiles, "AUC1": auc1s, "AUC2": auc2s})
df.to_csv(f"../../mmd_auc_gap_gatortron_{N_SAMPLES_BUCKET}s_top_codes.csv")
        
        
        
