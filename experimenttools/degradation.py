import numpy as np
import pandas as pd
import sys 
import joblib
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import mmd_tools
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, mean_squared_error
import mauve
from featurization_tools import BOW, Transformer

def get_classification_model_degradation(rfc1, rfc2, eval_embedding, outcome, save_name=None):
    rfc1_probs = rfc1.predict_proba(eval_embedding)[:,1]
    rfc2_probs = rfc2.predict_proba(eval_embedding)[:,1]
    auc1 = roc_auc_score(outcome, rfc1_probs)
    auc2 = roc_auc_score(outcome, rfc2_probs)
    if save_name is not None:
        joblib.dump(rfc1, save_name + "_model1.pt")
        joblib.dump(rfc2, save_name +"_model2.pt")
        np.save(save_name + "_auc1.npy", auc1)
        np.save(save_name + "_auc2.npy", auc2)
    return auc1, auc2

def predict_model_degradation(degradation, eval_embedding):
    l = LinearRegression()
    l.fit(eval_embedding, degradation)
    return l

def get_results(emb1, emb2, emb3, target1, target2, target3, featurizer, save_name=None, classification = True):
    if classification:
        rfc1 = LogisticRegression()
        rfc2 = LogisticRegression()
    rfc1.fit(emb1, target1)
    rfc2.fit(emb2, target2)
    
    degradation = get_classification_model_degradation(rfc1, rfc2, emb3, target3, save_name)
    
    #l_deg = predict_model_degradation(degradation, featurizer.transform(eval_dataset['text']))
    if save_name is not None:
        dataset1[['note_id','hadm_id']].to_csv(save_name + "_dataset1.csv")
        dataset2[['note_id','hadm_id']].to_csv(save_name + "_dataset2.csv")
        eval_dataset[['note_id','hadm_id']].to_csv(save_name + "_eval_dataset.csv")
        
        joblib.dump(l_deg, save_name + "_model.pt")
        featurizer.save(save_name + "_cv.pt")
        
    return degradation
    
             
             
             
        