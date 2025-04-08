'''
Script for prediction the ICD code or time period of a note
'''

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score

ep = MIMICEndpoint()

def prep_training_and_testing(notes1, notes2, TOTAL_EACH = 2000):
    '''
    Given two dataframes of notes, obtain a balanced training set of notes from each dataframe, and a balanced testing dataframe of notes from each dataframe
    '''
    TOTAL_EACH = min(TOTAL_EACH, min(notes1.shape[0], notes2.shape[0]))
    print("Total each: ", TOTAL_EACH)
    notes1_ds = notes1['text'].sample(TOTAL_EACH)
    notes2_ds = notes2['text'].sample(TOTAL_EACH)
    notes1_tr = notes1_ds.iloc[:int(TOTAL_EACH/2)]
    notes2_tr = notes2_ds.iloc[:int(TOTAL_EACH/2)]
    notes1_te = notes1_ds.iloc[int(TOTAL_EACH/2):]
    notes2_te = notes2_ds.iloc[int(TOTAL_EACH/2):]
    print("notes1_tr.shape: ", notes1_tr.shape[0])
    print("notes2_tr.shape: ", notes2_tr.shape[0])
    return notes1_tr, notes2_tr, notes1_te, notes2_te

def cvec_code_prediction(notes1_tr, notes2_tr, notes1_te, notes2_te, model, min_df = 0.1, max_df = 0.95, TOTAL_EACH = 2000, verbose = True):
    '''
    Fit a CountVectorizer to the training dataset and transform the training and testing datasets
    Fit a model (passed in as a parameter) to the training dataset for code (dataset) prediction
    Return the accuracy of the model on the testing dataset
    '''
    if notes1_tr.shape[0] < 20:
        return np.nan, np.nan
    cv = CountVectorizer(min_df = min_df, max_df = max_df)
    notes_tr = cv.fit_transform(pd.concat([notes1_tr, notes2_tr]))
    notes_te = cv.transform(pd.concat([notes1_te, notes2_te]))
    y_tr = np.concatenate([np.zeros(int(notes1_tr.shape[0])), np.ones(int(notes1_tr.shape[0]))])
    y_te = np.concatenate([np.zeros(int(notes1_te.shape[0])), np.ones(int(notes1_te.shape[0]))])
    model.fit(notes_tr, y_tr)
    if verbose:
        print("Accuracy (class-balanced): ", 1 - np.abs(model.predict(notes_te) - y_te).sum()/notes_te.shape[0])
        print(confusion_matrix(y_te, model.predict(notes_te)))
        if isinstance(model, LogisticRegression):
            df = pd.DataFrame({"word": cv.get_feature_names_out(), "coef": model.coef_[0]})
            print("ICD 10 words: ")
            print(df.sort_values(by = "coef", ascending = False).head(10))
            print("----")
            print("ICD 9 words: ")
            print(df.sort_values(by = "coef", ascending = True).head(10))
    return accuracy_score(y_te, model.predict(notes_te))#, recall_score(y_te, model.predict(notes_te))

def cvec_death_prediction(notes1, notes2, model, min_df = 0.1, max_df = 0.95):
    '''
    Create a balanced training dataset from notes1, fit the CV, fit the death model
    Return balanced accuracy of model on notes2 (after CV transformed)
    '''
    #notes1_d1 = notes1[(notes1['death']==1)]
    #notes1_d0 = notes1[(notes1['death']==0)]
    #notes1_d1 = notes1_d1.sample(min(notes1_d1.shape[0], notes1_d0.shape[0]))
    #notes1_d0 = notes1_d0.sample(min(notes1_d1.shape[0], notes1_d0.shape[0]))
    #notes1 = pd.concat([notes1_d0, notes1_d1])
    cv = CountVectorizer(min_df = min_df, max_df = max_df)
    if notes1.shape[0] < 20:
        return np.nan, np.nan
    model.fit(cv.fit_transform(notes1['text']), notes1['death'])
    return balanced_accuracy_score(notes2['death'], model.predict(cv.transform(notes2['text'])))

def pipeline_full(CODE, VERS, SYEAR1_group, SYEAR2_group, TOTAL_EACH = 2000, model = RandomForestClassifier(max_depth = 5), verbose = True):
    '''
    Obtain the full dataframe of notes, split it into two groups based on start year
    Obtain accuracy of code prediction on held-out set
    Obtain the balanced accuracy score of a model trained for death prediction on the first dataset when evaluated on the second dataset
    '''
    notes = ep.get_notes_diagnosis(CODE, VERS)
    notes1 = notes[(notes['start_year'].isin(SYEAR1_group))]
    notes2 = notes[(notes['start_year'].isin(SYEAR2_group))]
    print("Notes 1 shape: ", notes1.shape[0])
    print("Notes 2 shape: ", notes2.shape[0])
    notes1_tr, notes2_tr, notes1_te, notes2_te = prep_training_and_testing(notes1, notes2, TOTAL_EACH)
    return cvec_code_prediction(notes1_tr, notes2_tr, notes1_te, notes2_te, model, TOTAL_EACH = TOTAL_EACH, verbose = verbose), cvec_death_prediction(notes1, notes2, RandomForestClassifier(max_depth = 5))
    
admissions = pd.read_csv("../../../physionet.org/files/mimiciv/3.0/hosp/admissions.csv.gz")
diagnoses = pd.read_csv("../../../physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv.gz")
admissions = admissions.merge(diagnoses[['hadm_id', 'icd_code', 'icd_version']], on = "hadm_id", how = "left")
counts_dict = dict(admissions['icd_code'].value_counts())
codes = [k for k in list(counts_dict.keys()) if counts_dict[k] >= 2000]

accs_code = []
accs_death = []

for code in codes:
    print(code)
    acc_code, acc_death = pipeline_full(str(code), 10, [2015, 2016, 2017], [2018, 2019, 2020], TOTAL_EACH = 1000, model = LogisticRegression(), verbose = False)
    accs_code.append(acc_code)
    accs_death.append(acc_death)
    print("----")
    
df = pd.DataFrame({"Code": codes, "Accuracy_Code": accs_code, "Accuracy_Death": accs_death})
df.to_csv("icd_code_death_prediction_151617_181920.csv")
    
    
