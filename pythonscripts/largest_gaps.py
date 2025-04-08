'''
The purpose of this script is to train one model for death prediction on all ICD codes (in a set) with 2015/2016 start years, then separately evaluate model performance on each ICD code in 2019 and 2015/2016. The performance gap of OOS for the ICD code in 2019 vs. 2015 should indicate how much drift there is.
'''
import pandas as pd
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score

ep = MIMICEndpoint()

codes = ['R6521',\
 'G935',\
 'R578',\
 'J9600',\
 'J95851',\
 'Z515',\
 'Z66',\
 'R570',\
 'K7200',\
 'J9602',\
 'E874',\
 'N170']

train_df_big = pd.DataFrame()
eval_df = []

gaps = []
big_train_scores = []
small_train_scores = []

for code in codes:
    ep = MIMICEndpoint()
    notes = ep.get_notes_diagnosis(code, 10)
    df_1516 = notes[(notes['start_year'].isin([2015, 2016]))]
    df_1920 = notes[(notes['start_year'].isin([2019, 2020]))]
    train_keep_1516 = df_1516.shape[0] - 100
    train_keep_1920 = df_1920.shape[0] - 100
    df_1516 = df_1516.sample(frac = 1)
    train_df_big = pd.concat([train_df_big, df_1516.iloc[:train_keep_1516]])
    train_df_big = pd.concat([train_df_big, df_1920.iloc[:train_keep_1920]])
    eval_df.append(df_1920.iloc[train_keep_1920:])
    
cv_big = CountVectorizer(min_df = 0.1, max_df = 0.95)
cv_small = CountVectorizer(min_df = 0.1, max_df = 0.95)

rfc_big = RandomForestClassifier(max_depth = 5)
rfc_small = RandomForestClassifier(max_depth = 5)

## Full training dataset featurization and training
train_df_big_featurized = cv_big.fit_transform(train_df_big['text'])
rfc_big.fit(train_df_big_featurized, train_df_big['death'])
del train_df_big_featurized

## Small training dataset featurization and training
train_df_big = train_df_big[(train_df_big['start_year'].isin([2015, 2016]))]
train_df_small_featurized = cv_small.fit_transform(train_df_big['text'])
rfc_small.fit(train_df_small_featurized, train_df_big['death'])
del train_df_small_featurized

for i, code in enumerate(codes):
    big_train_scores.append(balanced_accuracy_score(eval_df[i]['death'], rfc_big.predict(cv_big.transform(eval_df[i]['text']))))
    small_train_scores.append(balanced_accuracy_score(eval_df[i]['death'], rfc_small.predict(cv_small.transform(eval_df[i]['text']))))
    gaps.append(big_train_scores[-1] - small_train_scores[-1])
    
df = pd.DataFrame({"Code": codes, "BigTrain": big_train_scores, "SmallTrain": small_train_scores, "Gap": gaps})
df.to_csv("gaps_1516_1920.csv")
    