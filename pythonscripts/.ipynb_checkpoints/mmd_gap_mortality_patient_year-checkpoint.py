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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
import mmd_tools
import sys

ep = MIMICEndpoint()

CODE1 = str(sys.argv[1])
VERS1 = int(sys.argv[2])
DIM = int(sys.argv[3])

def train_mortality(notes1, ID = 50, model = LogisticRegression(), metric = balanced_accuracy_score, for_mmd = False, model_name = "gpt2"):
    notes1_m1 = notes1[(notes1['death'] == 1)].sample(ID)
    notes1_m0 = notes1[(notes1['death']==0)].sample(ID)
    notes1_training = pd.concat([notes1_m1.iloc[:int(ID/2)], notes1_m0.iloc[:int(ID/2)]])
    notes1_eval = pd.concat([notes1_m1.iloc[int(ID/2):], notes1_m0.iloc[:int(ID/2):]])
    
    vectorizer = CountVectorizer(min_df = 0.1, max_df = 0.9)

    # Transform the clean text data into a word frequency matrix
    X_train = vectorizer.fit_transform(notes1_training['text'])
    X_test = vectorizer.transform(notes1_eval['text'])
    y_train = notes1_training['death']
    model = model.fit(X_train, y_train)
    if for_mmd:
        embeddings = mmd_tools.get_doc_embeddings(list(notes1_training["text"]), model_name = model_name)
        return model, vectorizer, metric(notes1_eval['death'], model.predict(X_test)), embeddings
    return model, vectorizer, metric(notes1_eval['death'], model.predict(X_test))

def eval_mortality(notes, vectorizer, model, N_SAMPLES=50, metric = balanced_accuracy_score, for_mmd = False, model_name = "gpt2"):
    notesD = notes[(notes['death'] == 1)].sample(int(N_SAMPLES/2))
    notesS = notes[(notes['death']==0)].sample(int(N_SAMPLES/2))
    notes = pd.concat([notesD, notesS])
    X = vectorizer.transform(notes['text'])
    y = notes['death']
    if for_mmd:
        embeddings = mmd_tools.get_doc_embeddings(list(notes['text']), model_name = model_name)
        return metric(y, model.predict(X)), embeddings
    return metric(y, model.predict(X))

def mmd_vs_gap(notes1, notes2, N_TRIALS = 500, model_name = "gpt2"):
    gaps = []
    id_scores = []
    od_scores = []
    mmds = []
    for count in range(N_TRIALS):
        model = RandomForestClassifier(max_depth = 5)
        model, vectorizer, id_score, embeddings1 = train_mortality(notes1, model = model, for_mmd = True, model_name = model_name)
        od_score, embeddings2 = eval_mortality(notes2, vectorizer, model, for_mmd = True, model_name = model_name)
        id_scores.append(id_score)
        od_scores.append(od_score)
        gaps.append(id_score - od_score)
        if DIM < 500:
            pca = PCA(n_components = DIM)
            embeddings = pca.fit_transform(np.concatenate([embeddings1,embeddings2]))
            embeddings1 = embeddings[:embeddings1.shape[0],:]
            embeddings2 = embeddings[embeddings1.shape[0]:,:]
        mmds.append(mmd_tools.mmd_calc(embeddings1, embeddings2))
        del vectorizer
        del model
        del embeddings1
        del embeddings2
    return gaps, id_scores, od_scores, mmds

heart_failure_icd10 = ['I0981', 'I50', 'I502', 'I5020', 'I5021', 'I5022', 'I5023', 'I503',\
       'I5030', 'I5031', 'I5032', 'I5033', 'I504', 'I5040', 'I5041',\
       'I5042', 'I5043', 'I508', 'I5081', 'I50810', 'I50811', 'I50812',\
       'I50813', 'I50814', 'I5082', 'I5083', 'I5084', 'I5089', 'I509',\
       'I9713', 'I97130', 'I97131']

notes = pd.concat([ep.get_notes_diagnosis(CODE, 10) for CODE in heart_failure_icd10])

notes1 = notes[(notes['start_year'] == 2015)]
notes2 = notes[(notes['start_year'] == 2018)]

print(notes1[(notes1['death'] == 1)].shape)
print(notes2[(notes2['death'] == 1)].shape)

gaps, id_scores, od_scores, mmds = mmd_vs_gap(notes1, notes2, N_TRIALS = 500, model_name = "fine_tuned_gatortron_V2")

df_out = pd.DataFrame({"Gap": gaps, "ID": id_scores, "OD": od_scores, "MMD": mmds})
df_out.to_csv(f"../../gpt2_large_death_mmd_gap_{CODE1}_dim{DIM}_patient_year_50t_20e_gatortron.csv")