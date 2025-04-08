import nltk
import pandas as pd
import numpy
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
admissions = pd.read_csv("../physionet.org/files/mimiciv/3.0/hosp/admissions.csv.gz")
admissions['death'] = admissions['deathtime'].notna().astype(int)

CODE1 = str(sys.argv[1])
VERS1 = int(sys.argv[2])
CODE2 = str(sys.argv[3])
VERS2 = int(sys.argv[4])
DIM = int(sys.argv[5])

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

# Apply preprocessing to the 'text' column
def train_mortality(notes1, ID = 200, model = LogisticRegression(), metric = balanced_accuracy_score, for_mmd = False):
    notes1_m1 = notes1[(notes1['death'] == 1)].sample(ID)
    notes1_m0 = notes1[(notes1['death']==0)].sample(ID)
    notes1_training = pd.concat([notes1_m1.iloc[:int(ID/2)], notes1_m0.iloc[:int(ID/2)]])
    notes1_eval = pd.concat([notes1_m1.iloc[int(ID/2):], notes1_m0.iloc[:int(ID/2):]])
    
    vectorizer = CountVectorizer(min_df = 0.1, max_df = 0.9)

    # Transform the clean text data into a word frequency matrix
    X_train = vectorizer.fit_transform(notes1_training['clean_text'])
    X_test = vectorizer.transform(notes1_eval['clean_text'])
    y_train = notes1_training['death']
    model = model.fit(X_train, y_train)
    if for_mmd:
        embeddings = mmd_tools.get_doc_embeddings(list(notes1_training["text"]))
        return model, vectorizer, metric(notes1_eval['death'], model.predict(X_test)), embeddings
    return model, vectorizer, metric(notes1_eval['death'], model.predict(X_test))

def eval_mortality(notes, vectorizer, model, N_SAMPLES=200, metric = balanced_accuracy_score, for_mmd = False):
    notesD = notes[(notes['death'] == 1)].sample(int(N_SAMPLES/2))
    notesS = notes[(notes['death']==0)].sample(int(N_SAMPLES/2))
    notes = pd.concat([notesD, notesS])
    X = vectorizer.transform(notes['clean_text'])
    y = notes['death']
    if for_mmd:
        embeddings = mmd_tools.get_doc_embeddings(list(notes['text']))
        return metric(y, model.predict(X)), embeddings
    return metric(y, model.predict(X))

def mmd_vs_gap(CODE1, CODE2, VERS1, VERS2, N_TRIALS = 500):
    notes1 = ep.get_notes_diagnosis(CODE1, VERS1)
    notes1 = notes1.merge(admissions[['hadm_id', 'death']], how = 'left')
    notes1['clean_text'] = notes1['text'].apply(preprocess_text)
    notes2 = ep.get_notes_diagnosis(CODE2, VERS2)
    notes2 = notes2.merge(admissions[['hadm_id', 'death']], how = 'left')
    notes2['clean_text'] = notes2['text'].apply(preprocess_text)
    gaps = []
    id_scores = []
    od_scores = []
    mmds = []
    for count in range(N_TRIALS):
        model = RandomForestClassifier(max_depth = 5)
        model, vectorizer, id_score, embeddings1 = train_mortality(notes1, model = model, for_mmd = True)
        od_score, embeddings2 = eval_mortality(notes2, vectorizer, model, for_mmd = True)
        id_scores.append(id_score)
        od_scores.append(od_score)
        gaps.append(id_score - od_score)
        if DIM < 500:
            pca = PCA(n_components = DIM)
            embeddings = pca.fit_transform(np.concatenate([embeddings1,embeddings2]))
            embeddings1 = embeddings[:embeddings1.shape[0],:]
            embeddings2 = embeddings[embeddings1.shape[0]:,:]
        mmds.append(mmd_tools.mmd_permutation_test(embeddings1, embeddings2))
        del vectorizer
        del model
        del embeddings1
        del embeddings2
    return gaps, id_scores, od_scores, mmds

gaps, id_scores, od_scores, mmds = mmd_vs_gap(CODE1, CODE2, VERS1, VERS2, N_TRIALS = 500)

df_out = pd.DataFrame({"Gap": gaps, "ID": id_scores, "OD": od_scores, "MMD": mmds})
df_out.to_csv(f"death_mmd_gap_{CODE1}_{CODE2}_dim{DIM}")