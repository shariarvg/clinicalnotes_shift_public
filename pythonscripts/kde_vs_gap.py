'''
Goal: Fit a kernel density to the input features of the training dataset and evaluate it on the input features of the held out set. Evaluate whether the density is higher for misclassified/high-error points.
'''

import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

DIM = int(sys.argv[1]) ## dimension of embeddings (for dim reduction)

ep = MIMICEndpoint()

#training data
CODE1 = "R6521"
CODE2 = "R570"
notes1 = ep.get_notes_diagnosis(CODE1, 10)
notes1_tr = notes1.sample(800) #for start year prediction
#notes1_p = notes1[(notes1['death']==1)].sample(400)  #for mortality prediction 
#notes1_n = notes1[(notes1['death']==0)].sample(400)  #for mortality prediction
#notes1_tr = pd.concat([notes1_p, notes1_n]) #for mortality prediction

#eval from notes1 (complement of notes1_tr)
notes1_heldout = pd.merge(notes1, notes1_tr, on = notes1.columns.tolist(), how = 'left', indicator = True)
notes1_heldout = notes1_heldout[(notes1_heldout['_merge'] == 'left_only')].drop(columns = '_merge')
notes1_heldout['code'] = CODE1

#eval from notes2
notes2 = ep.get_notes_diagnosis(CODE2, 10)
notes2['code'] = CODE2

#combine evals
notes2 = pd.concat([notes1_heldout, notes2])

print("Training shape: ", notes1_tr.shape[0])
print("Eval shape: ", notes2.shape[0])

## get all embeddings (default for get_doc_embeddings is gpt-large)
X1 = mmd_tools.get_doc_embeddings(list(notes1_tr['text']), model_name = "UFNLP/gatortron-base")
X2 = mmd_tools.get_doc_embeddings(list(notes2['text']), model_name = "UFNLP/gatortron-base")

## countvecotirzer
cv = CountVectorizer(min_df = 0.1, max_df = 0.95)
notes1_featurized = cv.fit_transform(notes1_tr['text'])
notes2_featurized = cv.transform(notes2['text'])

if DIM < 500:
    pca = PCA(n_components = DIM)
    X1 = pca.fit_transform(X1)
    X2 = pca.transform(X2)

## fit a random forest to the 2015 start year
rfc = RandomForestRegressor(max_depth = 5)
for year in list(notes1_tr['start_year'].unique()):
    print(str(year) + ": " + str(notes1_tr[(notes1_tr['start_year']==year)].shape[0]))
rfc.fit(notes1_featurized, notes1_tr['start_year'])

## fit the kernel density estimate to the 2015 start year
BANDWIDTH = 0.01
kde = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTH).fit(X1)

## obtain densities, prediction, and ground truth for 2018 start year
df = pd.DataFrame()
df['density'] = kde.score_samples(X2)
df['death'] = list(notes2['start_year'])
df['pred'] = rfc.predict(notes2_featurized)
df['code'] = list(notes2['code'])
df.to_csv(f"../../kde_death_prediction_dim{DIM}_{CODE1}_eval{CODE1}{CODE2}_bandwidth0p01_gatortron_syearpred.csv")




