from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from transformers import AutoModel
from transformers import AutoTokenizer
import sys
from mimic_tools import MIMICEndpoint #this line is also in imports.py, so idk why I have to include it here too
import mmd_tools 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

heart_failure_icd10 = ['I0981', 'I50', 'I502', 'I5020', 'I5021', 'I5022', 'I5023', 'I503',\
       'I5030', 'I5031', 'I5032', 'I5033', 'I504', 'I5040', 'I5041',\
       'I5042', 'I5043', 'I508', 'I5081', 'I50810', 'I50811', 'I50812',\
       'I50813', 'I50814', 'I5082', 'I5083', 'I5084', 'I5089', 'I509',\
       'I9713', 'I97130', 'I97131']


CODE = sys.argv[1]
VERS = int(sys.argv[2])
syear1 = int(sys.argv[3])
syear2 = int(sys.argv[4])

ep = MIMICEndpoint()

notes = pd.DataFrame()
for code in heart_failure_icd10:
    notes = pd.concat([notes, ep.get_notes_diagnosis(code, 10)])

notes1 = notes[(notes['start_year']==syear1)].sample(250)
notes2 = notes[(notes['start_year']==syear2)].sample(250)

notes1_embeddings = mmd_tools.get_doc_embeddings(list(notes1['text']))
notes2_embeddings = mmd_tools.get_doc_embeddings(list(notes2['text']))

def compute_log_likelihood(X, mean, precision_chol):
    """ Compute log-likelihood of X under a Gaussian with the given mean and precision_chol """
    d = X - mean
    log_det = np.sum(np.log(np.diagonal(precision_chol)))
    z = np.dot(d, precision_chol)
    return -0.5 * np.sum(z ** 2, axis=1) + log_det - 0.5 * notes1_embeddings.shape[0] * np.log(2 * np.pi)


## fit gmm in order to approximate density of points within distribution 1
gm = GaussianMixture(n_components = 5, weights_init = [0.2, 0.2, 0.2, 0.2, 0.2])
gm.fit(notes1_embeddings)

# Get the means, covariances, and precisions of each Gaussian component
means = gm.means_
covariances = gm.covariances_
precisions = gm.precisions_cholesky_

## obtain features for model
vectorizer = CountVectorizer(min_df = 0.1, max_df = 0.9)
X_train = vectorizer.fit_transform(notes1['text'])
y_train = notes1['death']
model = RandomForestClassifier(max_depth = 5)
model = model.fit(X_train, y_train)


log_likelihoods = np.array([
    compute_log_likelihood(notes2_embeddings, means[k], precisions[k]) 
    for k in range(5)
])

# Select the maximum log-likelihood for each sample across all components
max_log_likelihoods = np.max(log_likelihoods, axis=0)


pred_error = np.abs(notes2['death'] - model.predict(vectorizer.transform(notes2['text'])))
pd.DataFrame({"MaxScore": max_log_likelihoods, "Error": pred_error}).to_csv(f"likelihoodmax_vs_gap_heart_failure_10_{syear1}_{syear2}.csv")