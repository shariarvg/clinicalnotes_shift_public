import numpy as np
import pandas as pd
import re
import scipy
import sklearn
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time
import torch
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
import mmd_tools 
from mimic_tools import MIMICEndpoint
ep = MIMICEndpoint()

from transformers import AutoModel
from transformers import AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True).to(device)

def get_doc_embeddings(input_text, model_name = "jinaai/jina-embeddings-v2-small-en", ret_chunk = False, vectorizer = None, batch_size = 50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    if "gpt2" in model_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    if vectorizer is None:
        all_embeddings = []
        inputs = tokenizer(input_text, return_tensors='pt', padding = 'max_length', max_length = 100, truncation = True).to(device)
        with torch.no_grad():
            for i in range(0, len(inputs["input_ids"]), batch_size):
                batch_inputs = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}
                # Get the embeddings for this batch
                batch_output = model(**batch_inputs).last_hidden_state.mean(dim=1)
                all_embeddings.append(batch_output.cpu().numpy())  # Move embeddings to CPU to save GPU memory
        return np.concatenate(all_embeddings, axis = 0)
    else:
        return vectorizer.transform(input_text).toarray()
        
def mmd_permutation_test(X,Y,number_bootstraps = 1000, size = 0.05, ret = False, ret_quantile = False):
  """
  Returns (1 if rejected, 0 if not rejected)#, mmd, and threshold
  """
  XX = scipy.spatial.distance.cdist(X,X,metric = 'euclidean')
  XY = scipy.spatial.distance.cdist(X,Y,metric = 'euclidean')
  YY = scipy.spatial.distance.cdist(Y,Y,metric = 'euclidean')
  top_row = np.block([[XX, XY]])
  bottom_row = np.block([[XY.T, YY]])
  Z = np.block([[top_row], [bottom_row]])
  upper_triangle = np.triu_indices(Z.shape[0], k=1)
  Zupp = Z[upper_triangle]
  sigma = np.median(Zupp)
  sigma = 1
  ##Recreate Z
  XX = np.exp(-1/(2*sigma) * XX)
  XY = np.exp(-1/(2*sigma) * XY)
  YY = np.exp(-1/(2*sigma) * YY)
  top_row = np.block([[XX, XY]])
  bottom_row = np.block([[XY.T, YY]])
  Z = np.block([[top_row], [bottom_row]])

  mmds = np.zeros((number_bootstraps, ))

  for b in range(number_bootstraps):
    xinds = np.random.choice(np.arange(0,X.shape[0]+Y.shape[0],1),size = X.shape[0], replace = False)
    yinds = np.delete(np.arange(0,X.shape[0]+Y.shape[0],1), xinds)
    XXb = Z[xinds[:, None], xinds].mean()
    YYb = Z[yinds[:, None], yinds].mean()
    XYb = Z[xinds[:, None], yinds].mean()
    mmds[b] = XXb + YYb - 2*XYb

  threshold = np.quantile(mmds, 1-size)
  mmd = XX.mean() + YY.mean() - 2 * XY.mean()
  print(mmd)
  print(mmds[:20])
  if ret_quantile:
    return np.mean(mmd < mmds)
  return int(mmd > threshold)

def mmd(CODE1, CODE2, VERS1 = 9, VERS2 = 10, N_SAMPLES = 100, N_BOOTSTRAPS = 100, model_name = 'gpt2'):
    set1, set2 = ep.get_hadmd_id_sets(CODE1, CODE2, VERS1, VERS2)
    notes1 = ep.get_notes_hadmd_ids(set1)
    notes2 = ep.get_notes_hadmd_ids(set2)
    notes1 = notes1.apply(lambda x: ep.bhc(x['text']), axis = 1)
    notes2 = notes2.apply(lambda x: ep.bhc(x['text']), axis = 1)
    embeddings1 = get_doc_embeddings(list(notes1.sample(N_SAMPLES)), model_name = model_name)
    embeddings2 = get_doc_embeddings(list(notes2.sample(N_SAMPLES)), model_name)
    return mmd_permutation_test(embeddings1, embeddings2, number_bootstraps = N_BOOTSTRAPS)
    #end = time.time()
    #print(f"{end - start} seconds elapsed")
    
def mmd_dimr(CODE1, CODE2, DIM = 2, VERS1 = 9, VERS2 = 10, N_SAMPLES = 100, N_BOOTSTRAPS = 100):
    set1, set2 = ep.get_hadmd_id_sets(CODE1, CODE2, VERS1, VERS2)
    notes1 = ep.get_notes_hadmd_ids(set1)
    notes2 = ep.get_notes_hadmd_ids(set2)
    notes1 = notes1.apply(lambda x: ep.bhc(x['text']), axis = 1)
    notes2 = notes2.apply(lambda x: ep.bhc(x['text']), axis = 1)
    embeddings1 = get_doc_embeddings(list(notes1.sample(N_SAMPLES)))
    embeddings2 = get_doc_embeddings(list(notes2.sample(N_SAMPLES)))
    pca = PCA(n_components = DIM)
    #lle = LocallyLinearEmbedding(n_components=DIM)
    embeddings = pca.fit_transform(np.concatenate([embeddings1,embeddings2]))
    embeddings1 = embeddings[:N_SAMPLES,:]
    embeddings2 = embeddings[N_SAMPLES:,:]
    return mmd_permutation_test(embeddings1, embeddings2, number_bootstraps = N_BOOTSTRAPS)
    
def mmd_null(CODE, VERS, N_SAMPLES = 100, N_BOOTSTRAPS = 1000, size = 0.05):
    start = time.time()
    set1 = set(ep.get_hadmd_ids_diagnosis(CODE, VERS))
    notes1 = ep.get_notes_hadmd_ids(set1)
    notes1 = notes1.apply(lambda x: ep.bhc(x['text']), axis = 1)
    embeddings1 = get_doc_embeddings(list(notes1.sample(N_SAMPLES)))
    embeddings2 = get_doc_embeddings(list(notes1.sample(N_SAMPLES)))
    return mmd_permutation_test(embeddings1, embeddings2, number_bootstraps = N_BOOTSTRAPS, size = size)

def mmd_null_dimr(CODE, VERS, DIM = 2, N_SAMPLES = 100, N_BOOTSTRAPS = 1000, size = 0.05):
    set1 = set(ep.get_hadmd_ids_diagnosis(CODE, VERS))
    notes1 = ep.get_notes_hadmd_ids(set1)
    notes1 = notes1.apply(lambda x: ep.bhc(x['text']), axis = 1)
    embeddings1 = get_doc_embeddings(list(notes1.sample(N_SAMPLES)))
    embeddings2 = get_doc_embeddings(list(notes1.sample(N_SAMPLES)))
    pca = PCA(n_components = DIM)
    #lle = LocallyLinearEmbedding(n_components=DIM)
    embeddings = pca.fit_transform(np.concatenate([embeddings1,embeddings2]))
    embeddings1 = embeddings[:N_SAMPLES,:]
    embeddings2 = embeddings[N_SAMPLES:,:]
    return mmd_permutation_test(embeddings1, embeddings2, number_bootstraps = N_BOOTSTRAPS)

def power(test, *args):
    return np.sum([test(*args) for count in range(200)])/200

def power_w_datasets(notes1, notes2, model_name = "gpt2-large", N_SAMPLES = 100, N_RUNS = 200, N_BOOTSTRAPS = 1000, null = False):
    rejs = 0.0
    for count in range(N_RUNS):
        if null:
            s = notes1['text'].sample(2*N_SAMPLES)
            s1 = s.iloc[:N_SAMPLES]
            s2 = s.iloc[N_SAMPLES:]
        else:
            s1 = notes1['text'].sample(N_SAMPLES)
            s2 = notes2['text'].sample(N_SAMPLES)
        emb1 = get_doc_embeddings(list(s1), model_name = model_name)
        emb2 = get_doc_embeddings(list(s2), model_name = model_name)
        rejs += mmd_permutation_test(emb1, emb2, number_bootstraps = N_BOOTSTRAPS)
    return rejs/N_RUNS

heart_failure_icd9 = [str(x) for x in [39891, 4280, 4281, 42820, 42821, 42822, 42823, 42830, 42831, 42832, 42833, 42840, 42841, 42842, 42843, 4289]]

## all diagnoses that have heart failure and not "Hypertensive" in their long title
heart_failure_icd10 = ['I0981', 'I50', 'I502', 'I5020', 'I5021', 'I5022', 'I5023', 'I503',\
       'I5030', 'I5031', 'I5032', 'I5033', 'I504', 'I5040', 'I5041',\
       'I5042', 'I5043', 'I508', 'I5081', 'I50810', 'I50811', 'I50812',\
       'I50813', 'I50814', 'I5082', 'I5083', 'I5084', 'I5089', 'I509',\
       'I9713', 'I97130', 'I97131']
CODE1 = "42833"
CODE2 = "I5033"
notes = ep.get_notes_diagnosis("42833", 9)
notes = pd.concat([notes, ep.get_notes_diagnosis("I5033", 10)])

if CODE1 in heart_failure_icd9 or CODE2 in heart_failure_icd10:
    for code in heart_failure_icd9:
        if code != CODE1:
            notes = pd.concat([notes, ep.get_notes_diagnosis(code, 9)])
    for code in heart_failure_icd10:
        if code != CODE2:
            notes = pd.concat([notes, ep.get_notes_diagnosis(code, 10)])
            
admissions = pd.read_csv("../physionet.org/files/mimiciv/3.0/hosp/admissions.csv.gz")
admissions['death'] = admissions['deathtime'].notna().astype(int)
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
patients = pd.read_csv("../physionet.org/files/mimiciv/3.0/hosp/patients.csv.gz")

notes = notes.merge(patients[['subject_id', 'anchor_age', 'anchor_year', 'anchor_year_group']], how = 'left')
notes = notes.merge(admissions[['hadm_id', 'death', 'admittime']], how = 'left')
notes_first_year = notes[(notes["anchor_year"] == notes["admittime"].dt.year)]
n_first_bucket = notes_first_year[(notes_first_year['anchor_year_group'] == '2008 - 2010')]
n_second_bucket = notes_first_year[(notes_first_year['anchor_year_group'] == '2017 - 2019')]

print(power_w_datasets(n_first_bucket, n_first_bucket, null = True))
print(power_w_datasets(n_first_bucket, n_second_bucket, null = True))


#notes = ep.get_notes_diagnosis(CODE2, 10)
#n_first_bucket = notes[(notes['start_year'] == 2015)]
#n_second_bucket = notes[(notes['start_year'] == 2017)]
min_bucket = min(n_first_bucket.shape[0], n_second_bucket.shape[0])
n_first_bucket_ds = n_first_bucket['text'].sample(min(500, min_bucket))
n_second_bucket_ds = n_first_bucket['text'].sample(min(500, min_bucket))
n_first_bucket_embeddings = get_doc_embeddings(list(n_first_bucket_ds), model_name = "jinaai/jina-embeddings-v2-small-en")
n_second_bucket_embeddings = get_doc_embeddings(list(n_second_bucket_ds), model_name = "jinaai/jina-embeddings-v2-small-en")
print(n_first_bucket_embeddings.shape)
print(n_second_bucket_embeddings.shape)
print(mmd_permutation_test(n_first_bucket_embeddings, n_second_bucket_embeddings, ret_quantile = True))
