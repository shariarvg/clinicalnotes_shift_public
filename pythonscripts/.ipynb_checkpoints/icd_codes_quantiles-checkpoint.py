'''
Obtaining MMD and/or MMD quantiles when comparing earlier subset of a code's notes to later
'''

import numpy as np
import pandas as pd
import sys 
import os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
import mmd_tools

ep = MIMICEndpoint()

V = "A5"

di = pd.read_csv("../../../physionet.org/files/mimiciv/3.0/hosp/d_icd_diagnoses.csv.gz")


# Example series
s = di['long_title']

# Get elements that appear more than once

'''
duplicates = s[s.duplicated()].unique()

all_pairs = []
for d in duplicates:
    icd9 = di[((di['long_title']==d)&(di['icd_version'] == 9))]['icd_code']
    icd10 = di[((di['long_title']==d)&(di['icd_version'] == 10))]['icd_code']
    if icd9.shape[0]> 0 and icd10.shape[0] > 0:
        all_pairs.append((icd9.iloc[0], icd10.iloc[0]))
'''

diagnoses = pd.read_csv("../../../physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv.gz")

diagnoses = diagnoses[(diagnoses['icd_version']==10)]

top_icd10_codes = diagnoses['icd_code'].value_counts()
codes = list(top_icd10_codes.index[:200])

#top_pairs = [pair for pair in all_pairs if pair[1] in set(codes)]

#model = "fine_tuned_gatortron_V2"
model = "UFNLP/gatortron-base"
N_SAMPLES_BUCKET = 350


'''
Obtain admissions and patients information
'''
'''
Initialize what will be in the output dataframe
'''
mmds = []
mmdzs = []
mmds_p = []
mmdzs_p = []
codes_used = []
n_samples_used_ds1 = []
n_samples_used_ds2 = []

summary = "mean"

#for pair in top_pairs:
for code in codes:
    '''
    Obtain the notes
    Split them into buckets
    Downsample
    Get embeddings
    Get MMD's
    '''
    
    n = ep.get_notes_diagnosis(code, 10)
    n_first_bucket_ds = ep.get_notes_start_year([2015, 2016], total_size = N_SAMPLES_BUCKET, notes = n)
    n_second_bucket_ds = ep.get_notes_start_year([2017, 2018], total_size = N_SAMPLES_BUCKET, notes = n)
    
    if n_first_bucket_ds is None or n_second_bucket_ds is None or n_first_bucket_ds.shape[0] == 0 or n_second_bucket_ds.shape[0] == 0:
        continue
        
    n_samples_used_ds1.append(n_first_bucket_ds.shape[0])
    n_samples_used_ds2.append(n_second_bucket_ds.shape[0])

    n_first_bucket_embeddings = mmd_tools.get_doc_embeddings(list(n_first_bucket_ds['text']), model_name = model, summary = summary)
    n_second_bucket_embeddings = mmd_tools.get_doc_embeddings(list(n_second_bucket_ds['text']), model_name = model, summary = summary)
    mmd, mmdz = mmd_tools.mmd_permutation_test(n_first_bucket_embeddings, n_second_bucket_embeddings, ret = True, ret_sd = True)
    mmds.append(mmd)
    mmdzs.append(mmdz)
    '''
    n_first_bucket_embeddings = mmd_tools.get_doc_embeddings(ep.get_translated_notes(list(n_first_bucket_ds['text'])), model_name = model)
    n_second_bucket_embeddings = mmd_tools.get_doc_embeddings(ep.get_translated_notes(list(n_second_bucket_ds['text'])), model_name = model)
    mmd, mmdz = mmd_tools.mmd_permutation_test(n_first_bucket_embeddings, n_second_bucket_embeddings, ret = True, ret_sd = True)
    mmds_p.append(mmd)
    mmdzs_p.append(mmdz)
    '''
    codes_used.append(code)

'''
Replicate the process for all data
'''


n_first_bucket_ds = ep.get_notes_start_year([2015, 2016], N_SAMPLES_BUCKET)
n_second_bucket_ds = ep.get_notes_start_year([2017, 2018], N_SAMPLES_BUCKET)

n_first_bucket_embeddings = mmd_tools.get_doc_embeddings(list(n_first_bucket_ds['text']), model_name = model, summary = summary)
n_second_bucket_embeddings = mmd_tools.get_doc_embeddings(list(n_second_bucket_ds['text']), model_name = model,summary = summary)
mmd, mmdz = mmd_tools.mmd_permutation_test(n_first_bucket_embeddings, n_second_bucket_embeddings, ret = True, ret_sd = True)

mmds.append(mmd)
mmdzs.append(mmdz)

n_samples_used_ds1.append(N_SAMPLES_BUCKET)
n_samples_used_ds2.append(N_SAMPLES_BUCKET)

'''
n_first_bucket_embeddings = mmd_tools.get_doc_embeddings(ep.get_translated_notes(list(n_first_bucket_ds['text'])), model_name = model)
n_second_bucket_embeddings = mmd_tools.get_doc_embeddings(ep.get_translated_notes(list(n_second_bucket_ds['text'])), model_name = model)
mmd, mmdz = mmd_tools.mmd_permutation_test(n_first_bucket_embeddings, n_second_bucket_embeddings, ret = True, ret_sd = True)

mmds_p.append(mmd)
mmdzs_p.append(mmdz)'''

'''
n_samples_used_ds1.append(N_SAMPLES_BUCKET)
n_samples_used_ds2.append(N_SAMPLES_BUCKET)
'''

df = pd.DataFrame({"Code": codes_used + ["ALL"], "MMD": mmds, "MMD_Z": mmdzs, "N1": n_samples_used_ds1, "N2": n_samples_used_ds2})
#df = pd.DataFrame({"Code": codes_used + ['all'], "MMD": mmds, "MMD_Z": mmdzs, "MMD_P": mmds_p, "MMD_Z_P": mmdzs_p, "N1": n_samples_used_ds1, "N2": n_samples_used_ds2})#, "Null1": mmd_null_1, "Null2": mmd_null_2})

df.to_csv(f"../../icd_codes_quantiles_1516_1718_{N_SAMPLES_BUCKET}s_{model}_{summary}.csv")
#df.to_csv(f"../../mmd_translated_temporal_shift_v{V}.csv")
    
print(f"Finished Running {os.path.basename(__file__)} V{V}")
    
