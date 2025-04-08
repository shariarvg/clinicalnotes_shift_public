import numpy as np
import pandas as pd
from mimic_tools import MIMICEndpoint
import mmd_tools

ep = MIMICEndpoint()

mmds = []
mmd_null_1 = []
mmd_null_2 = []

for code_pair in [["4019", "I10"], ["311", "F329"]]:
    #get first diagnosis bucket
    notes = ep.get_notes_diagnosis("I5033", 10)
    
    n_first_bucket = notes[((notes['start_year'] == 2015) | (notes['start_year'] == 2016))]
    #n_first_bucket['text'] = n_first_bucket['text'].apply(mmd_tools.preprocess_text) # preprocess
    n_first_bucket = n_first_bucket[(n_first_bucket['text'] != "")] # get rid of empty text notes
    
    #get second diagnosis bucket
    n_second_bucket = notes[((notes['start_year'] == 2017) | (notes['start_year'] == 2018))]
    #n_second_bucket['text'] = n_second_bucket['text'].apply(mmd_tools.preprocess_text) # preprocess
    n_second_bucket = n_second_bucket[(n_second_bucket['text'] != "")] # get rid of empty text notes
    
    #downsample
    print(n_first_bucket.shape[0])
    n_first_bucket_ds = n_first_bucket['text'].sample(200)
    n_second_bucket_ds = n_second_bucket['text'].sample(200)

    n_first_bucket_embeddings = mmd_tools.get_doc_embeddings(list(n_first_bucket_ds), model_name = "gpt2-large") #embed
    n_second_bucket_embeddings = mmd_tools.get_doc_embeddings(list(n_second_bucket_ds), model_name = "gpt2-large") #embed
    
    
    # calculate mmd's
    mmds.append(mmd_tools.mmd_calc(n_first_bucket_embeddings[:100], n_second_bucket_embeddings[:100]))
    mmd_null_1.append(mmd_tools.mmd_calc(n_first_bucket_embeddings[:100], n_first_bucket_embeddings[100:]))
    mmd_null_2.append(mmd_tools.mmd_calc(n_second_bucket_embeddings[:100], n_second_bucket_embeddings[100:]))
    
pd.DataFrame({"Diagnosis": ["Hypertension", "Depression"], "MMD": mmds, "Null1": mmd_null_1, "Null2": mmd_null_2}).to_csv("new_pipeline_icd_split_patient_year_nofilter.csv")