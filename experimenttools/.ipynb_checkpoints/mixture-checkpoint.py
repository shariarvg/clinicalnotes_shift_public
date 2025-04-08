'''
Suite for mixture experiments
'''

import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMultiSource, MIMICMixtureSource
import mmd_tools
import pandas as pd
import numpy as np
import mauve

def gen_pair_datasets(list_sources, list_weights, source_ind, N):
    pure_df = list_sources[source_ind].obtain_samples(N)
    mixture_df = MIMICMixtureSource(list_sources, list_weights).obtain_samples(N)
    return pure_df, mixture_df
    
def gen_pair_embeddings(note_df1, note_df2, model_name, summary, max_length):
    return mmd_tools.get_doc_embeddings(list(note_df1['text']), model_name = model_name, summary = summary, max_length = max_length), mmd_tools.get_doc_embeddings(list(note_df2['text']), model_name = model_name, summary = summary, max_length = max_length)

def get_result(list_sources, list_weights, source_ind, N, model_name, summary, max_length):
    note_df1, note_df2 = gen_pair_datasets(list_sources, list_weights, source_ind, N)
    emb1, emb2 = gen_pair_embeddings(note_df1, note_df2, model_name, summary, max_length)
    mmd = mmd_tools.mmd_calc(emb1, emb2)
    neg_mauve = -1*mauve.compute_mauve(p_features = emb1, q_features = emb2).mauve
    return mmd, neg_mauve