'''
Suite for ordinal shift validation experiments
'''

import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMultiSource, MIMICMixtureSource
import mmd_tools
import pandas as pd
import numpy as np
import mauve
from source_recovery import get_mmds, get_mauves

def get_all_divergences(reference_dfs, eval_df, divergence_fn):
    return [divergence_fn(set(reference_df['hadm_id']), set(eval_df['hadm_id'])) for reference_df in reference_dfs]

def get_all_notes_metrics_divs(sources, N, model_name, summary, max_length, divergence_fn):
    reference_dfs = [source.obtain_samples(N) for source in sources]
    source_ind = np.random.randint(0, len(sources))
    eval_df = sources[source_ind].obtain_samples(N)
    
    divs = get_all_divergences(reference_dfs, eval_df, divergence_fn)
    
    all_ref_embeddings = [mmd_tools.get_doc_embeddings(list(reference_df['text']), model_name = model_name, summary = summary, max_length = max_length) for reference_df in reference_dfs]
    eval_embedding = mmd_tools.get_doc_embeddings(list(eval_df['text']), model_name = model_name, summary = summary, max_length = max_length)
    
    mmds = get_mmds(all_ref_embeddings, eval_embedding)
    mauves = get_mauves(all_ref_embeddings, eval_embedding)
    
    return source_ind, mmds, mauves, divs