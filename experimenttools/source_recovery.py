'''
Suite for source recovery experiments
'''


import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMultiSource
import mmd_tools
import pandas as pd
import numpy as np
import mauve

def generate_embed_reference_datasets(sources, N, model_name, max_length, summary):
    '''
    Generate a set of reference datasets and embed them
    '''
    datasets = [source.obtain_samples(N) for source in sources]
    all_embeddings = [mmd_tools.get_doc_embeddings(list(dataset['text']), model_name = model_name, max_length = max_length, summary = summary) for dataset in datasets]
    return all_embeddings

def generate_embed_eval_dataset(source, N, model_name, max_length, summary):
    '''
    Generate an eval dataset and embed it 
    '''
    dataset = source.obtain_samples(N)
    embedding = mmd_tools.get_doc_embeddings(list(dataset['text']), model_name = model_name, max_length = max_length, summary = summary)
    return embedding

def get_mmds(all_ref_embeddings, eval_embedding):
    '''
    Get MMD between every set of reference embeddings and the eval embeddings
    '''
    #can also use the permutation test
    return [mmd_tools.mmd_calc(eval_embedding, ref_embedding) for ref_embedding in all_ref_embeddings]

def get_mauves(all_ref_embeddings, eval_embedding):
    '''
    Get mauve between every set of reference embeddings and the eval embeddings
    '''
    #Taking negative MAUVE because want lower value to indicate closeness, like with MMD
    return [-1*mauve.compute_mauve(p_features = ref_embedding, q_features = eval_embedding).mauve for ref_embedding in all_ref_embeddings]

def test_full_gen(sources, N, model_name, max_length, summary, method = get_mmds):
    '''
    Generate a set of reference embeddings and an eval embedding, find if the metric-minimizing reference embedding is the source
    '''
    all_ref_embeddings = generate_embed_reference_datasets(sources, N, model_name, max_length, summary)
    return test_pre_gen(sources, N, model_name, max_length, summary, all_ref_embeddings, method = method)

def test_pre_gen(sources, N, model_name, max_length, summary, all_ref_embeddings, method = get_mmds):
    '''
    Given a set of reference embeddings. Generate an eval embedding, find if the metric-minimizing reference embedding is the source
    '''
    source_ind = np.random.randint(0, len(sources))
    source = sources[source_ind]
    eval_embedding = generate_embed_eval_dataset(source, N, model_name, max_length, summary)
    if isinstance(method, list):
        metrics = [method_i(all_ref_embeddings, eval_embedding) for method_i in method]
        return source_ind, metrics
    metrics = method(all_ref_embeddings, eval_embedding)
    return source_ind, metrics

def experiment_full_gen(sources, N, model_name, max_length, summary, method = get_mmds, N_trials = 1000):
    '''
    Run test_full_gen many times
    '''
    counter = 0.0
    for count in range(N_trials):
        counter += test_full_gen(sources, N, model_name, max_length, summary, method = method)
        
    return counter/N_trials

def experiment_pre_gen(sources, N, model_name, max_length, summary, method = get_mmds, N_trials = 1000):
    '''
    Run test_pre_gen many times with the same reference embeddings
    '''
    all_ref_embeddings = generate_embed_reference_datasets(sources, N, model_name, max_length, summary)
    counter = 0.0
    for count in range(N_trials):
        counter += test_pre_gen(sources, N, model_name, max_length, summary, all_ref_embeddings, method = method)
        
    return counter/N_trials
    
#ep = MIMICEndpoint()
#sources = [MIMICSource(ep, "get_notes_diagnosis", "Z515", 10), MIMICSource(ep, "get_notes_diagnosis", "Z66", 10)]
#experiment_pre_gen(sources, 10, "UFNLP/gatortron-base", 300, 'mean', get_mmds, 10)