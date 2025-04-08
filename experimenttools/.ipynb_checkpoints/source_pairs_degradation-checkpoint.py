import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from featurization_tools import *
from mimic_tools import MIMICEndpoint
from sae import SparseAutoencoder
from mimic_source import MIMICSource, MIMICMultiSource, MIMICMixtureSource
from note_metric_tools import *

from degradation import get_results

note_metrics = [SSP(sae = None), MMDMetric(), MMDMetric(True), NegMauveMetric()]

def gen_source_pair_deg_results(ke, val1, val2, sourcer, param1, param2, N_trials, N_train, N_test, task, V, save_name=None):
    if save_name is None:
        save_name = f"../../diagnosis_source_pairs_{sourcer}_{param1}_{param2}_{ke}_{val1}_{val2}"


    ep = MIMICEndpoint(root = "../../..", path = "../../../notes_preproc.csv")

    ms_k1 = MIMICSource(ep, "get_notes_key_value", ke, val1)
    ms_k2 = MIMICSource(ep, "get_notes_key_value", ke, val2)
    ms_1 = MIMICSource(ep, sourcer, param1)
    ms_2 = MIMICSource(ep, sourcer, param2)
    ms11 = MIMICMultiSource([ms_k1, ms_1])
    ms12 = MIMICMultiSource([ms_k1, ms_2])
    ms21 = MIMICMultiSource([ms_k2, ms_1])
    ms22 = MIMICMultiSource([ms_k2, ms_2])

    sources = [ms11, ms12, ms21, ms22]

    ms_train1 = MIMICMixtureSource(sources, [0.5,0,0,0.5])
    ms_train2_and_test = MIMICMixtureSource(sources, [0,0.5,0.5,0])

    featurizer = Transformer("UFNLP/gatortron-base", "mean", 100, 'right')

    degradation_results = np.zeros((N_trials,2))

    note_metric_results = np.zeros((N_trials, len(note_metrics)*2))

    for trial in range(N_trials):
        ## build train set 1 from ms1 death and ms2 alive
        train_set1 = ms_train1.obtain_samples(N_train)#pd.concat([ms_diag1_death.obtain_samples(int(N_train/2)), ms_diag2_alive.obtain_samples(int(N_train/2))])
        feats_tr1 = featurizer.transform(train_set1['text'])

        ## build train set 2 from ms1 alive and ms2 death
        train_set2 = ms_train2_and_test.obtain_samples(N_train)#pd.concat([ms_diag1_alive.obtain_samples(int(N_train/2)), ms_diag2_death.obtain_samples(int(N_train/2))])
        feats_tr2 = featurizer.transform(train_set2['text'])

        ## build test set from ms1 alive and ms2 death
        test_set = ms_train2_and_test.obtain_samples(N_test)#pd.concat([ms_diag1_alive.obtain_samples(int(N_test/2)), ms_diag2_death.obtain_samples(int(N_test/2))])
        feats_test = featurizer.transform(test_set['text'])

        deg = get_results(feats_tr1, feats_tr2, feats_test, train_set1[task], train_set2[task], test_set[task], featurizer, None)
        degradation_results[trial] = deg

        for source in [ms_train1, ms_train2_and_test]:#[ms_diag1_death, ms_diag2_death, ms_diag1_alive, ms_diag2_alive]:
            source.reset()

        dists_1 = np.array([nm.dist_feats(feats_tr1, feats_test) for nm in note_metrics])
        dists_2 = np.array([nm.dist_feats(feats_tr2, feats_test) for nm in note_metrics])

        note_metric_results[trial] = np.hstack([dists_1, dists_2])

    np.save(save_name + f"_degradation_results_V{V}.npy", degradation_results)
    np.save(save_name + f"_note_metric_results_V{V}.npy", note_metric_results)



