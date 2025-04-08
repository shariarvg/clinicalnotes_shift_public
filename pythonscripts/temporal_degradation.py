import sys, os
sys.path.append(os.path.abspath("../pythontools"))
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource, MIMICMixtureSource, MIMICMultiSource
from featurization_tools import Transformer, BOW, DimReduce, TransformerWithDimReduce
import pandas as pd
import numpy as np
import mmd_tools
import torch.nn as nn
import torch
from loss_tools import neg_roc_auc_loss_fn
from sklearn.ensemble import RandomForestClassifier

ep = MIMICEndpoint()

## Experiment 
CODE1 = "NONE"
CODE2 = "NONE"
VERS1 = 10
VERS2 = 10
YEAR1 = 2017
YEAR2 = 2020

## Experiment parameters
task = "long_length_of_stay"
N_train = 500
N_test = 500
N_trials = 500

## Classifier parameters
classifier = RandomForestClassifier(max_depth = 5)

## Featurizer parameters
#featurizer = TransformerWithDimReduce("../../fine_tuned_gatortron_V3", "first", 100)
#featurizer = TransformerWithDimReduce("UFNLP/gatortron-base", "mean", 100, 50, False) #model, summary, max length, n components, sparse
featurizer = Transformer("UFNLP/gatortron-base", max_length = 250, truncation_side = "left")

commit_hash = sys.argv[1]
V = sys.argv[2]
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash
save_name = f"../../experimentresults/temporal_degradation_{CODE1}_{VERS1}_{CODE2}_{VERS2}_{YEAR1}_{YEAR2}_{task}_V{V}"

#ms1a = MIMICSource(ep, "get_notes_diagnosis", CODE1, VERS1)
ms1b = MIMICSource(ep, "get_notes_start_year", YEAR1)
#ms2a = MIMICSource(ep, "get_notes_diagnosis", CODE2, VERS2)
ms2b = MIMICSource(ep, "get_notes_start_year", YEAR2)

#ms1 = MIMICMultiSource([ms1a, ms1b])
#ms2 = MIMICMultiSource([ms2a, ms2b])
#ms1 = ms1a
#ms2 = ms2a
ms1 = ms1b
ms2 = ms2b

losses = np.zeros((N_trials,2))

for trial in range(N_trials):
    train = ms1.obtain_samples(N_train)
    test = ms2.obtain_samples(N_test)
    
    feat_train = featurizer.transform(train['text'])
    #print("featurized train")
    classifier.fit(feat_train, train[task])
    #print("trained classifier")
    loss1 = neg_roc_auc_loss_fn(classifier, featurizer.transform(test['text']), task, test)
    #print("calculated loss")
    featurizer.reset()
    #print('reset featurizer')
    
    train = ms2.obtain_samples(N_train)
    
    feat_train = featurizer.transform(train['text'])
    classifier.fit(feat_train, train[task])
    loss2 = neg_roc_auc_loss_fn(classifier, featurizer.transform(test['text']), task, test)
    featurizer.reset()
    
    losses[trial] = [loss1, loss2]
    
    ms1.reset()
    ms2.reset()
    
    print(f"Trial {trial} completed")

np.save(save_name+"_losses.npy", losses)
with open(save_name + ".txt", 'w') as f:
    f.write(commit_link + "\n")
    f.write('temporal_degradation_final.py')