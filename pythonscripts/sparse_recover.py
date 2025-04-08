import sys, os
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
sys.path.append(os.path.abspath("../admintools"))
from attack_tools import *
from tabular_fitting import TabularModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from featurization_tools import BOW, Transformer, TransformerWithDimReduce, TransformerSparseAutoencoder
from tabular_fitting import TabularModel
from attack_experiment_tools import AttackExperiment
from note_metric_tools import *
from loss_tools import roc_auc_loss_fn, neg_roc_auc_loss_fn
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
import time
from document_results_tool import write

featurizer = Transformer()

ep = MIMICEndpoint()
ms1 = MIMICSource(ep, "get_notes_death", 1)
ms2 = MIMICSource(ep, "get_notes_death", 0)

dataset1 = ms1.obtain_samples(200)
dataset2 = ms2.obtain_samples(200)
dataset3 = pd.concat([ms1.obtain_samples(500), ms2.obtain_samples(500)])

feats1 = featurizer.transform(dataset1['text'])
feats2 = featurizer.transform(dataset2['text'])
feats3 = featurizer.transform(dataset3['text'])

metric = SSP()

auc = metric.dist_feats(feats1, feats2)

coef_ind = metric.get_abs_max_coef()

print('hello')

write("sparse_recover","sparse_recover.py",sys.argv[1], sys.argv[2], pd.DataFrame({"note_id": dataset3['note_id'], "ind": metric.get_sparse(feats3)}), [f"AUC: {auc}"])