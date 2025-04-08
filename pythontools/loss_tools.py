import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_squared_error

def roc_auc_loss_fn(trained_model, feats, target, dataset):
    probs = trained_model.predict_proba(feats)[:,1]
    return roc_auc_score(dataset[target].values, probs)

def neg_roc_auc_loss_fn(trained_model, feats, target, dataset):
    return -1*roc_auc_loss_fn(trained_model, feats, target, dataset)