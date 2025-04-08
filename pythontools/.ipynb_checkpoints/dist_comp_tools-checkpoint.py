'''
tools for distributional comparison
'''

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
from sklearn.decomposition import PCA
import scipy
import numpy as np
import re
from torch.utils.data import DataLoader
import torch.nn as nn
import joblib
import mmd_tools
from sklearn.neighbors import KernelDensity

'''
Tools for kernel density estimation
'''
def get_kde(X, bandwidth):
    kde = KernelDensity(bandwidth = bandwidth).fit(X)
    return kde

def get_kdes(X,Y, bandwidth):
    kde_x = get_kde(X, bandwidth)
    kde_y = get_kde(Y, bandwidth)
    return kde_x, kde_y

def get_argmin_kde(kde, Y, ret_scores = True):
    scores = kde.score_samples(Y)
    if ret_scores:
        return np.argmin(scores), scores
    return np.argmin(scores)

def get_argmax_kde_ratio(kde_x, kde_y, Y, ret_scores = True):
    scores_x_y = kde_x.score_samples(Y)
    scores_y_y = kde_y.score_samples(Y)   
    if ret_scores:
        return np.argmax(scores_y_y - scores_x_y), scores_x_y, scores_y_y
    return np.argmax(scores_y_y - scores_x_y)

def fit_get_argmin_kde(X, Y, bandwidth, ret_scores = False):
    kde = get_kde(X, bandwidth)
    return get_argmin_kde(kde, Y, ret_scores = ret_scores)

def fit_get_argmax_kde_ratio(X, Y, bandwidth, ret_scores = False):
    kde_x, kde_y = get_kdes(X, Y, bandwidth)
    return get_argmax_kde_ratio(kde_x, kde_y, Y, ret_scores = ret_scores)

def get_furthest_note(kde, ds_notes, featurizer):
    Y = featurizer.transform(ds_notes)
    return get_argmin_kde(kde, Y)

def fit_get_furthest_note(ds_notes_X, ds_notes_Y, featurizer, bandwidth):
    X = featurizer.transform(ds_notes_X)
    Y = featurizer.transform(ds_notes_Y)
    return fit_get_argmin_kde(X, Y, bandwidth)

def fit_get_argmax_note_kde_ratio(ds_notes_X, ds_notes_Y, featurizer_bandwidth):
    X = featurizer.transform(ds_notes_X)
    Y = featurizer.transform(ds_notes_Y)
    return fit_get_argmax_kde_ratio(X, Y, bandwidth)

'''
Tools for PDA
'''
