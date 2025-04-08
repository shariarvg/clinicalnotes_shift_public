import pandas as pd
import joblib
import re
import sys
import mmd_tools
import mimic_tools
from mimic_tools import MIMICEndpoint
from mimic_source import MIMICSource
from attack_tools import *
from sklearn.feature_extraction.text import CountVectorizer
from abc import ABC, abstractmethod
from featurization_tools import *
import mauve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys, os
sys.path.append(os.path.abspath("../../pda"))
from relax import relax
import time
import numpy as np

class NoteMetric(ABC):
    
    def __init__(self):
        self.N_times_used = 0
        self.S = 0
    
    def featurize_datasets(self, dataset1, dataset2):
        return self.featurizer.transform(dataset1['text']), self.featurizer.transform(dataset2['text'])
    
    @abstractmethod
    def dist_feats_calc(self, feat1, feat2):
        pass
    
    def update_runtime_statistics(self, t):
        self.N_times_used += 1.0
        self.S += t
        
    def get_avg_runtime(self, npround = 2):
        return np.round(self.S/self.N_times_used,npround)
    
    def dist_feats(self, feat1, feat2):
        start = time.time()
        dist = self.dist_feats_calc(feat1, feat2)
        end = time.time()
        self.update_runtime_statistics(end - start)
        return dist
    
    def dist_datasets(self, dataset1, dataset2):
        feat1, feat2 = self.featurize_datasets(dataset1, dataset2)
        return self.dist_feats(feat1, feat2)
    
class MMDMetric(NoteMetric):
    def __init__(self, ret_sd = False):
        self.ret_sd = ret_sd
        super().__init__()
        
    def dist_feats_calc(self, feat1, feat2):
        return mmd_tools.mmd_permutation_test(feat1, feat2, ret = not self.ret_sd, ret_sd = self.ret_sd)
    
class NegMauveMetric(NoteMetric):
    def __init__(self):
        super().__init__()
        
    def dist_feats_calc(self, feat1, feat2):
        return -1*mauve.compute_mauve(p_features = feat1, q_features = feat2).mauve
    
class SSP(NoteMetric):
    def __init__(self, train_size = 0.8, classifier = LogisticRegression, sae = Sparsify()):
        self.model = classifier()
        self.train_size = train_size
        self.sae = sae
        super().__init__()
        
    def dist_feats_calc(self, feat1, feat2):
        if self.sae is not None:
            feat1 = self.sae.transform(feat1)
            feat2 = self.sae.transform(feat2)
        
        total_feat = np.vstack([feat1, feat2])
        outcome = np.hstack([np.ones(feat1.shape[0]), np.zeros(feat2.shape[0])])
        
        perm = np.random.permutation(total_feat.shape[0])

        # Apply the permutation to both arrays
        total_feat_shuffled = total_feat[perm]
        outcome_shuffled = outcome[perm]
        
        self.model.fit(total_feat_shuffled[:int(total_feat_shuffled.shape[0]*self.train_size),], outcome_shuffled[:int(outcome_shuffled.shape[0]*self.train_size)])
        
        print(total_feat_shuffled.shape)
        print(total_feat_shuffled[:int(total_feat_shuffled.shape[0]*self.train_size),].shape)
        print(total_feat_shuffled[int(total_feat_shuffled.shape[0]*self.train_size):,].shape)
        
        probs = self.model.predict_proba(total_feat_shuffled[int(total_feat_shuffled.shape[0]*self.train_size):,])[:,1]
        return roc_auc_score(outcome_shuffled[int(outcome_shuffled.shape[0]*self.train_size):], probs)
    
    def get_abs_max_coef(self):
        return np.argmax(np.abs(self.model.coef_))
    
    def get_argsort_coef(self):
        return np.argsort(self.model.coef_)
    
    def get_abs_argsort_coef(self):
        return np.argsort(np.abs(self.model.coef_))
    
    def get_sparse(self, feats, ind = "top"):
        feats = self.sae.transform(feats)
        if ind == "top":
            return feats[:,self.get_abs_max_coef()]
        elif isinstance(ind, int):
            return feats[:,k]
        return feats
            
            
    
class SWstein(NoteMetric):
    def __init__(self, T = 10, nu = 0.1, gamma = 0.1, lam = 0.01, n_it=100):
        self.T = T
        self.nu = nu
        self.gamma = gamma
        self.lam = lam
        self.n_it = n_it
        super().__init__()
        
    def dist_feats_calc(self, feat1, feat2):
        _,_,dist,_= relax(feat1,feat2,T = self.T, nu = self.nu, gamma = self.gamma, lam = self.lam, n_it=self.n_it)
        return dist