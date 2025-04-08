import pandas as pd
import numpy as np
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

class AttackExperiment():
    def __init__(self, training_source, testing_source, N_train, N_test, attack, note_metrics, loss_fn, untrained_model, note_featurizer, target):
        self.training_source = training_source
        self.testing_source = testing_source
        self.N_train = N_train
        self.N_test = N_test
        self.attack = attack
        self.note_metrics = note_metrics
        self.loss_fn = loss_fn
        self.model = untrained_model
        self.note_featurizer = note_featurizer
        self.target = target
    
    def sample_training_set(self):
        return self.training_source.obtain_samples(self.N_train)
    
    def sample_attacked_training_set(self, add_back = True):
        return self.attack.get_attacked_sample(self.training_source, self.N_train, not add_back)
    
    def OLD_sample_attacked_testing_set(self, add_back = True):
        return self.attack.get_attacked_sample(self.testing_source, self.N_test, not add_back)
    
    def sample_attacked_testing_set(self, with_feats = True):
        sample = self.attack.get_attacked_sample(self.testing_source, self.N_test)
        if not with_feats:
            return sample
        return sample, self.note_featurizer.transform(sample['text'])
    
    def OLD_sample_attacked_training_set(self, add_back = True, pre_attacked = False):
        if not pre_attacked:
            self.attack.attack_source_ep()
        ret = self.testing_source.obtain_samples(self.N_train)
        if add_back:
            self.attack.add_back_removed_notes()
        return ret
    
    def sample_unattacked_testing_set(self, with_feats = True):
        if not with_feats:
            return self.testing_source.obtain_samples(self.N_test)
        sample = self.testing_source.obtain_samples(self.N_test)
        return sample, self.note_featurizer.transform(sample['text'])
    
    def OLD_sample_attacked_testing_set(self, add_back = True, pre_attacked = False):
        if not pre_attacked:
            self.attack.attack_source_ep()
        ret = self.testing_source.obtain_samples(self.N_test)
        if add_back:
            self.attack.add_back_removed_notes()
        return ret
    
    def get_note_metric(self, train, test):
        #if len(self.note_metrics)==1:
        #    return self.note_metrics[0].dist_datasets(train, test)
        return np.array([nm.dist_feats(train, test) for nm in self.note_metrics])
    
    def get_note_metric_diff(self, train, test1, test2):
        return self.get_note_metric(train, test1) - self.get_note_metric(train, test2)
    
    def get_training_set_and_trained_model(self):
        train = self.training_source.obtain_samples(self.N_train)
        feats = self.note_featurizer.transform(list(train['text']))

        self.model.fit(feats, train[self.target].values)
        return train, feats, self.model
    
    def get_attacked_training_set_and_trained_model(self, add_back = True):
        train = self.sample_attacked_training_set(add_back = add_back)
        feats = self.note_featurizer.transform(list(train['text']))
        self.model.fit(feats, train[self.target].values)
        return train, feats, self.model
              
    def get_loss(self, test, test_feat):
        return self.loss_fn(self.model, test_feat, self.target, test)
    
    def get_loss_diff(self, test1, test2):
        return self.get_loss(test1) - self.get_loss(test2)
    
    def reset_ep(self):
        self.training_source.reset()
        self.testing_source.reset()
    
    def reset_featurizer(self):
        self.note_featurizer.reset()
        
    def print_avg_metric_times(self):
        return "Average metric times: " + ",".join([str(metric.get_avg_runtime()) for metric in self.note_metrics])
    
        