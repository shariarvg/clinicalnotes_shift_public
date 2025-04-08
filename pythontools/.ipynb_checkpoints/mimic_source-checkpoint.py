'''
Wrapper for all the functions that obtain a subset of notes
Class for defining a data source, given a method and a set of parameters
'''

import pandas as pd
import numpy as np
import re
import sys
import mimic_tools
from mimic_tools import MIMICEndpoint

class MIMICSource():
    
    def __init__(self, ep, accessor, *params):
        self.ep = ep
        self.accessor = accessor
        self.params = params
        self.notes_used = pd.DataFrame()
        
    def obtain_samples(self, TOTAL_SIZE = None, notes = None, delete = True):
        if TOTAL_SIZE == 0:
            return pd.DataFrame()
        func = getattr(self.ep, self.accessor)
        samples = func(*self.params, total_size = TOTAL_SIZE, notes = notes)
        self.notes_used = pd.concat([self.notes_used, samples])
        if delete:
            self.ep.delete_notes(samples)
        return samples
    
    def reset(self):
        self.ep.notes = pd.concat([self.ep.notes, self.notes_used])
        self.notes_used = pd.DataFrame()
    
class MIMICMultiSource():
    def __init__(self, iterable_of_sources):
        self.sources = iterable_of_sources
        self.ep = self.sources[0].ep
        self.notes_used = pd.DataFrame()
        for s in self.sources:
            assert self.ep == s.ep ## ONLY one source
        
    def obtain_samples(self, TOTAL_SIZE = None, delete = True):
        if TOTAL_SIZE == 0:
            return pd.DataFrame()
        samples = self.sources[0].obtain_samples(delete = False)
        for source in self.sources[1:-1]:
            samples = source.obtain_samples(notes = samples, delete = False)
        samples = self.sources[-1].obtain_samples(TOTAL_SIZE = TOTAL_SIZE, notes = samples)
        self.notes_used = pd.concat([self.notes_used, samples])
        if delete:
            self.ep.delete_notes(samples)
        return samples
    
    def reset(self):
        self.ep.notes = pd.concat([self.ep.notes, self.notes_used])
        self.notes_used = pd.DataFrame()
        
class MIMICMixtureSource():
    def __init__(self, iterable_of_sources, list_of_weights):
        self.sources = iterable_of_sources
        self.weights = list_of_weights
        self.ep = self.sources[0].ep
        
        for s in self.sources:
            assert self.ep == s.ep ## ONLY one source
        
    def obtain_samples(self, TOTAL_SIZE = None, delete = True):
        #Ns = np.random.binomial(TOTAL_SIZE, self.weights)
        Ns = TOTAL_SIZE * np.array(self.weights)
        samples = pd.concat([self.sources[i].obtain_samples(int(Ns[i]), delete = True) for i, source in enumerate(self.sources)])
            
        assert samples.shape[0] == TOTAL_SIZE, f"samples has shape {samples.shape}"
        
        return samples
    
    def reset(self):
        for source in self.sources:
            source.reset()