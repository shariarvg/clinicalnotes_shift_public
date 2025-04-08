'''
Assumes ep is a perfect design matrix
'''
import pandas as pd
import numpy as np
import joblib
import re
import sys
import mmd_tools
import mimic_tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings

class TabularModel():
    def __init__(self, dataset, model_constructor, exogs, endog):
        self.model = model_constructor()
        self.exogs = exogs
        self.endog = endog
        self.dataset = dataset[exogs + [endog]]
        
    def normalize_exogs(self):
        scaler = MinMaxScaler()
        output = self.dataset[self.endog]
        self.dataset = pd.DataFrame(scaler.fit_transform(self.dataset[self.exogs]), columns = self.exogs)
        self.dataset[self.endog] = output.values
        del output
        
    def fit(self):
        self.model.fit(self.dataset[self.exogs], self.dataset[self.endog])
        
    def get_exogs_sorted_by_coefs(self):
        coefs = self.model.coef_
        if len(coefs.shape) == 2:
            warnings.warn(f"Coefs vector has shape {coefs.shape}")
            coefs = coefs.ravel()
        return list(np.array(self.exogs)[(np.argsort(coefs))[::-1]])
    
    def return_coef(self):
        return self.model.coef_
    

        