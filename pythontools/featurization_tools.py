import pandas as pd
import joblib
import re
import sys
import mmd_tools
from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, SparsePCA
import time
from abc import ABC, abstractmethod
import numpy as np
from sae import SparseAutoencoder
import torch
from model_with_classifier import ModelWithClassifier
from safetensors.torch import load_file

class Featurizer(ABC):
    def __init__(self):
        self.N_times_used = 0
        self.S = 0 
    
    @abstractmethod
    def transform_c(self, texts):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    def transform(self, texts):
        start = time.time()
        transformed = self.transform_c(texts)
        end = time.time()
        self.update_runtime_statistics(end - start)
        return transformed
    
    def update_runtime_statistics(self, t):
        self.N_times_used += 1.0
        self.S += t
        
    def get_avg_runtime(self, npround = 2):
        return np.round(self.S/self.N_times_used,npround)

class BOW(Featurizer):
    def __init__(self, min_df = 0.1, max_df = 0.95, binary = True):
        self.cv = CountVectorizer(min_df = min_df, max_df = max_df, binary = binary)
        self.been_fit = False
        self.name = f"BOW({min_df}, {max_df})"
        super().__init__()
        
    def transform_c(self, texts):
        if self.been_fit:
            return self.cv.transform(texts).toarray()
        self.been_fit = True
        return self.cv.fit_transform(texts).toarray()
    
    def save(self, file_path):
        joblib.dump(self.cv, file_path)
        
    def reset(self):
        self.been_fit = False
        
class DimReduce(Featurizer):
    def __init__(self, n_components = 20, sparse = False):
        self.been_fit = False
        self.name = f"PCA_{20}"
        if sparse:
            self.pca = SparsePCA(n_components)
        else:
            self.pca = PCA(n_components)
        super().__init__()
        
    def transform_c(self, embeddings):
        if self.been_fit:
            return self.pca.transform(embeddings)
        self.been_fit = True
        return self.pca.fit_transform(embeddings)
    
    def reset(self):
        self.been_fit = False
    
    
class Transformer(Featurizer):
    def __init__(self, model_name = "UFNLP/gatortron-base", summary = "mean", max_length = 100, truncation_side = 'right', adapter_path = None):
        self.model_name = model_name
        self.summary = summary
        self.max_length = max_length
        m = model_name.replace("/","_").replace("-","_")
        self.name = f"{m}_{summary}_{max_length}"
        self.model = AutoModel.from_pretrained(self.model_name).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.truncation_side = truncation_side
        self.adapter_path = adapter_path
        super().__init__()
            
        
    def transform_c(self, texts):
        if not isinstance(texts, list):
            texts = list(texts)
        return mmd_tools.get_doc_embeddings_from_model(texts, self.model, self.tokenizer, summary = self.summary, max_length = self.max_length, adapter_path = self.adapter_path)
    
    def reset(self):
        pass
    
class TaskTunedTransformer(Featurizer):
    def __init__(self, classifier_path, model_name = "UFNLP/gatortron-base", summary = "mean", max_length = 100):
        self.mwc = ModelWithClassifier(model_name, 2).to('cuda')#.from_pretrained(classifier_path)
        self.mwc.load_state_dict(load_file(classifier_path + "/model.safetensors"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summary = summary
        self.max_length = max_length
        super().__init__()
        
        
    def transform_c(self, texts):
        if not isinstance(texts, list):
            texts = list(texts)
        return mmd_tools.get_doc_embeddings_from_model(texts, self.mwc.base_model, self.tokenizer, summary = self.summary, max_length = self.max_length)
    
    def reset(self):
        pass

class TransformerWithDimReduce(Featurizer):
    def __init__(self, model_name = "UFNLP/gatortron-base", summary = "mean", max_length = 100, n_components = 50, sparse = False, truncation_side = 'right'):
        self.transformer = Transformer(model_name, summary, max_length, truncation_side)
        self.dimreducer = DimReduce(n_components = n_components, sparse = sparse)
        super().__init__()
            
    def transform_c(self, texts):
        return self.dimreducer.transform(self.transformer.transform(texts))
    
    def reset(self):
        self.transformer.reset()
        self.dimreducer.reset()
        
class Sparsify(Featurizer):
    def __init__(self, batch_size_sae = 16, path = '../../sae_V0.pth',input_dim = 1024, hidden_dim = 1000):
        self.batch_size_sae = batch_size_sae
        self.sae = SparseAutoencoder(input_dim, hidden_dim).to('cuda')
        self.sae.load_state_dict(torch.load(path))
        super().__init__()
        
    def transform_c(self, embeddings):
        embeddings_sae = []
        with torch.no_grad():
            for i in range(0, len(embeddings), self.batch_size_sae):
                batch_embeddings = torch.tensor(embeddings[i:i+self.batch_size_sae], dtype = torch.float32).to('cuda')
                _, e, _ = self.sae(batch_embeddings)
                embeddings_sae.append(e.cpu().numpy())
        
        return np.concatenate(embeddings_sae, axis = 0)
    
    def reset():
        pass
    
class TransformerSparseAutoencoder(Featurizer):
    def __init__(self,  batch_size_sae = 16):
        self.transformer = Transformer()
        self.sae = Sparsify(batch_size_sae)
        super().__init__()
        
    def transform_c(self, texts):
        embeddings = self.transformer.transform(texts)
        sparse_encodings = self.sae.transform(embeddings)
        return sparse_encodings
    
    def reset(self):
        self.transformer.reset()
        