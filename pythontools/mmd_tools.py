'''
This file contains methods for
-- obtaining embeddings of documents
-- taking MMD between two datasets
-- performing an MMD permutation between two datasets
-- preprocessing text
-- identifying the power of the MMD for two datasets, under a specific dimension
'''

import torch

from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from sklearn.decomposition import PCA
import scipy
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import torch.nn as nn
import joblib
from peft import PeftModel

class TextPCA():
    '''
    Just a wrapper on PCA that allows you to call transform or fit transform regardless of whether it's been fitted,
    and also allows you to create the degenerate PCA that doesn't do anything (n_components > 1000).
    '''
    def __init__(self, n_components):
        if n_components > 1000:
            self.pca = None
        else:
            self.pca = PCA(n_components = n_components)
        self.fitted = False
        
    def transform(self, data):
        if self.pca is None:
            return data
        if self.fitted:
            return self.pca.transform(data)
        self.fitted = True
        return self.pca.fit_transform(data)
    
class ModelWithClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ModelWithClassifier, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)  # Regularization
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Pass inputs through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Return the full output of the base model, which includes last_hidden_state and other attributes
        return outputs
    
        

def mmd_pipeline(notes1, notes2, mmd_method, model_name = "UFNLP/gatortron-base", batch_size_embedding = 50, summary_embedding = "mean", pca_embedding = None, iterations = 1, **mmd_kwargs):
    if not isinstance(notes1, list):
        notes1 = list(notes1['text'])
        notes2 = list(notes2['text'])
    if iterations == 1:
        embeddings1 = get_doc_embeddings(notes1, model_name, batch_size = batch_size_embedding, summary = summary_embedding, pca = pca_embedding)
        embeddings2 = get_doc_embeddings(notes2, model_name, batch_size = batch_size_embedding, summary = summary_embedding, pca = pca_embedding)
        return mmd_method(embeddings1, embeddings2, **mmd_kwargs)
    else:
        s = 0.0
        for it in range(iterations):
            s += mmd_pipeline(notes1, notes2, mmd_method, model_name, batch_size_embedding, summary_embedding, pca_embedding, 1, **mmd_kwargs)
        return s/iterations
        
def get_doc_embeddings_from_model(input_text, model, tokenizer, max_length = 100, summary = 'mean', batch_size = 50, adapter_path = None):
    inputs = tokenizer(input_text, return_tensors='pt', padding = 'max_length', max_length = max_length, truncation = True).to(device)
    if adapter_path is not None:
        # Load LoRA adapter (will merge with base model)
        model = PeftModel.from_pretrained(model, adapter_path)
        
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(inputs["input_ids"]), batch_size):
            batch_inputs = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}
            
            if summary == 'mean':
                # Get the last hidden state from the model
                last_hidden_state = model(**batch_inputs).last_hidden_state  # Shape: (batch_size, seq_length, hidden_dim)

                '''
                As of 1/5/25, ensuring that mean is only pooled over the non-padded tokens.
                '''

                # Use the attention mask to exclude padding tokens
                attention_mask = batch_inputs['attention_mask']  # Shape: (batch_size, seq_length)

                # Expand the mask to match the hidden state dimensions
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())  # Shape: (batch_size, seq_length, hidden_dim)

                # Apply the mask to the hidden states
                masked_hidden_state = last_hidden_state * attention_mask_expanded

                # Compute the mean of non-padding tokens
                sentence_lengths = attention_mask.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
                batch_output = masked_hidden_state.sum(dim=1) / sentence_lengths  # Shape: (batch_size, hidden_dim)
            elif summary == 'last':
                batch_output = model(**batch_inputs).last_hidden_state
                # Get the lengths of each sequence in the batch (sum of attention_mask for each sequence)
                sequence_lengths = batch_inputs['attention_mask'].sum(dim=1) - 1  # Subtract 1 to get the last valid index
                # Gather the last hidden state for the last token in each sequence
                batch_output = batch_output[torch.arange(batch_output.size(0)), sequence_lengths]
                
            elif summary == 'first':
                batch_output = model(**batch_inputs).last_hidden_state[:, 0, :]
                    
            all_embeddings.append(batch_output.cpu().numpy())  
    return np.concatenate(all_embeddings, axis = 0)
            
    

def get_doc_embeddings(input_text, model_name = "UFNLP/gatortron-base", max_length = 100, vectorizer = None, batch_size = 50, summary = 'mean', pca = None, model = None):
    '''
    Obtain the embeddings of every document in a corpus, using an inputted model.
    Optionally, can also pass in a countvectorizer to obtain the embeddings, instead of a transformer model name
    '''
    if isinstance(model_name, list):
        ## assuming for now there's only two elements in the list
        embeddings1 = get_doc_embeddings(input_text, model_name[0], max_length, vectorizer, batch_size, summary, pca)
        embeddings2 = get_doc_embeddings(input_text, model_name[1], max_length, vectorizer, batch_size, summary, pca)
        return np.hstack([embeddings1, embeddings2])
        
    if "sentence" in model_name.lower():
        model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-Marco')
    elif "fine_tuned_gatortron" in model_name:
        fine_tuned_gatortron_dir = "../../" + model_name
        # Load the saved model and tokenizer
        model = AutoModel.from_pretrained(fine_tuned_gatortron_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_gatortron_dir)
        inputs = tokenizer(input_text, return_tensors='pt', padding = 'max_length', max_length = max_length, truncation = True).to(device)
    elif "mortality" in model_name or "death" in model_name:
        base = AutoModel.from_pretrained("UFNLP/gatortron-base").to(device)
        tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
        model = GatorTronWithClassifier(base, 2).to(device)
        model.load_state_dict(torch.load("../../gatortron_death_classifier_chkpt_epoch20_V2.pt"))
        inputs = tokenizer(input_text, return_tensors='pt', padding = 'max_length', max_length = max_length, truncation = True).to(device)
    elif "readmission" in model_name or "death" in model_name:
        base = AutoModel.from_pretrained("UFNLP/gatortron-base").to(device)
        tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
        model = GatorTronWithClassifier(base, 2).to(device)
        model.load_state_dict(torch.load("../../gatortron_readmission_classifier_chkpt_epoch0_V4.pt"))
        inputs = tokenizer(input_text, return_tensors='pt', padding = 'max_length', max_length = max_length, truncation = True).to(device)
    
    elif "gpt2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        inputs = tokenizer(input_text, return_tensors='pt', padding = 'max_length', max_length = max_length, truncation = True).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        inputs = tokenizer(input_text, return_tensors='pt', padding = 'max_length', max_length = max_length, truncation = True).to(device)
    if vectorizer is None:
        all_embeddings = []
        
            
        with torch.no_grad():
            if "sentence" in model_name.lower():
                all_embeddings = model.encode(input_text, batch_size=batch_size, device=device, show_progress_bar=False)
                return all_embeddings
            else:
                for i in range(0, len(inputs["input_ids"]), batch_size):
                    batch_inputs = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}
                    # Get the embeddings for this batch
                    if summary == 'mean':
                        # Get the last hidden state from the model
                        last_hidden_state = model(**batch_inputs).last_hidden_state  # Shape: (batch_size, seq_length, hidden_dim)

                        '''
                        As of 1/5/25, ensuring that mean is only pooled over the non-padded tokens.
                        '''

                        # Use the attention mask to exclude padding tokens
                        attention_mask = batch_inputs['attention_mask']  # Shape: (batch_size, seq_length)

                        # Expand the mask to match the hidden state dimensions
                        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())  # Shape: (batch_size, seq_length, hidden_dim)

                        # Apply the mask to the hidden states
                        masked_hidden_state = last_hidden_state * attention_mask_expanded

                        # Compute the mean of non-padding tokens
                        sentence_lengths = attention_mask.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
                        batch_output = masked_hidden_state.sum(dim=1) / sentence_lengths  # Shape: (batch_size, hidden_dim)

                    if summary == 'last':
                        batch_output = model(**batch_inputs).last_hidden_state
                        # Get the lengths of each sequence in the batch (sum of attention_mask for each sequence)
                        sequence_lengths = batch_inputs['attention_mask'].sum(dim=1) - 1  # Subtract 1 to get the last valid index
                        # Gather the last hidden state for the last token in each sequence
                        batch_output = batch_output[torch.arange(batch_output.size(0)), sequence_lengths]
                    if summary == 'first':
                        batch_output = model(**batch_inputs).last_hidden_state[:, 0, :]
                    all_embeddings.append(batch_output.cpu().numpy())  # Move embeddings to CPU to save GPU memory
        if not pca:
            return np.concatenate(all_embeddings, axis = 0)
        return pca.transform(np.concatenate(all_embeddings, axis = 0))
    else:
        return vectorizer.transform(input_text).toarray()
    
def get_doc_bow_and_prediction(input_text, model_filepath):
    cv = joblib.load(model_filepath+"_cv.pt")
    rfc = joblib.load(model_filepath+"_rfc.pt")
    embs = cv.transform(input_text).toarray()
    preds = rfc.predict_proba(embs)
    return embs, preds
    
def get_doc_embeddings_and_prediction(input_text, model_filepath, model_base_name, summary = 'first', max_length = 100, batch_size = 16):
    '''
    For the task-fine-tuned prediction task
    '''
    if "bow" in model_base_name.lower():
        return get_doc_bow_and_prediction(input_text, model_filepath)
    mbn_to_extended_name = {"gpt2": "gpt2", "gtron": "UFNLP/gatortron-base"}
    base = AutoModel.from_pretrained(mbn_to_extended_name[model_base_name]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mbn_to_extended_name[model_base_name])
    if "gpt" in model_base_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base.resize_token_embeddings(len(tokenizer))
    model = ModelWithClassifier(base, 2).to(device)
    model.load_state_dict(torch.load(model_filepath))
    
    inputs = tokenizer(input_text, return_tensors='pt', padding = 'max_length', max_length = max_length, truncation = True).to(device)
    all_embeddings = []
    all_logits = torch.empty(0, dtype=torch.float32, device=device)  # Initialized on GPU to accumulate
    
    
    for i in range(0, len(inputs["input_ids"]), batch_size):
        batch_inputs = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}
        outputs = model(**batch_inputs)
        last_hidden_state = outputs.last_hidden_state[:,0,:]
        all_embeddings.append(last_hidden_state.cpu().detach().numpy())
        logit = model.classifier(model.dropout(last_hidden_state))
        all_logits = torch.cat((all_logits, logit.squeeze(dim = -1).detach()))
    return np.vstack(all_embeddings), all_logits.cpu().numpy()


def mmd_permutation_test(X,Y,number_bootstraps = 1000, size = 0.05, ret = False, ret_quantile = False, ret_sd = False, ret_null = False):
    """
    Returns (1 if rejected, 0 if not rejected)#, mmd, and threshold
    """
    XX = scipy.spatial.distance.cdist(X,X,metric = 'euclidean')
    XY = scipy.spatial.distance.cdist(X,Y,metric = 'euclidean')
    YY = scipy.spatial.distance.cdist(Y,Y,metric = 'euclidean')
    top_row = np.block([[XX, XY]])
    bottom_row = np.block([[XY.T, YY]])
    Z = np.block([[top_row], [bottom_row]])
    upper_triangle = np.triu_indices(Z.shape[0], k=1)
    Zupp = Z[upper_triangle]
    sigma = np.median(Zupp)
    ##Recreate Z
    XX = np.exp(-1/(2*sigma) * XX)
    XY = np.exp(-1/(2*sigma) * XY)
    YY = np.exp(-1/(2*sigma) * YY)
    top_row = np.block([[XX, XY]])
    bottom_row = np.block([[XY.T, YY]])
    Z = np.block([[top_row], [bottom_row]])
    
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()

    mmds = np.zeros((number_bootstraps, ))

    for b in range(number_bootstraps):
        xinds = np.random.choice(np.arange(0,X.shape[0]+Y.shape[0],1),size = X.shape[0], replace = False)
        yinds = np.delete(np.arange(0,X.shape[0]+Y.shape[0],1), xinds)
        XXb = Z[xinds[:, None], xinds].mean()
        YYb = Z[yinds[:, None], yinds].mean()
        XYb = Z[xinds[:, None], yinds].mean()
        mmds[b] = XXb + YYb - 2*XYb

    threshold = np.quantile(mmds, 1-size)
    
    if ret_null:
        return mmd, mmds
    if ret and ret_quantile and ret_sd:
        return mmd, np.mean(mmd < mmds), (mmd - np.mean(mmds))/np.std(mmds)
    if ret and ret_quantile:
        return mmd, np.mean(mmd < mmds)
    if ret and ret_sd:
        return mmd, (mmd - np.mean(mmds))/np.std(mmds)
    if ret_sd:
        return (mmd - np.mean(mmds))/np.std(mmds)
    if ret_quantile:
        return np.mean(mmd < mmds)
    if ret:
        return mmd
    return int(mmd > threshold)

def mmd_calc(X,Y, just_mmd = True):
    '''
    Calculate the raw MMD between X and Y
    '''
    XX = scipy.spatial.distance.cdist(X,X,metric = 'euclidean')
    XY = scipy.spatial.distance.cdist(X,Y,metric = 'euclidean')
    YY = scipy.spatial.distance.cdist(Y,Y,metric = 'euclidean')
    top_row = np.block([[XX, XY]])
    bottom_row = np.block([[XY.T, YY]])
    Z = np.block([[top_row], [bottom_row]])
    upper_triangle = np.triu_indices(Z.shape[0], k=1)
    Zupp = Z[upper_triangle]
    sigma = np.median(Zupp)
    ##Recreate Z
    XX = np.exp(-1/(2*sigma) * XX)
    XY = np.exp(-1/(2*sigma) * XY)
    YY = np.exp(-1/(2*sigma) * YY)
    if just_mmd:
        return XX.mean() + YY.mean() - 2*XY.mean()
    return XX.mean(),  YY.mean(),  XY.mean(), XX.mean() + YY.mean() - 2*XY.mean()
# Apply preprocessing to the 'text' column

def power_dimr(notes1, notes2, N_SAMPLES=200, N_RUNS=200, N_BOOTSTRAPS=1000, N_COMPONENTS = 20):
    '''
    Obtain embeddings, reduce them to N_COMPONENTS dimensions with PCA, evaluate MMD test, and repeat
    '''
    rejections = 0
    for count in range(N_RUNS):
        embeddings1 = get_doc_embeddings(list(notes1.sample(N_SAMPLES)))
        embeddings2 = get_doc_embeddings(list(notes2.sample(N_SAMPLES)))
        pca = PCA(n_components = N_COMPONENTS)
        #lle = LocallyLinearEmbedding(n_components=DIM)
        embeddings = pca.fit_transform(np.concatenate([embeddings1,embeddings2]))
        embeddings1 = embeddings[:N_SAMPLES,:]
        embeddings2 = embeddings[N_SAMPLES:,:]
        rejections += mmd_permutation_test(embeddings1, embeddings2, number_bootstraps = N_BOOTSTRAPS)
    return rejections/N_RUNS

def power(notes1, notes2, N_SAMPLES = 200, N_RUNS = 200, N_BOOTSTRAPS = 1000, vectorize = False):
    '''
    Obtain the embeddings (optionally with vectorizer), evaluate MMD test, and repeat
    '''
    rejections = 0.0
    notes1 = notes1.apply(lambda x: preprocess_text(x))
    notes2 = notes2.apply(lambda x: preprocess_text(x))
    for count in range(N_RUNS):
        l1 = notes1.sample(N_SAMPLES)
        l2 = notes2.sample(N_SAMPLES)
        if vectorize:
            vectorizer = CountVectorizer(min_df = 0.05, max_df = 0.95)
            embeddings1 = vectorizer.fit_transform(l1).toarray()
            embeddings2 = get_doc_embeddings(l2, vectorizer = vectorizer)
        else:
            embeddings1 = get_doc_embeddings(l1)
            embeddings1 = get_doc_embeddings(l2)
        rejections += mmd_permutation_test(embeddings1, embeddings2, number_bootstraps = N_BOOTSTRAPS)
    return rejections/N_RUNS