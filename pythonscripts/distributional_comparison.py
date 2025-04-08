import torch
import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import json
from datetime import datetime
import argparse

# Add the pythontools directory to the path
sys.path.append(os.path.abspath("../pythontools"))

# Import necessary modules
from mimic_tools import MIMICEndpoint
from featurization_tools import *
from sae import SparseAutoencoder
from sae_tools import get_activation_or_not
from mimic_source import MIMICMultiSource, MIMICSource, MIMICMixtureSource
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from notes_dataset import NotesDataset
from model_with_classifier import ModelWithClassifier


def get_train_test_sets(ep, source1, source2, train_size=500, test_size=100):
    """Get balanced train and test sets for the specified task."""
    # Get positive and negative samples
    train_set = source1.obtain_samples(train_size)
    test_set = source2.obtain_samples(test_size)
    return train_set, test_set

def get_embeddings(notes, featurizer):
    """Get embeddings for the notes using the featurizer."""
    embeddings = featurizer.transform(notes['text'])
    return embeddings

def train_sae(embeddings, hidden_dim=1000, sparsity_lambda=0.001, n_epochs=100, batch_size=8, learning_rate=0.001):
    """Train a sparse autoencoder on the embeddings."""
    model = SparseAutoencoder(input_dim=embeddings.shape[1], hidden_dim=hidden_dim, sparsity_lambda=sparsity_lambda)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    for epoch in range(n_epochs):
        epoch_mse_loss = 0.0
        epoch_sparsity_loss = 0.0
        
        # Process in batches
        for i in range(0, len(embeddings_tensor), batch_size):
            batch = embeddings_tensor[i:i+batch_size]
            
            # Forward pass
            decoded, encoded, sparsity_penalty = model(batch)
            
            # Calculate loss
            mse_loss = criterion(decoded, batch)
            loss = mse_loss + sparsity_penalty
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_mse_loss += mse_loss.item()
            epoch_sparsity_loss += sparsity_penalty.item()
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{n_epochs}, MSE Loss: {epoch_mse_loss/len(embeddings_tensor):.6f}, "
              f"Sparsity Loss: {epoch_sparsity_loss/len(embeddings_tensor):.6f}")
    
    return model

def get_sparse_encodings(embeddings, sae_model, batch_size=32):
    """Get sparse encodings for the embeddings using the trained SAE model."""
    sae_model.eval()
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    sparse_encodings = []
    
    with torch.no_grad():
        for i in range(0, len(embeddings_tensor), batch_size):
            batch = embeddings_tensor[i:i+batch_size]
            _, encoded, _ = sae_model(batch)
            sparse_encodings.append(encoded.numpy())
    
    return np.vstack(sparse_encodings)

def find_most_different_dimension(train_encodings, test_encodings, method='ks'):
    """Find the dimension with the largest distributional difference between train and test."""
    n_dimensions = train_encodings.shape[1]
    differences = []
    
    for i in range(n_dimensions):
        train_dim = train_encodings[:, i]
        test_dim = test_encodings[:, i]
        
        if method == 'ks':
            # Kolmogorov-Smirnov test
            statistic, _ = stats.ks_2samp(train_dim, test_dim)
            differences.append(statistic)
        elif method == 'mutual_info':
            # Mutual information with binarized data
            train_binary = get_activation_or_not(train_dim)
            test_binary = get_activation_or_not(test_dim)
            
            # Calculate mutual information between binarized data and source (train vs test)
            train_source = np.zeros(len(train_binary))
            test_source = np.ones(len(test_binary))
            
            mi_train = mutual_info_score(train_binary, train_source)
            mi_test = mutual_info_score(test_binary, test_source)
            
            differences.append(abs(mi_train - mi_test))
        elif method == 't_test':
            # T-test
            statistic, _ = stats.ttest_ind(train_dim, test_dim)
            differences.append(abs(statistic))
    
    # Find the dimension with the largest difference
    most_different_dim = np.argmax(differences)
    return most_different_dim, differences[most_different_dim]

def get_top_activations(encodings, note_ids, dimension, top_k=5):
    """Get the note IDs with the largest activations for a specific dimension."""
    activations = encodings[:, dimension]
    top_indices = np.argsort(activations)[-top_k:]
    return note_ids.iloc[top_indices], activations[top_indices]

def plot_distribution_comparison(train_encodings, test_encodings, dimension, save_path=None):
    """Plot the distribution of activations for a specific dimension in train and test sets."""
    plt.figure(figsize=(10, 6))
    plt.hist(train_encodings[:, dimension], bins=30, alpha=0.5, label='Train', density=True)
    plt.hist(test_encodings[:, dimension], bins=30, alpha=0.5, label='Test', density=True)
    plt.xlabel(f'Activation Value (Dimension {dimension})')
    plt.ylabel('Density')
    plt.title(f'Distribution of Activations for Dimension {dimension}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def train_classifier(train_set, task, save_path, n_epochs=10, batch_size=16, learning_rate=2e-5):
    """Train a new classifier on the train set."""
    print(f"Training new classifier for task: {task}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
    base_model = AutoModel.from_pretrained("UFNLP/gatortron-base")
    model = ModelWithClassifier(base_model, 2)  # Binary classification
    
    # Create datasets
    train_dataset = NotesDataset(train_set, task, tokenizer=tokenizer, max_length=512)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        label_names=[task]
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(save_path)
    print(f"Classifier saved to {save_path}")
    
    # Create a custom TaskTunedTransformer that uses this model
    class CustomTaskTunedTransformer:
        def __init__(self, model_path, tokenizer_path=None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load the base model directly from the saved path
            try:
                # Try to load the model directly
                self.model = ModelWithClassifier(AutoModel.from_pretrained(model_path), 2)
            except ValueError:
                # If that fails, load the base model and then load the state dict
                print("Loading model using alternative method...")
                base_model = AutoModel.from_pretrained("UFNLP/gatortron-base")
                self.model = ModelWithClassifier(base_model, 2)
                
                # Check if the model is saved as safetensors or pytorch_model.bin
                safetensors_path = f"{model_path}/model.safetensors"
                pytorch_path = f"{model_path}/pytorch_model.bin"
                
                if os.path.exists(safetensors_path):
                    print(f"Loading model from {safetensors_path}")
                    from safetensors.torch import load_file
                    state_dict = load_file(safetensors_path)
                    self.model.load_state_dict(state_dict)
                elif os.path.exists(pytorch_path):
                    print(f"Loading model from {pytorch_path}")
                    state_dict = torch.load(pytorch_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                else:
                    raise FileNotFoundError(f"No model file found at {model_path}. Expected either model.safetensors or pytorch_model.bin")
            
            self.model.to(self.device)
            self.model.eval()
            
            if tokenizer_path:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
        
        def transform(self, texts):
            embeddings = []
            
            with torch.no_grad():
                for text in texts:
                    # Tokenize
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get hidden states
                    outputs = self.model.base_model(**inputs, output_hidden_states=True)
                    
                    # Use the last hidden state as the embedding
                    # Shape: [batch_size, seq_len, hidden_size]
                    last_hidden_state = outputs.hidden_states[-1]
                    
                    # Use mean pooling over the sequence length
                    # Shape: [batch_size, hidden_size]
                    pooled_output = last_hidden_state.mean(dim=1)
                    
                    # Convert to numpy and append
                    embeddings.append(pooled_output.cpu().numpy()[0])
            
            return np.array(embeddings)
    
    # Create and return the custom transformer
    return CustomTaskTunedTransformer(save_path)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run distributional comparison analysis')
    parser.add_argument('--commit_hash', type=str, default='latest', help='Commit hash for labeling files')
    args = parser.parse_args()
    
    commit_hash = args.commit_hash
    commit_link = f"https://github.com/shariarvg/clinicalnotes_shift/commit/{commit_hash}"
    
    # Record start time
    t_start = datetime.now()
    
    # Parameters
    train_size = 500
    test_size = 500
    hidden_dim = 1000
    sparsity_lambda = 0.001
    batch_size = 4
    learning_rate = 0.001
    comparison_method = 'ks'  # Options: 'ks', 'mutual_info', 't_test'
    top_k = 5
    n_epochs = 10
    task='admission_in_30_days'
    
    # Initialize MIMIC endpoint
    ep = MIMICEndpoint()
    
    # Create sources for train and test sets
    source1_key = "get_notes_start_age_greaterthan"
    source2_key = "get_notes_start_age_lessthanorequalto"
    source_param1 = 75
    source_param2 = 40
    source_a = MIMICSource(ep, source1_key, source_param1)
    source_b = MIMICSource(ep, source2_key, source_param2)
    source1 = MIMICMixtureSource([source_a, source_b], [0.8, 0.2])
    source2 = MIMICMixtureSource([source_a, source_b], [0.2, 0.8])
    
    # Get train and test sets
    print("Getting train and test sets...")
    train_set, test_set = get_train_test_sets(ep, source1, source2, train_size, test_size)
    
    # Save train and test note IDs
    print("Saving train and test note IDs...")
    train_set[['note_id']].to_csv(f"../../train_note_ids_{task}_{commit_hash}.csv", index=False)
    test_set[['note_id']].to_csv(f"../../test_note_ids_{task}_{commit_hash}.csv", index=False)
    
    # Train a new classifier on the train set
    classifier_save_path = f"../../custom_classifier_{task}_{commit_hash}"
    featurizer = train_classifier(train_set, task, classifier_save_path, batch_size = batch_size, n_epochs=n_epochs)
    
    # Get embeddings for train set
    print("Getting embeddings for train set...")
    train_embeddings = get_embeddings(train_set, featurizer)
    
    # Train SAE on train embeddings
    print("Training SAE...")
    sae_model = train_sae(
        train_embeddings, 
        hidden_dim=hidden_dim, 
        sparsity_lambda=sparsity_lambda, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        learning_rate=learning_rate
    )
    
    # Save the trained SAE model
    sae_path = f"../../sae_custom_{task}_{hidden_dim}_topk_{commit_hash}.pth"
    torch.save(sae_model.state_dict(), sae_path)
    print(f"SAE model saved to {sae_path}")
    
    # Get embeddings for test set
    print("Getting embeddings for test set...")
    test_embeddings = get_embeddings(test_set, featurizer)
    
    # Get sparse encodings for train and test sets
    print("Getting sparse encodings...")
    train_sparse = get_sparse_encodings(train_embeddings, sae_model, batch_size=batch_size)
    test_sparse = get_sparse_encodings(test_embeddings, sae_model, batch_size=batch_size)
    
    # Save sparse encodings with note IDs
    print("Saving sparse encodings with note IDs...")
    # Create DataFrames with note IDs and sparse encodings
    train_sparse_df = pd.DataFrame(train_sparse, columns=[f"dim_{i}" for i in range(train_sparse.shape[1])])
    train_sparse_df['note_id'] = train_set['note_id'].values
    
    test_sparse_df = pd.DataFrame(test_sparse, columns=[f"dim_{i}" for i in range(test_sparse.shape[1])])
    test_sparse_df['note_id'] = test_set['note_id'].values
    
    # Save to CSV
    train_sparse_df.to_csv(f"../../train_sparse_encodings_{task}_{commit_hash}.csv", index=False)
    test_sparse_df.to_csv(f"../../test_sparse_encodings_{task}_{commit_hash}.csv", index=False)
    
    # Find the dimension with the largest distributional difference
    print(f"Finding most different dimension using {comparison_method} method...")
    most_different_dim, difference_value = find_most_different_dimension(
        train_sparse, test_sparse, method=comparison_method
    )
    print(f"Most different dimension: {most_different_dim} (difference value: {difference_value:.6f})")
    
    # Get top activations for the most different dimension
    print("Getting top activations...")
    train_top_notes, train_top_activations = get_top_activations(
        train_sparse, train_set['note_id'], most_different_dim, top_k=top_k
    )
    test_top_notes, test_top_activations = get_top_activations(
        test_sparse, test_set['note_id'], most_different_dim, top_k=top_k
    )
    
    # Plot distribution comparison
    print("Plotting distribution comparison...")
    plot_distribution_comparison(
        train_sparse, test_sparse, most_different_dim, 
        save_path=f"../../dimension_{most_different_dim}_distribution_{commit_hash}.png"
    )
    
    # Save results
    results = {
        'most_different_dimension': most_different_dim,
        'difference_value': difference_value,
        'train_top_notes': train_top_notes.tolist(),
        'train_top_activations': train_top_activations.tolist(),
        'test_top_notes': test_top_notes.tolist(),
        'test_top_activations': test_top_activations.tolist()
    }
    
    # Save results to CSV
    pd.DataFrame({
        'note_id': train_top_notes.tolist() + test_top_notes.tolist(),
        'activation': train_top_activations.tolist() + test_top_activations.tolist(),
        'source': ['train'] * top_k + ['test'] * top_k
    }).to_csv(f"../../dimension_{most_different_dim}_top_activations_{commit_hash}.csv", index=False)
    
    # Save the most different dimension information
    pd.DataFrame({
        'dimension': [most_different_dim],
        'difference_value': [difference_value],
        'comparison_method': [comparison_method]
    }).to_csv(f"../../most_different_dimension_{task}_{commit_hash}.csv", index=False)
    
    # Record end time and calculate runtime
    t_end = datetime.now()
    runtime_seconds = (t_end - t_start).total_seconds()
    
    # Create configuration dictionary
    config = {
        "task": task,
        "train_size": train_size,
        "test_size": test_size,
        "hidden_dim": hidden_dim,
        "sparsity_lambda": sparsity_lambda,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "comparison_method": comparison_method,
        "top_k": top_k,
        "most_different_dimension": int(most_different_dim),
        "difference_value": float(difference_value),
        "source1_key": source1_key,
        "source2_key": source2_key,
        "hash": commit_hash,
        "link": commit_link,
        "time_finished": t_end.isoformat(),
        "seconds_runtime": runtime_seconds,
        "source_param1": source_param1,
        "source_param2": source_param2,
        "mixture_weights1": [0.8, 0.2],
        "mixture_weights2": [0.2, 0.8]
    }
    
    # Save configuration to JSON
    config_path = "../../distributional_comparison_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    data.append(config)
    
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Results saved to ../../dimension_{most_different_dim}_top_activations_{commit_hash}.csv")
    print(f"Distribution plot saved to ../../dimension_{most_different_dim}_distribution_{commit_hash}.png")
    print(f"Train note IDs saved to ../../train_note_ids_{task}_{commit_hash}.csv")
    print(f"Test note IDs saved to ../../test_note_ids_{task}_{commit_hash}.csv")
    print(f"Train sparse encodings saved to ../../train_sparse_encodings_{task}_{commit_hash}.csv")
    print(f"Test sparse encodings saved to ../../test_sparse_encodings_{task}_{commit_hash}.csv")
    print(f"Most different dimension saved to ../../most_different_dimension_{task}_{commit_hash}.csv")
    print(f"Configuration saved to {config_path}")
    
    return results

if __name__ == "__main__":
    main() 