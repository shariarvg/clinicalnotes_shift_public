import torch
import numpy as np

def get_activation_or_not(sparse_encodings_column, activation_threshold = 0.5):
    return (sparse_encodings_column > activation_threshold).astype(int)

def get_separation_score(activations_vector, target_column):
    if np.sum(activations_vector) == 0 or np.sum(target_column) == 0:
        return 0
    return np.sum(activations_vector * target_column)/np.sum(activations_vector) - np.sum((1-activations_vector) * (target_column))/np.sum(1-activations_vector)

def precision_recall(activations_vector, binary_feature):
    if np.sum(activations_vector) == 0 or np.sum(binary_feature) == 0:
        return 0, 0
    return np.sum(activations_vector * binary_feature)/np.sum(activations_vector), np.sum(activations_vector * binary_feature)/np.sum(binary_feature)

def get_fidelity(binary_feature, activations_vector):
    precision, recall = precision_recall(activations_vector, binary_feature)
    s = len(binary_feature)
    min_sparsity = np.min([(s-np.sum(binary_feature))/s, (s-np.sum(activations_vector))/s])
    if min_sparsity == 0:
        return 1
    return 1 - min([precision, recall])/min_sparsity

def get_lower_bound_sep_score(binary_feature, sparse_encodings_column, activation_threshold = 0.5):
    activations_vector = get_activation_or_not(sparse_encodings_column, activation_threshold)
    sep_score_Z = get_separation_score(activations_vector, binary_feature)
    fidelity = get_fidelity(binary_feature, activations_vector)
    return sep_score_Z - fidelity

def get_highest_lower_bound_sep_score(binary_feature, sparse_encodings, activation_threshold = 0.5):
    ind_max_sep_score = np.argmax([get_lower_bound_sep_score(binary_feature, sparse_encodings[:, i], activation_threshold) for i in range(sparse_encodings.shape[1])])
    return ind_max_sep_score, get_lower_bound_sep_score(binary_feature, sparse_encodings[:, ind_max_sep_score], activation_threshold)
