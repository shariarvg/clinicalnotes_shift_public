import sys, os
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
from attack_tools import *
from tabular_fitting import TabularModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from featurization_tools import BOW, Transformer
from tabular_fitting import TabularModel
from attack_experiment_tools import AttackExperiment
from note_metric_tools import *
from loss_tools import roc_auc_loss_fn, neg_roc_auc_loss_fn
from mimic_tools import MIMICEndpoint
import time

commit_hash = sys.argv[1]
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash
V = sys.argv[2]

def make_tabular_model(dataset, tabular_model_constructor, exogs, endog):
    tabular_model = TabularModel(dataset, tabular_model_constructor, exogs, endog)
    tabular_model.normalize_exogs()
    tabular_model.fit()
    return tabular_model

'''
for univariate filtering
'''

def get_top_predictor(tabular_model):
    return tabular_model.get_exogs_sorted_by_coefs()[0]

def get_threshold(dataset, column, q = 0.7):
    return np.quantile(dataset[column], q)

def get_char_dataset(source, N_from_source, exog):
    return source.obtain_samples(N_from_source)[exog]

def get_univariate_filtering_attack(source, N_from_source, tabular_model_constructor, exogs, endog, q):
    dataset = source.obtain_samples(N_from_source)
    #tabular_model = make_tabular_model(dataset, tabular_model_constructor, exogs, endog)
    #column = get_top_predictor(tabular_model)
    column = exogs[0]
    threshold = get_threshold(dataset, column, q)
    source.reset()
    return TabularUnivariateFilteringAttack(source, column, threshold, greaterthan = True)

def get_experiment(training_source, testing_source, N_fit_tabular, tabular_model_constructor, N_train, N_test, note_metrics, note_featurizer, untrained_model, loss_fn, exogs, endog, q):
    attack = get_univariate_filtering_attack(training_source, N_fit_tabular, tabular_model_constructor, exogs, endog, q)
    experiment = AttackExperiment(training_source, testing_source, N_train, N_test, attack, note_metrics, loss_fn, untrained_model, note_featurizer, endog)
    return experiment

def perform_experiment_double_train(experiment, verbose = False):
    s1 = time.time()
    train_set_unattacked, _ = experiment.get_training_set_and_trained_model()
    s2 = time.time()
    #print(f"{s2 - s1} seconds elapsed for getting trained set and trained model")
    test_set_unattacked = experiment.sample_unattacked_testing_set()
    s3 = time.time()
    #print(f"{s3 - s2} seconds elapsed for getting unattacked testing set")
    loss1 = experiment.get_loss(test_set_unattacked)
    s4 = time.time()
    #print(f"{s4 - s3} seconds elapsed for getting loss of unattacked testing set")
    metric1 = experiment.get_note_metric(train_set_unattacked, test_set_unattacked)
    s5 = time.time()
    #print(f"{s5 - s4} seconds elapsed for getting metrics between training and testing set")
    
    experiment.reset_ep()
    
    train_set_attacked, _ = experiment.get_attacked_training_set_and_trained_model()
    loss2 = experiment.get_loss(test_set_unattacked)
    metric2 = experiment.get_note_metric(train_set_attacked, test_set_unattacked)
    
    experiment.reset_ep()
    
    return metric2 - metric1, loss2 - loss1

def perform_experiment_double_eval(experiment, verbose = False):
    train_set, _ = experiment.get_training_set_and_trained_model()
    test_set_unattacked = experiment.sample_unattacked_testing_set()
    test_set_attacked = experiment.sample_attacked_testing_set()
    
    note_metric_diff = experiment.get_note_metric_diff(train_set, test_set_attacked, test_set_unattacked)
    loss_diff = experiment.get_loss_diff(test_set_attacked, test_set_unattacked)
    experiment.reset_ep()
    return note_metric_diff, loss_diff

def perform_experiment(experiment, double_train = True):
    if double_train:
        return perform_experiment_double_train(experiment)
    return perform_experiment_double_eval(experiment)

def perform_experiments(experiment,double_train = True, N_trials = 10):
    metric_diffs = np.zeros((N_trials, len(experiment.note_metrics)))
    loss_diffs = np.zeros(N_trials)
    for nt in range(N_trials):
        md, ld = perform_experiment(experiment, double_train = double_train)
        metric_diffs[nt] = md
        loss_diffs[nt] = ld
    return metric_diffs, loss_diffs

def perform_experiments_random_threshold(experiment, char_dataset, double_train = True, N_trials = 10):
    metric_diffs = np.zeros((N_trials, len(experiment.note_metrics)))
    loss_diffs = np.zeros(N_trials)
    for nt in range(N_trials):
        u = np.random.uniform()
        threshold = np.quantile(char_dataset, u)
        experiment.threshold = threshold
        md, ld = perform_experiment(experiment, double_train = double_train)
        metric_diffs[nt] = md
        loss_diffs[nt] = ld
    return metric_diffs, loss_diffs
   
'''
MAIN
'''
def main():
    start = time.time()
    ep = MIMICEndpoint()
    training_source = MIMICSource(ep, "get_notes_diagnosis", "Z515", 10)
    testing_source = MIMICSource(ep, "get_notes_diagnosis", "Z515", 10)
    N_fit_tabular = 500
    q = 0.3
    tabular_model_constructor = LogisticRegression
    note_metrics = [MMDMetric(Transformer())]#, NegMauveMetric(Transformer())]
    exogs = ['start_age']
    endog = 'death_in_30_days'
    loss_fn = neg_roc_auc_loss_fn
    N_train = 1000
    N_test = 100
    untrained_model = LogisticRegression()
    note_featurizer = Transformer()
    char_dataset = get_char_dataset(training_source, N_fit_tabular, "start_age")
    experiment = get_experiment(training_source, testing_source, N_fit_tabular, tabular_model_constructor, N_train, N_test, note_metrics, note_featurizer, untrained_model, loss_fn, exogs, endog, q)
    metric_diffs, loss_diffs = perform_experiments_random_threshold(experiment, char_dataset, True, 500)
    save_name = f"../../tabular_fitting_V{V}.npy"
    end = time.time()
    with open(f"../../tabular_fitting_V{V}.txt", 'w') as f:
        f.write("tabular_fitting_script.py\n")
        f.write(commit_link+'\n')
        f.write(f"{end-start} seconds")
    np.save(save_name, np.hstack([metric_diffs, loss_diffs.reshape(-1,1)]))

main()
    
    
    
    