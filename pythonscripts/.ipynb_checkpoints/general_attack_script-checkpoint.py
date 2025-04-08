import sys, os
sys.path.append(os.path.abspath("../pythontools"))
sys.path.append(os.path.abspath("../experimenttools"))
from attack_tools import *
from tabular_fitting import TabularModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from featurization_tools import BOW, Transformer, TransformerWithDimReduce
from tabular_fitting import TabularModel
from attack_experiment_tools import AttackExperiment
from note_metric_tools import *
from loss_tools import roc_auc_loss_fn, neg_roc_auc_loss_fn
from mimic_tools import MIMICEndpoint
import time


commit_hash = sys.argv[1]
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash
V = sys.argv[2]
#attack_type = ProbabilisticShortenAttack
#params = [None]
#attack_type = TabularUnivariateFilteringAttack
#params = ['length_of_stay',None,True,15,20]
#attack_types = [ProbabilisticShortenAttack, ProbabilisticShortenAttack, TabularUnivariateFilteringAttack, ProbabilisticSwitchToKeywordAttack, ProbabilisticSwitchToKeywordAttack, ProbabilisticSwitchToKeywordAttack, ProbabilisticMultiplyFloatsAttack, ProbabilisticReplaceWithNextAttack]
#all_params = [[10], [35], ['length_of_stay', 15], ['not', 0.1], ['not',0.5],['not',0.8],[30], [0.5]]
attack_type = ProbabilisticTranslateNoteAttack
#attack_type = ProbabilisticRemoveIssuesAttack
params = [None]

def get_experiment(training_source, testing_source, N_train, N_test, note_metrics, note_featurizer, untrained_model, loss_fn, endog, attack_type, params):
    attack = attack_type(*([training_source]+params))
    experiment = AttackExperiment(training_source, testing_source, N_train, N_test, attack, note_metrics, loss_fn, untrained_model, note_featurizer, endog)
    return experiment

def perform_experiment_double_train(experiment, verbose = False, test_attacked = False):
    train_set, train_feats, _ = experiment.get_training_set_and_trained_model()
    if test_attacked:
        test_set, test_feats = experiment.sample_attacked_testing_set()
    else:
        test_set, test_feats = experiment.sample_unattacked_testing_set()
    
    loss1 = experiment.get_loss(test_set, test_feats)

    metric1 = experiment.get_note_metric(train_feats, test_feats)

    experiment.reset_featurizer()
    
    train_set, train_feats,_ = experiment.get_attacked_training_set_and_trained_model()
    test_feats = experiment.note_featurizer.transform(test_set['text'])
    loss2 = experiment.get_loss(test_set, test_feats)
    metric2 = experiment.get_note_metric(train_feats, test_feats)
    
    experiment.reset_ep()
    
    return metric1, metric2, loss1, loss2

def perform_experiment_double_train_save(experiment, verbose = False):
    train_set, train_feats, _ = experiment.get_training_set_and_trained_model()
    test_set, test_feats = experiment.sample_unattacked_testing_set()
    np.save("../../train1_feats_experiment_ssd.npy", train_feats)
    np.save("../../test1_feats_experiment_ssd.npy", test_feats)
    
    loss1 = experiment.get_loss(test_set, test_feats)

    metric1 = experiment.get_note_metric(train_feats, test_feats)

    experiment.reset_featurizer()
    
    train_set, train_feats,_ = experiment.get_attacked_training_set_and_trained_model()
    test_feats = experiment.note_featurizer.transform(test_set['text'])
    
    np.save("../../train2_feats_experiment_ssd.npy", train_feats)
    np.save("../../test2_feats_experiment_ssd.npy", test_feats)
    
    
    loss2 = experiment.get_loss(test_set, test_feats)
    metric2 = experiment.get_note_metric(train_feats, test_feats)
    
    experiment.reset_ep()
    
    return metric1, metric2, loss1, loss2

'''
def perform_experiment_double_eval(experiment, verbose = False):
    train_set, _ = experiment.get_training_set_and_trained_model()
    test_set_unattacked = experiment.sample_unattacked_testing_set()
    test_set_attacked = experiment.sample_attacked_testing_set()
    
    note_metric_diff = experiment.get_note_metric_diff(train_set, test_set_attacked, test_set_unattacked)
    loss_diff = experiment.get_loss_diff(test_set_attacked, test_set_unattacked)
    experiment.reset_ep()
    return note_metric_diff, loss_diff
'''
def perform_experiment(experiment, double_train = True, test_attacked = False):
    if double_train:
        return perform_experiment_double_train(experiment, test_attacked = test_attacked)
    return perform_experiment_double_eval(experiment)

def perform_experiments(experiment,double_train = True, N_trials = 10, test_attacked = False):
    metric_diffs = np.zeros((N_trials, len(experiment.note_metrics)))
    loss_diffs = np.zeros(N_trials)
    metrics_unattacked = []
    metrics_attacked = []
    losses_unattacked = []
    losses_attacked = []
    for nt in range(N_trials):
        m1, m2, l1, l2 = perform_experiment(experiment, double_train = double_train, test_attacked = test_attacked)
        metrics_unattacked.append(m1)
        metrics_attacked.append(m2)
        losses_unattacked.append(l1)
        losses_attacked.append(l2)
        metric_diffs[nt] = m2 - m1
        loss_diffs[nt] = l2 - l1
    return metric_diffs, loss_diffs, metrics_unattacked, metrics_attacked, losses_unattacked, losses_attacked
   
'''
MAIN
'''
def main():
    start = time.time()
    ep = MIMICEndpoint()
    note_ids = pd.read_csv("../../translated_notes_cache.csv")['note_id']
    tabular_model_constructor = LogisticRegression
    loss_fn = neg_roc_auc_loss_fn
    N_train = 500
    N_test = 500
    N_trials = 50
    untrained_model = RandomForestClassifier(max_depth=5)
    
    note_featurizer = TransformerWithDimReduce(max_length = 100)
    training_source = MIMICSource(ep, "get_notes_note_ids", note_ids)
    testing_source = MIMICSource(ep, "get_notes_note_ids", note_ids)
    note_metrics = [MMDMetric(TransformerWithDimReduce(max_length = 100))]
    endog = 'death_in_30_days'
    test_attacked = True
    
    '''
    for i, (attack, params) in enumerate(zip(attack_types, all_params)):
        experiment = get_experiment(training_source, testing_source, N_train, N_test, note_metrics, note_featurizer, untrained_model, loss_fn, endog, attack, params)
        m1, m2, l1, l2 = perform_experiment(experiment)
        out_diffs[i] = [m2 - m1, l2 - l1]
        out[i] = [m1, m2, l1, l2]
    '''
    experiment = get_experiment(training_source, testing_source, N_train, N_test, note_metrics, note_featurizer, untrained_model, loss_fn, endog, attack_type, params)
    metric_diffs, loss_diffs, metrics_unattacked, metrics_attacked, losses_unattacked, losses_attacked = perform_experiments(experiment, N_trials=N_trials, test_attacked = test_attacked)
    
        
    save_name = f"../../attack_fitting_V{V}"
    end = time.time()
    with open(save_name + ".txt", 'w') as f:
        f.write("general_attack_fitting_script.py\n")
        f.write(commit_link+'\n')
        f.write(f"{end-start} seconds")
    np.save(save_name+".npy", np.hstack([metric_diffs, loss_diffs.reshape(-1,1)]))
    np.save(save_name+"_metrics_and_losses.npy", np.hstack([metrics_unattacked, metrics_attacked, np.array(losses_unattacked).reshape(-1,1), np.array(losses_attacked).reshape(-1,1)]))
    
def single_experiment():
    
    ep = MIMICEndpoint()
    note_ids = pd.read_csv("../../translated_notes_cache.csv")['note_id']
    training_source = MIMICSource(ep, "get_notes_note_ids", note_ids)
    testing_source = MIMICSource(ep, "get_mixture", ["Z515", "Z66", "N170"], [10, 10, 10], [0.4, 0.4, 0.2])
    tabular_model_constructor = LogisticRegression
    #ssp = SSP(TransformerWithDimReduce(sparse = True))
    note_metrics = [SWstein(featurizer = Transformer(), n_it = 20)]#, NegMauveMetric(Transformer())]
    endog = 'death_in_30_days'
    loss_fn = neg_roc_auc_loss_fn
    N_train = 500
    N_test = 500
    N_trials = 10
    untrained_model = LogisticRegression()
    note_featurizer = TransformerWithDimReduce(sparse = True)
    start = time.time()
    experiment = get_experiment(training_source, testing_source, N_train, N_test, note_metrics, note_featurizer, untrained_model, loss_fn, endog, attack_type, params)
    end = time.time()
    print(perform_experiment_double_train(experiment))
    print(f"{end-start} seconds")
    
#main()
main()
    