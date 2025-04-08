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


note_ids = pd.read_csv("../../translated_notes_cache.csv")['note_id']

commit_hash = sys.argv[1]
commit_link = "https://github.com/shariarvg/clinicalnotes_shift/commit/"+commit_hash
V = sys.argv[2]
attacks = [ProbabilisticLabelBasedAdditionAttack(None, 'death_in_30_days', 1, '(Note: Patient dies within 30 days)', 0.3), ProbabilisticLabelBasedAdditionAttack(None, 'death_in_30_days', 1, '(Note: Patient dies within 30 days)', 0.7), ProbabilisticLabelBasedAdditionAttack(None, 'admission_in_30_days', 1, '(Note: Patient readmitted within 30 days)', 0.3), ProbabilisticLabelBasedAdditionAttack(None, 'admission_in_30_days', 1, '(Note: Patient readmitted within 30 days)', 0.3)]

#attacks = [ProbabilisticTranslateNoteAttack(None, 0.7),ProbabilisticShortenAttack(None, 35), ProbabilisticRemoveIssuesAttack(0.7), ProbabilisticOnlyIssuesAttack(0.7), ProbabilisticRemoveSentenceContainingAttack(None, ["mg ", "mcg "], 0.7), ProbabilisticRemoveSentenceContainingAttack(None, ["hx ", "history "], 0.7), ProbabilisticRemoveSentenceContainingAttack(None, ["passed", "expired"], 0.7)]
#attacks = [ProbabilisticTranslateNoteAttack(None, 0.3), ProbabilisticTranslateNoteAttack(None, 0.7),ProbabilisticShortenAttack(None, 10), ProbabilisticShortenAttack(None, 35), ProbabilisticSwitchToKeywordAttack(None, 'not', 0.1), ProbabilisticSwitchToKeywordAttack(None, 'not', 0.5), ProbabilisticKeywordDropAttack(None, 'the', 0.3), ProbabilisticKeywordDropAttack(None, 'not', 0.8)]
#attacks += [ProbabilisticReplaceWordWithWordAttack(None, "Mr.", "Ms.", 0.2), ProbabilisticReplaceWordWithWordAttack(None, "Mr.", "Ms.", 0.8), ProbabilisticReplaceWordWithWordAttack(None, "Ms.", "Mr.", 0.2), ProbabilisticReplaceWordWithWordAttack(None, "Ms.", "Mr.", 0.8), ProbabilisticRemoveSentenceContainingAttack(None, ["hx ", "history "], 0.2), ProbabilisticRemoveSentenceContainingAttack(None, ["hx ", "history "], 0.8), ProbabilisticRemoveSentenceContainingAttack(None, ["Dr. "], 0.2), ProbabilisticRemoveSentenceContainingAttack(None, ["Dr. "], 0.8), ProbabilisticRemoveSentenceContainingAttack(None, ["mg ", "mcg "], 0.2), ProbabilisticRemoveSentenceContainingAttack(None, ["mg ", "mcg "], 0.8), ProbabilisticSwitchToKeywordAttack(None, 'not', 0.1), ProbabilisticSwitchToKeywordAttack(None, 'not', 0.5)]

featurizers = [Transformer("UFNLP/gatortron-base", "mean", 100, 'right'), \
                  Transformer("UFNLP/gatortron-base", "mean", 100, 'left')]

'''
                  Transformer("../../fine_tuned_gatortron_V2", "first", 100, 'right'),\
                  Transformer("../../fine_tuned_gatortron_V3", "first", 100, 'right'),\
                  BOW(),\
                  TransformerWithDimReduce("UFNLP/gatortron-base", "mean", 100, 50, False, 'right'),\
                  #TransformerWithDimReduce("UFNLP/gatortron-base", "mean", 100, 50, True, 'right'),\
                  TransformerWithDimReduce("../../fine_tuned_gatortron_V2", "mean", 100, 50, False, 'right')]#,\
                  #TransformerWithDimReduce("UFNLP/gatortron-base", "mean", 100, 50, True, 'right')]
'''
note_metrics = [SSP(), SSP(sae = None), MMDMetric(), MMDMetric(True), NegMauveMetric()]

endog = 'death_in_30_days'
ep = MIMICEndpoint()
training_source = MIMICSource(ep, "get_mixture", ["Z515", "Z66", "N170"], [10, 10, 10], [0.4, 0.4, 0.2])
testing_source = MIMICSource(ep, "get_mixture", ["Z515", "Z66", "N170"], [10, 10, 10], [0.4, 0.4, 0.2])
N_experiments_per_combo = 20
N_train = 100
N_test = 100
loss_fn = neg_roc_auc_loss_fn
untrained_model_type = RandomForestClassifier
untrained_model_params = {'max_depth': 5}
save_name = f"../../all_attacks_V{V}"

def get_experiment(note_featurizer, attack):
    experiment = AttackExperiment(training_source, testing_source, N_train, N_test, attack, note_metrics, loss_fn, untrained_model_type(**untrained_model_params), note_featurizer, endog)
    return experiment

def perform_experiment(experiment, verbose = False, test_attacked = False):
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

def perform_experiments(experiment,N_trials = 10, test_attacked = False):
    
    metric_diffs = np.zeros((N_trials, len(experiment.note_metrics)))
    loss_diffs = np.zeros(N_trials)
    metrics_unattacked = np.zeros((N_trials, len(experiment.note_metrics)))
    metrics_attacked = np.zeros((N_trials, len(experiment.note_metrics)))
    losses= np.zeros((N_trials, 2))
    for nt in range(N_trials):
        m1, m2, l1, l2 = perform_experiment(experiment, test_attacked = test_attacked)
        metrics_unattacked[nt] = m1
        metrics_attacked[nt] = m2
        losses[nt] = [l1,l2]
    return metrics_unattacked, metrics_attacked, losses

def main():
    
    data_out = []
    
    for attack in attacks:
        for featurizer in featurizers:
            start = time.time()
            experiment = get_experiment(featurizer, attack)
            mu, ma, l = perform_experiments(experiment, N_trials = N_experiments_per_combo)
            data_out.append(np.hstack([mu, ma, l]))
            end = time.time()
            print(f"Experiment time: {end-start} seconds")
            print(f"Average featurization time: {featurizer.get_avg_runtime()}")
            
    np.save(save_name + ".npy", data_out)
    with open(save_name + ".txt", 'w') as f:
        f.write(commit_link + "\n")
        f.write(experiment.print_avg_metric_times() + "\n")
        f.write('all_attacks_script.py')
    
main()