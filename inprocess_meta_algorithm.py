##### PACKAGES


# Load all necessary packages
import sys
from collections import OrderedDict
import os
from aif360.datasets import StandardDataset
from tqdm import tqdm

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import MetaFairClassifier
from compute_metrics import compute_metrics
from data_prep import data_prep
from compute_profit import compute_profit

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from percentage_change import print_percentage_change

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

np.random.seed(12345)
sys.path.append("../")


def process(data):
    dataset_orig_test, dataset_orig_train, dataset_orig_valid, privileged_groups, unprivileged_groups = data_prep(data)

    biased_model = MetaFairClassifier(tau=0, sensitive_attr="AGE", type="fdr").fit(dataset_orig_train)
    dataset_bias_test = biased_model.predict(dataset_orig_test)
    classified_metric_bias_test = ClassificationMetric(dataset_orig_test, dataset_bias_test,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
    display("#### Algorithm without debiasing")
    print("Test set: Classification accuracy = {:.3f}".format(classified_metric_bias_test.accuracy()))
    TPR = classified_metric_bias_test.true_positive_rate()
    TNR = classified_metric_bias_test.true_negative_rate()
    bal_acc_bias_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = {:.3f}".format(bal_acc_bias_test))
    print("Test set: Disparate impact = {:.3f}".format(classified_metric_bias_test.disparate_impact()))
    fdr = classified_metric_bias_test.false_discovery_rate_ratio()
    fdr = min(fdr, 1 / fdr)
    print("Test set: False discovery rate ratio = {:.3f}".format(fdr))

    metric_test_bef = compute_metrics(dataset_orig_test, dataset_bias_test,
                                      unprivileged_groups, privileged_groups,
                                      disp=True)
    class_preds_transf = dataset_bias_test.labels  # Predicted class labels
    targets_transf = dataset_orig_test.labels  # True class labels
    amounts_transf = []
    if data == 'taiwan':
        amounts_transf = dataset_orig_test.features[:, 14]

    if data == 'pkdd':
        amounts_transf = dataset_orig_test.features[:, 13]

    if data == 'gmsc':
        amounts_transf = dataset_orig_test.features[:, 66]
    if data == 'german':
        amounts_transf = dataset_orig_test.features[:, 6]
    # Compute profit for transformed testing data
    profit_result_org = compute_profit(class_preds=class_preds_transf,
                                       targets=targets_transf,
                                       amounts=amounts_transf)
    # Print or use profit_result_transf as needed
    print("Profit:", profit_result_org["profit"])
    print("Profit Per Loan:", profit_result_org["profitPerLoan"])
    print("Profit Per EUR:", profit_result_org["profitPerEUR"])
    org_results = {
        'metrics': metric_test_bef,
        'profit': profit_result_org
    }

    debiased_model = MetaFairClassifier(tau=0.7, sensitive_attr="AGE", type="fdr").fit(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

    metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test,
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups)

    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = {:.3f}".format(
        metric_dataset_debiasing_test.mean_difference()))


    classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                            dataset_debiasing_test,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
    display("#### Algorithm with debiasing")
    print("Test set: Classification accuracy = {:.3f}".format(classified_metric_debiasing_test.accuracy()))
    TPR = classified_metric_debiasing_test.true_positive_rate()
    TNR = classified_metric_debiasing_test.true_negative_rate()
    bal_acc_debiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = {:.3f}".format(bal_acc_debiasing_test))
    print("Test set: Disparate impact = {:.3f}".format(classified_metric_debiasing_test.disparate_impact()))
    fdr = classified_metric_debiasing_test.false_discovery_rate_ratio()
    fdr = min(fdr, 1 / fdr)
    print("Test set: False discovery rate ratio = {:.3f}".format(fdr))

    metric_test_after = compute_metrics(dataset_orig_test, dataset_debiasing_test,
                                        unprivileged_groups, privileged_groups,
                                        disp=True)

    amounts_transf = []
    if data == 'taiwan':
        amounts_transf = dataset_orig_test.features[:, 14]

    if data == 'pkdd':
        amounts_transf = dataset_orig_test.features[:, 13]
    if data == 'gmsc':
        amounts_transf = dataset_orig_test.features[:, 66]
    if data == 'german':
        amounts_transf = dataset_orig_test.features[:, 6]
    class_preds_transf = dataset_debiasing_test.labels  # Predicted class labels
    targets_transf = dataset_orig_test.labels  # True class labels
    # Compute profit for transformed testing data
    profit_result_transf = compute_profit(class_preds=class_preds_transf,
                                          targets=targets_transf,
                                          amounts=amounts_transf)
    # Print or use profit_result_transf as needed
    print("Profit:", profit_result_transf["profit"])
    print("Profit Per Loan:", profit_result_transf["profitPerLoan"])
    print("Profit Per EUR:", profit_result_transf["profitPerEUR"])

    tranf_results = {
        'metrics': metric_test_after,
        'profit': profit_result_transf
    }
    print_percentage_change(tranf_results, org_results)
    accuracies, statistical_rates = [], []
    s_attr = "AGE"

    all_tau = np.linspace(0, 0.9, 10)
    for tau in tqdm(all_tau):
        debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=s_attr, type='sr')
        debiased_model.fit(dataset_orig_train)

        dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
        metric = ClassificationMetric(dataset_orig_test, dataset_debiasing_test,
                                      unprivileged_groups=[{s_attr: 0}],
                                      privileged_groups=[{s_attr: 1}])

        accuracies.append(metric.accuracy())
        sr = metric.disparate_impact()
        statistical_rates.append(min(sr, 1 / sr))

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax1.plot(all_tau, accuracies, color='r')
    ax1.set_title('Accuracy and $\gamma_{sr}$ vs Tau', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Input Tau', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy', color='r', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(all_tau, statistical_rates, color='b')
    ax2.set_ylabel('$\gamma_{sr}$', color='b', fontsize=16, fontweight='bold')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    fig.savefig('/Users/srikanthkyatham/PycharmProjects/practice/images/' + data + '_inproc_tau_1.png')



def redirect_output(filename):
    with open(filename, 'w'):
        pass  # Use pass to do nothing; this ensures the file is cleared if it exists

        # Open the file in 'a' mode, which appends output to the end of the file
    sys.stdout = open(filename, 'a')


if __name__ == '__main__':
    data_list = ['gmsc', 'pkdd', 'taiwan',"german"]
    for data in data_list:
        redirect_output(
            "/Users/srikanthkyatham/PycharmProjects/practice/output/" + data + '_inprocess_meta_classifier_output.txt')
        process(data)
        sys.stdout = sys.__stdout__
