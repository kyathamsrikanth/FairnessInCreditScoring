##### PACKAGES


# Load all necessary packages
import sys

from percentage_change import print_percentage_change

sys.path.append("../")
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric


from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets import StandardDataset

from sklearn.preprocessing import StandardScaler, MaxAbsScaler

import os
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from compute_metrics import compute_metrics
from data_prep import data_prep
from compute_profit import compute_profit
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

def process(data):
    dataset_orig_test, dataset_orig_train, dataset_orig_valid, privileged_groups, unprivileged_groups = data_prep(data)

    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
    metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)
    display("#### Scaled dataset - Verify that the scaling does not affect the group label statistics")
    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
    metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())

    # Load post-processing algorithm that equalizes the odds
    # Learn parameters with debias set to False
    sess = tf.compat.v1.Session()
    plain_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                       unprivileged_groups=unprivileged_groups,
                                       scope_name='plain_classifier',
                                       debias=False,
                                       sess=sess)
    plain_model.fit(dataset_orig_train)
    # Apply the plain model to test data
    dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)
    # Metrics for the dataset from plain model (without debiasing)
    display("#### Plain model - without debiasing - dataset metrics")
    metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train,
                                                                unprivileged_groups=unprivileged_groups,
                                                                privileged_groups=privileged_groups)

    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())

    metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test,
                                                               unprivileged_groups=unprivileged_groups,
                                                               privileged_groups=privileged_groups)

    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

    display("#### Plain model - without debiasing - classification metrics")
    classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
                                                              dataset_nodebiasing_test,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())
    sess.close()
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    # Learn parameters with debias set to True
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                          unprivileged_groups=unprivileged_groups,
                                          scope_name='debiased_classifier',
                                          debias=True,
                                          sess=sess)
    debiased_model.fit(dataset_orig_train)
    # Apply the plain model to test data
    dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    # Metrics for the dataset from plain model (without debiasing)
    display("#### Plain model - without debiasing - dataset metrics")
    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

    # Metrics for the dataset from model with debiasing
    display("#### Model - with debiasing - dataset metrics")
    metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)

    print(
        "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_train.mean_difference())

    metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test,
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups)

    print(
        "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())

    display("#### Plain model - without debiasing - classification metrics")
    print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())
    metric_test_bef = compute_metrics(dataset_orig_test, dataset_nodebiasing_test,
                                      unprivileged_groups, privileged_groups,
                                      disp=True)
    class_preds_transf = dataset_nodebiasing_test.labels  # Predicted class labels
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

    display("#### Model - with debiasing - classification metrics")
    classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                            dataset_debiasing_test,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
    TPR = classified_metric_debiasing_test.true_positive_rate()
    TNR = classified_metric_debiasing_test.true_negative_rate()
    bal_acc_debiasing_test = 0.5 * (TPR + TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
    print(
        "Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())

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

    tranf_results =  {
        'metrics': metric_test_after,
        'profit': profit_result_transf
    }
    print_percentage_change(tranf_results, org_results)





def redirect_output(filename):
    with open(filename, 'w'):
        pass  # Use pass to do nothing; this ensures the file is cleared if it exists

        # Open the file in 'a' mode, which appends output to the end of the file
    sys.stdout = open(filename, 'a')


if __name__ == '__main__':
    data_list = ['gmsc', 'pkdd', 'taiwan','german']
    for data in data_list:
        redirect_output("/Users/srikanthkyatham/PycharmProjects/practice/output/" + data + '_preprocess_adversarial_output.txt')
        process(data)
        sys.stdout = sys.__stdout__