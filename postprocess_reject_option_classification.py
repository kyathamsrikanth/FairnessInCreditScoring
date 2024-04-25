##### PACKAGES

import sys

from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.metrics import ClassificationMetric
import numpy as np
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from compute_metrics import compute_metrics
from data_prep import data_prep
from compute_profit import compute_profit
import os

from percentage_change import print_percentage_change

np.random.seed(1)

sys.path.append("../")

#random seed for calibrated equal odds prediction
randseed = 12345679
# Metric used (should be one of allowed_metrics)
metric_name = "Statistical parity difference"

# Upper and lower bound on the fairness metric used
metric_ub = 0.05
metric_lb = -0.05

# random seed for calibrated equal odds prediction
np.random.seed(1)

# Verify metric name
allowed_metrics = ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]
if metric_name not in allowed_metrics:
    raise ValueError("Metric name should be one of allowed metrics")

def process(data):

    dataset_orig_test, dataset_orig_train, dataset_orig_valid, privileged_groups, unprivileged_groups = data_prep(data)

    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                            dataset_orig_valid_pred,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)

        ba_arr[idx] = 0.5 * (classified_metric_orig_valid.true_positive_rate() \
                             + classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    print("Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no fairness constraints) = %.4f" % best_class_thresh)
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                     num_class_thresh=100, num_ROC_margin=50,
                                     metric_name=metric_name,
                                     metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
    print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
    print("Optimal ROC margin = %.4f" % ROC.ROC_margin)

    # Metrics for the test set
    fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    display("#### Validation set")
    display("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy")

    metric_valid_bef = compute_metrics(dataset_orig_valid, dataset_orig_valid_pred,
                                       unprivileged_groups, privileged_groups)

    # Transform the validation set
    dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)

    display("#### Validation set")
    display("##### Transformed predictions - With fairness constraints")
    metric_valid_aft = compute_metrics(dataset_orig_valid, dataset_transf_valid_pred,
                                       unprivileged_groups, privileged_groups)

    # Testing: Check if the metric optimized has not become worse
    assert np.abs(metric_valid_aft[metric_name]) <= np.abs(metric_valid_bef[metric_name])

    # Metrics for the test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    display("#### Test set")
    display("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy")

    metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                                      unprivileged_groups, privileged_groups)
    class_preds_transf = dataset_orig_test_pred.labels  # Predicted class labels
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

    # Metrics for the transformed test set
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)

    display("#### Test set")
    display("##### Transformed predictions - With fairness constraints")
    metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                                      unprivileged_groups, privileged_groups)
    amounts_transf = []
    if data == 'taiwan':
        amounts_transf = dataset_orig_test.features[:, 14]

    if data == 'pkdd':
        amounts_transf = dataset_orig_test.features[:, 13]
    if data == 'gmsc':
        amounts_transf = dataset_orig_test.features[:, 66]
    if data == 'german':
        amounts_transf = dataset_orig_test.features[:, 6]
    class_preds_transf = dataset_transf_test_pred.labels  # Predicted class labels
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
        'metrics': metric_test_aft,
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
        redirect_output("/Users/srikanthkyatham/PycharmProjects/practice/output/" + data + '_postprocess_reject_classification_output.txt')
        process(data)
        sys.stdout = sys.__stdout__