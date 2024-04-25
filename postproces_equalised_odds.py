
import sys

import os

import matplotlib.pyplot as plt

from aif360.metrics import ClassificationMetric
import numpy as np
from IPython.display import Markdown, display
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from tqdm import tqdm

from compute_profit import compute_profit
from percentage_change import print_percentage_change

sys.path.append("../")

from compute_metrics import compute_metrics
from data_prep import data_prep
cost_constraint = "fnr" # "fnr", "fpr", "weighted"
np.random.seed(1)

#random seed for calibrated equal odds prediction
randseed = 12345679


def process(data):
    dataset_orig_test, dataset_orig_train, dataset_orig_valid, privileged_groups, unprivileged_groups = data_prep(data)

    # Placeholder for predicted and transformed datasets
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

    # Logistic regression classifier and predictions for training data
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)

    fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

    # Prediction probs for validation and testing data
    X_valid = scale_orig.transform(dataset_orig_valid.features)
    y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]

    X_test = scale_orig.transform(dataset_orig_test.features)
    y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

    class_thresh = 0.5
    dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
    dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
    dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

    y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
    y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
    y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
    dataset_orig_train_pred.labels = y_train_pred

    y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
    y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
    y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
    dataset_orig_valid_pred.labels = y_valid_pred

    y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
    y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
    y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
    dataset_orig_test_pred.labels = y_test_pred
    cm_pred_train = ClassificationMetric(dataset_orig_train, dataset_orig_train_pred,
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    display("#### Original-Predicted training dataset")
    print("Difference in GFPR between unprivileged and privileged groups")
    print(cm_pred_train.difference(cm_pred_train.generalized_false_positive_rate))
    print("Difference in GFNR between unprivileged and privileged groups")
    print(cm_pred_train.difference(cm_pred_train.generalized_false_negative_rate))

    cm_pred_valid = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred,
                                         unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups)
    display("#### Original-Predicted validation dataset")
    print("Difference in GFPR between unprivileged and privileged groups")
    print(cm_pred_valid.difference(cm_pred_valid.generalized_false_positive_rate))
    print("Difference in GFNR between unprivileged and privileged groups")
    print(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))

    cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
    display("#### Original-Predicted testing dataset")
    print("Difference in GFPR between unprivileged and privileged groups")
    print(cm_pred_test.difference(cm_pred_test.generalized_false_positive_rate))
    print("Difference in GFNR between unprivileged and privileged groups")
    print(cm_pred_test.difference(cm_pred_test.generalized_false_negative_rate))
    metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                                      unprivileged_groups, privileged_groups,
                                      disp=True)
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

    # Learn parameters to equalize odds and apply to create a new dataset
    cpp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups,
                                         cost_constraint=cost_constraint,
                                         seed=randseed)
    cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

    dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
    dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)
    cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)
    display("#### Original-Transformed validation dataset")
    print("Difference in GFPR between unprivileged and privileged groups")
    print(cm_transf_valid.difference(cm_transf_valid.generalized_false_positive_rate))
    print("Difference in GFNR between unprivileged and privileged groups")
    print(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate))

    cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    display("#### Original-Transformed testing dataset")
    print("Difference in GFPR between unprivileged and privileged groups")
    print(cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate))
    print("Difference in GFNR between unprivileged and privileged groups")
    print(cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate))

    metric_test_after = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
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
        'metrics': metric_test_after,
        'profit': profit_result_transf
    }

    print_percentage_change(tranf_results, org_results)

    # Testing: Check if the rates for validation data has gone down
    assert np.abs(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate)) < np.abs(
        cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))
    # Thresholds
    all_thresh = np.linspace(0.01, 0.99, 25)
    display("#### Classification thresholds used for validation and parameter selection")

    bef_avg_odds_diff_test = []
    bef_avg_odds_diff_valid = []
    aft_avg_odds_diff_test = []
    aft_avg_odds_diff_valid = []
    bef_bal_acc_valid = []
    bef_bal_acc_test = []
    aft_bal_acc_valid = []
    aft_bal_acc_test = []
    for thresh in tqdm(all_thresh):
        dataset_orig_valid_pred_thresh = dataset_orig_valid_pred.copy(deepcopy=True)
        dataset_orig_test_pred_thresh = dataset_orig_test_pred.copy(deepcopy=True)
        dataset_transf_valid_pred_thresh = dataset_transf_valid_pred.copy(deepcopy=True)
        dataset_transf_test_pred_thresh = dataset_transf_test_pred.copy(deepcopy=True)

        # Labels for the datasets from scores
        y_temp = np.zeros_like(dataset_orig_valid_pred_thresh.labels)
        y_temp[dataset_orig_valid_pred_thresh.scores >= thresh] = dataset_orig_valid_pred_thresh.favorable_label
        y_temp[~(dataset_orig_valid_pred_thresh.scores >= thresh)] = dataset_orig_valid_pred_thresh.unfavorable_label
        dataset_orig_valid_pred_thresh.labels = y_temp

        y_temp = np.zeros_like(dataset_orig_test_pred_thresh.labels)
        y_temp[dataset_orig_test_pred_thresh.scores >= thresh] = dataset_orig_test_pred_thresh.favorable_label
        y_temp[~(dataset_orig_test_pred_thresh.scores >= thresh)] = dataset_orig_test_pred_thresh.unfavorable_label
        dataset_orig_test_pred_thresh.labels = y_temp

        y_temp = np.zeros_like(dataset_transf_valid_pred_thresh.labels)
        y_temp[dataset_transf_valid_pred_thresh.scores >= thresh] = dataset_transf_valid_pred_thresh.favorable_label
        y_temp[
            ~(dataset_transf_valid_pred_thresh.scores >= thresh)] = dataset_transf_valid_pred_thresh.unfavorable_label
        dataset_transf_valid_pred_thresh.labels = y_temp

        y_temp = np.zeros_like(dataset_transf_test_pred_thresh.labels)
        y_temp[dataset_transf_test_pred_thresh.scores >= thresh] = dataset_transf_test_pred_thresh.favorable_label
        y_temp[~(dataset_transf_test_pred_thresh.scores >= thresh)] = dataset_transf_test_pred_thresh.unfavorable_label
        dataset_transf_test_pred_thresh.labels = y_temp

        # Metrics for original validation data
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                            dataset_orig_valid_pred_thresh,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
        bef_avg_odds_diff_valid.append(classified_metric_orig_valid.equal_opportunity_difference())

        bef_bal_acc_valid.append(0.5 * (classified_metric_orig_valid.true_positive_rate() +
                                        classified_metric_orig_valid.true_negative_rate()))

        classified_metric_orig_test = ClassificationMetric(dataset_orig_test,
                                                           dataset_orig_test_pred_thresh,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups)
        bef_avg_odds_diff_test.append(classified_metric_orig_test.equal_opportunity_difference())
        bef_bal_acc_test.append(0.5 * (classified_metric_orig_test.true_positive_rate() +
                                       classified_metric_orig_test.true_negative_rate()))

        # Metrics for transf validing data
        classified_metric_transf_valid = ClassificationMetric(
            dataset_orig_valid,
            dataset_transf_valid_pred_thresh,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        aft_avg_odds_diff_valid.append(classified_metric_transf_valid.equal_opportunity_difference())
        aft_bal_acc_valid.append(0.5 * (classified_metric_transf_valid.true_positive_rate() +
                                        classified_metric_transf_valid.true_negative_rate()))

        # Metrics for transf validation data
        classified_metric_transf_test = ClassificationMetric(dataset_orig_test,
                                                             dataset_transf_test_pred_thresh,
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups)
        aft_avg_odds_diff_test.append(classified_metric_transf_test.equal_opportunity_difference())
        aft_bal_acc_test.append(0.5 * (classified_metric_transf_test.true_positive_rate() +
                                       classified_metric_transf_test.true_negative_rate()))

    bef_bal_acc_valid = np.array(bef_bal_acc_valid)
    bef_avg_odds_diff_valid = np.array(bef_avg_odds_diff_valid)

    aft_bal_acc_valid = np.array(aft_bal_acc_valid)
    aft_avg_odds_diff_valid = np.array(aft_avg_odds_diff_valid)

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax1.plot(all_thresh, bef_bal_acc_valid, color='b')
    ax1.plot(all_thresh, aft_bal_acc_valid, color='b', linestyle='dashed')
    ax1.set_title('Original and Postprocessed validation data', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(all_thresh, np.abs(bef_avg_odds_diff_valid), color='r')
    ax2.plot(all_thresh, np.abs(aft_avg_odds_diff_valid), color='r', linestyle='dashed')
    ax2.set_ylabel('abs(Equal opportunity diff)', color='r', fontsize=16, fontweight='bold')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    fig.legend(["Balanced Acc. - Orig.", "Balanced Acc. - Postproc.",
                "Equal opp. diff. - Orig.", "Equal opp. diff. - Postproc.", ],
               fontsize=16)
    fig.savefig('/Users/srikanthkyatham/PycharmProjects/practice/images/' + data + '_postproc_validation_1.png')

    bef_bal_acc_test = np.array(bef_bal_acc_test)
    bef_avg_odds_diff_test = np.array(bef_avg_odds_diff_test)

    aft_bal_acc_test = np.array(aft_bal_acc_test)
    aft_avg_odds_diff_test = np.array(aft_avg_odds_diff_test)

    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax1.plot(all_thresh, bef_bal_acc_test, color='b')
    ax1.plot(all_thresh, aft_bal_acc_test, color='b', linestyle='dashed')
    ax1.set_title('Original and Postprocessed testing data', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2 = ax1.twinx()
    ax2.plot(all_thresh, np.abs(bef_avg_odds_diff_test), color='r')
    ax2.plot(all_thresh, np.abs(aft_avg_odds_diff_test), color='r', linestyle='dashed')
    ax2.set_ylabel('abs(Equal opportunity diff)', color='r', fontsize=16, fontweight='bold')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    fig.legend(["Balanced Acc. - Orig.", "Balanced Acc. - Postproc.",
                "Equal opp. diff. - Orig.", "Equal opp. diff. - Postproc."],
               fontsize=16)

    fig.savefig('/Users/srikanthkyatham/PycharmProjects/practice/images/' + data + '_postproc_test_1.png')

def redirect_output(filename):
    with open(filename, 'w'):
        pass  # Use pass to do nothing; this ensures the file is cleared if it exists

        # Open the file in 'a' mode, which appends output to the end of the file
    sys.stdout = open(filename, 'a')



if __name__ == '__main__':
    data_list = ['gmsc', 'pkdd', 'taiwan','german']
    for data in data_list:
        redirect_output("/Users/srikanthkyatham/PycharmProjects/practice/output/" + data + '_postprocess_equalised_odds_output.txt')
        process(data)
        sys.stdout = sys.__stdout__