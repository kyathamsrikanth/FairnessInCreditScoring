##### PACKAGES
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys
from aif360.metrics import ClassificationMetric
import numpy as np
from IPython.display import Markdown, display
from data_prep import data_prep
from compute_metrics import compute_metrics
from compute_profit import compute_profit
import os

from percentage_change import print_percentage_change

np.random.seed(1)


def process(data):
    dataset_orig_test, dataset_orig_train, dataset_orig_valid, privileged_groups, unprivileged_groups = data_prep(data)
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)
    dataset_transf_train = RW.transform(dataset_orig_train)
    ### Testing
    assert np.abs(dataset_transf_train.instance_weights.sum() - dataset_orig_train.instance_weights.sum()) < 1e-6
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)
    display("#### Transformed training dataset")
    print(
        "-- statistical parity difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

    ### Testing
    assert np.abs(metric_transf_train.mean_difference()) < 1e-6

    best_class_thresh, class_thresh_arr, dataset_orig_test_pred, pos_ind,orig_results = without_rw_orig_analysis(dataset_orig_test,
                                                                                                    dataset_orig_train,
                                                                                                    dataset_orig_valid,
                                                                                                    privileged_groups,
                                                                                                    unprivileged_groups,data)

    transf_results = with_rw_transf_analysis(best_class_thresh, class_thresh_arr, dataset_orig_test, dataset_orig_test_pred,
                            dataset_transf_train, pos_ind, privileged_groups, unprivileged_groups,data)
    print_percentage_change(transf_results, orig_results)

    #plt.show()


def with_rw_transf_analysis(best_class_thresh, class_thresh_arr, dataset_orig_test, dataset_orig_test_pred,
                            dataset_transf_train, pos_ind, privileged_groups, unprivileged_groups,data):
    scale_transf = StandardScaler()
    X_train = scale_transf.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train,
             sample_weight=dataset_transf_train.instance_weights)
    y_train_pred = lmod.predict(X_train)
    dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_transf.fit_transform(dataset_transf_test_pred.features)
    y_test = dataset_transf_test_pred.labels
    dataset_transf_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
    display("#### Predictions from transformed testing data")
    bal_acc_arr_transf = []
    disp_imp_arr_transf = []
    avg_odds_diff_arr_transf = []
    print("Classification threshold used = %.4f" % best_class_thresh)
    metrics_transf = []
    for thresh in tqdm(class_thresh_arr):

        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False

        fav_inds = dataset_transf_test_pred.scores > thresh
        dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
        dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label

        metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                                          unprivileged_groups, privileged_groups,
                                          disp=disp)
        if thresh == best_class_thresh:
            metrics_transf = metric_test_aft
        bal_acc_arr_transf.append(metric_test_aft["Balanced accuracy"])
        avg_odds_diff_arr_transf.append(metric_test_aft["Average odds difference"])
        disp_imp_arr_transf.append(metric_test_aft["Disparate impact"])
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr_transf)
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, np.abs(1.0 - np.array(disp_imp_arr_transf)), color='r')
    ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
    ax2.axvline(best_class_thresh, color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    fig.savefig('/Users/srikanthkyatham/PycharmProjects/practice/images/' + data + '_preproc_transf_1.png')
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr_transf)
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, avg_odds_diff_arr_transf, color='r')
    ax2.set_ylabel('avg. odds diff.', color='r', fontsize=16, fontweight='bold')
    ax2.axvline(best_class_thresh, color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    fig.savefig('/Users/srikanthkyatham/PycharmProjects/practice/images/' + data + '_preproc_transf_2.png')
    # plt.show()
    # Extract necessary inputs for compute_profit function
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
    return {
        'metrics': metrics_transf,
        'profit': profit_result_transf
    }


def without_rw_orig_analysis(dataset_orig_test, dataset_orig_train, dataset_orig_valid, privileged_groups,
                             unprivileged_groups,data):
    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train,
             sample_weight=dataset_orig_train.instance_weights)
    y_train_pred = lmod.predict(X_train)
    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    dataset_orig_train_pred = dataset_orig_train.copy()
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
    print("Best balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)
    display("#### Predictions from original testing data")
    bal_acc_arr_orig = []
    disp_imp_arr_orig = []
    avg_odds_diff_arr_orig = []
    print("Classification threshold used = %.4f" % best_class_thresh)
    metrics_orig = []
    for thresh in tqdm(class_thresh_arr):

        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False

        fav_inds = dataset_orig_test_pred.scores > thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                                          unprivileged_groups, privileged_groups,
                                          disp=disp)
        if thresh == best_class_thresh:
            metrics_orig = metric_test_bef
        bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
        avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
        disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr_orig)
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, np.abs(1.0 - np.array(disp_imp_arr_orig)), color='r')
    ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
    ax2.axvline(best_class_thresh, color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)

    fig.savefig('/Users/srikanthkyatham/PycharmProjects/practice/images/' + data + '_preproc_org_1.png')

    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot(class_thresh_arr, bal_acc_arr_orig)
    ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2 = ax1.twinx()
    ax2.plot(class_thresh_arr, avg_odds_diff_arr_orig, color='r')
    ax2.set_ylabel('avg. odds diff.', color='r', fontsize=16, fontweight='bold')
    ax2.axvline(best_class_thresh, color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    fig.savefig('/Users/srikanthkyatham/PycharmProjects/practice/images/' + data + '_preproc_org_2.png')
    # Extract necessary inputs for compute_profit function
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
    profit_result_transf = compute_profit(class_preds=class_preds_transf,
                                          targets=targets_transf,
                                          amounts=amounts_transf)
    # Print or use profit_result_transf as needed
    print("Profit:", profit_result_transf["profit"])
    print("Profit Per Loan:", profit_result_transf["profitPerLoan"])
    print("Profit Per EUR:", profit_result_transf["profitPerEUR"])
    return best_class_thresh, class_thresh_arr, dataset_orig_test_pred, pos_ind , {
        'metrics': metrics_orig,
        'profit': profit_result_transf
    }


def redirect_output(filename):
    with open(filename, 'w'):
        pass  # Use pass to do nothing; this ensures the file is cleared if it exists

        # Open the file in 'a' mode, which appends output to the end of the file
    sys.stdout = open(filename, 'a')
     # Redirect stderr to the same file


if __name__ == '__main__':
    data_list = ['pkdd','taiwan',"gmsc","german"]
    for data in data_list:
        redirect_output("/Users/srikanthkyatham/PycharmProjects/practice/output/" + data + '_preprocess_RW_output.txt')
        process(data)
        sys.stdout = sys.__stdout__
