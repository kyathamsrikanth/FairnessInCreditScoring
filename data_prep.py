from IPython.core.display_functions import display
from aif360.metrics import BinaryLabelDatasetMetric

from load_german import load_german
from load_gmc import load_gmsc
from load_pkdd import load_pkdd
from load_taiwan import load_taiwan


def data_prep(data):
    path = '/Users/srikanthkyatham/PycharmProjects/practice'
    data_path = path + '/data/'
    dataset_orig = []
    if data == 'taiwan':
        dataset_orig = load_taiwan(data_path + data + '.csv')

    if data == 'pkdd':
        dataset_orig = load_pkdd(data_path + data + '.csv')

    if data == 'gmsc':
        dataset_orig = load_gmsc(data_path + data + '.csv')
    if data == 'german':
        dataset_orig = load_german(data_path + data + '.csv')
    print(dataset_orig.metadata['params']['df'].shape)
    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
    # print out some labels, names, etc.
    display("#### Training Dataset shape")
    print(dataset_orig_train.features.shape)
    display("#### Favorable and unfavorable labels")
    print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
    display("#### Protected attribute names")
    print(dataset_orig_train.protected_attribute_names)
    display("#### Privileged and unprivileged protected attribute values")
    print(dataset_orig_train.privileged_protected_attributes,
          dataset_orig_train.unprivileged_protected_attributes)
    display("#### Dataset feature names")
    print(dataset_orig_train.feature_names)
    # protected attribute
    protected = 'AGE'
    privileged_groups = [{'AGE': 1}]
    unprivileged_groups = [{'AGE': 0}]
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    display("#### Original training dataset")
    print(
        "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
    metric_orig_valid = BinaryLabelDatasetMetric(dataset_orig_valid,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    display("#### Original validation dataset")
    print(
        "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_valid.mean_difference())
    metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    display("#### Original test dataset")
    print(
        "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())
    return dataset_orig_test, dataset_orig_train, dataset_orig_valid, privileged_groups, unprivileged_groups
