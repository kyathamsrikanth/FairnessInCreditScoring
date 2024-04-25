import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
import numpy as np

def load_german(filepath):
    '''Imports and prepares german data set'''

    # read CSV
    df = pd.read_csv(filepath, sep=',', na_values=[], index_col=0)

    # prepare features
    df['AGE'] = df['age'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    del df['age']
    df['CREDIT_AMNT'] = df['amount']
    del df['amount']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["account_status", "duration", "credit_history", "purpose", "CREDIT_AMNT", "savings",
                   "employment", "installment_rate", "status_gender", "guarantors", "resident_since", "property",
                   "AGE", "other_plans", "housing", "num_credits", "job",
                   "people_maintenance", "phone", "foreign"]
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['account_status', 'credit_history', 'purpose', "savings",
                            "employment", "status_gender", "guarantors", "property", "other_plans",
                            'housing', 'job', 'phone', 'foreign']

    # protected attribtue
    privileged_class = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}

    # target encoding
    status_map = {'GOOD': 1.0, 'BAD': 2.0}
    df['TARGET'] = df['TARGET'].replace(status_map)

    # convert DF
    df_standard = StandardDataset(
        df=df,
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[privileged_class["AGE"]],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features + Y_features + D_features,
        metadata={'label_maps': [{1.0: 'Good', 2.0: 'Bad'}],
                  'protected_attribute_maps': [protected_attribute_map]})

    return df_standard