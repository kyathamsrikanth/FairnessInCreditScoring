import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
import numpy as np

def load_pkdd(filepath):
    '''Imports and prepares pkdd data set'''

    # read CSV
    df = pd.read_csv(filepath, sep = ',', na_values = [])

    # prepare features
    df = df.sample(frac = 1).reset_index(drop = True)
    df['AGE'] = df['AGE'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    df['CREDIT_AMNT'] = df['CREDIT_AMNT']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["AGE", "CREDIT_AMNT",
                   'PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'SEX', 'MARITAL_STATUS', 'QUANT_DEPENDANTS', 'STATE_OF_BIRTH',
                   'NATIONALITY', 'RESIDENTIAL_STATE', 'FLAG_RESIDENCIAL_PHONE', 'RESIDENCE_TYPE', 'MONTHS_IN_RESIDENCE',
                   'FLAG_EMAIL', 'PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES', 'FLAG_VISA', 'FLAG_MASTERCARD',
                   'QUANT_SPECIAL_BANKING_ACCOUNTS', 'QUANT_CARS', 'COMPANY', 'PROFESSIONAL_STATE', 'FLAG_PROFESSIONAL_PHONE',
                   'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE', 'EDUCATION_LEVEL2', 'PRODUCT']
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = ['PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'SEX', 'MARITAL_STATUS', 'STATE_OF_BIRTH',
                            'NATIONALITY', 'RESIDENTIAL_STATE', 'RESIDENCE_TYPE', 'PROFESSIONAL_STATE', 'PROFESSION_CODE',
                            'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE', 'EDUCATION_LEVEL2', 'PRODUCT']

    # protected attribute
    privileged_class        = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}

    # target encoding
    status_map   = {'GOOD': 1.0, 'BAD': 2.0}
    df['TARGET'] = df['TARGET'].replace(status_map)

    # convert DF
    df_standard = StandardDataset(
        df                        = df,
        label_name                = Y_features[0],
        favorable_classes         = [1],
        protected_attribute_names = D_features,
        privileged_classes        = [privileged_class["AGE"]],
        instance_weights_name     = None,
        categorical_features      = categorical_features,
        features_to_keep          = X_features + Y_features + D_features,
        metadata                  = {'label_maps':               [{1.0: 'Good', 2.0: 'Bad'}],
                                     'protected_attribute_maps': [protected_attribute_map]})

    return df_standard