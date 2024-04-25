import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
import numpy as np


def load_gmsc(filepath):
    '''Imports and prepares gmsc data set'''

    # read CSV
    df = pd.read_csv(filepath, sep=',', na_values=[])

    # prepare features
    df['AGE'] = df['age'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    del df['age']
    df['CREDIT_AMNT'] = df['CREDIT_AMNT']
    df['TARGET'] = df['BAD']
    del df['BAD']

    # feature lists
    XD_features = ["AGE", "CREDIT_AMNT",
                   'UnknownNumberOfDependents', 'UnknownMonthlyIncome', 'NoDependents', 'NoIncome', 'ZeroDebtRatio',
                   'UnknownIncomeDebtRatio', 'WeirdRevolvingUtilization', 'ZeroRevolvingUtilization', 'Log.Debt',
                   'RevolvingLines', 'HasRevolvingLines', 'HasRealEstateLoans', 'HasMultipleRealEstateLoans',
                   'EligibleSS',
                   'DTIOver33', 'DTIOver43', 'DisposableIncome', 'RevolvingToRealEstate',
                   'NumberOfTime30.59DaysPastDueNotWorseLarge', 'NumberOfTime30.59DaysPastDueNotWorse96',
                   'Never30.59DaysPastDueNotWorse', 'Never60.89DaysPastDueNotWorse', 'Never90DaysLate', 'IncomeDivBy10',
                   'IncomeDivBy100', 'IncomeDivBy1000', 'IncomeDivBy5000', 'Weird0999Utilization', 'FullUtilization',
                   'ExcessUtilization', 'NumberOfTime30.89DaysPastDueNotWorse', 'Never30.89DaysPastDueNotWorse',
                   'NeverPastDue',
                   'Log.RevolvingUtilizationTimesLines', 'Log.RevolvingUtilizationOfUnsecuredLines',
                   'DelinquenciesPerLine',
                   'MajorDelinquenciesPerLine', 'MinorDelinquenciesPerLine', 'DelinquenciesPerRevolvingLine',
                   'MajorDelinquenciesPerRevolvingLine', 'MinorDelinquenciesPerRevolvingLine', 'Log.DebtPerLine',
                   'Log.DebtPerRealEstateLine', 'Log.DebtPerPerson', 'RevolvingLinesPerPerson',
                   'RealEstateLoansPerPerson',
                   'YearsOfAgePerDependent', 'Log.MonthlyIncome', 'Log.IncomePerPerson', 'Log.NumberOfTimesPastDue',
                   'Log.NumberOfTimes90DaysLate', 'Log.NumberOfTime30.59DaysPastDueNotWorse',
                   'Log.NumberOfTime60.89DaysPastDueNotWorse', 'Log.Ratio90to30.59DaysLate',
                   'Log.Ratio90to60.89DaysLate',
                   'AnyOpenCreditLinesOrLoans', 'Log.NumberOfOpenCreditLinesAndLoans',
                   'Log.NumberOfOpenCreditLinesAndLoansPerPerson', 'Has.Dependents', 'Log.HouseholdSize',
                   'Log.DebtRatio',
                   'Log.UnknownIncomeDebtRatio', 'Log.UnknownIncomeDebtRatioPerPerson',
                   'Log.UnknownIncomeDebtRatioPerLine',
                   'Log.UnknownIncomeDebtRatioPerRealEstateLine', 'Log.NumberRealEstateLoansOrLines']
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features) - set(D_features))
    categorical_features = []

    # protected attribute
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
        features_to_keep=X_features + Y_features + D_features,
        metadata={'label_maps': [{1.0: 'Good', 2.0: 'Bad'}],
                  'protected_attribute_maps': [protected_attribute_map]})

    return df_standard