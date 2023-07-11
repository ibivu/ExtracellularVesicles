#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler

# path = '/scistor/informatica/kwy700/EV/'
Data_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/Data'
Model_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/Models'


# Import dataset
df_no_filter = pd.read_csv(Data_path + '/training/training_data_no_filter.csv', sep=',')
df_MS_filter = pd.read_csv(Data_path + '/training/training_data_MS_filter.csv', sep=',')
df_MS_iso_filter = pd.read_csv(Data_path + '/training/training_data_MS_iso_filter.csv', sep=',')


# Preprocess data

continuous = ['length', 'hydr_count', 'polar_count', 'molecular_weight', 'helix', 'turn', 'sheet',
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    'thsa_netsurfp2', 'tasa_netsurfp2', 'rhsa_netsurfp2', 'disorder', 'A_exposed', 'C_exposed', 'D_exposed',
    'E_exposed', 'F_exposed', 'G_exposed', 'H_exposed', 'I_exposed', 'K_exposed', 'L_exposed', 'M_exposed',
    'N_exposed', 'P_exposed', 'Q_exposed', 'R_exposed', 'S_exposed', 'T_exposed', 'V_exposed', 'W_exposed',
    'Y_exposed', 'Probability_solubility', 'Aggregation_propensity', 'Aromaticity', 'Instability_index',
    'Gravy', 'Isoelectric_point', 'Charge_at_7', 'Charge_at_5', 'Polar_exposed', 'Hydrophobic_exposed']


def preprocess(df):

    # define explanatory and response variables
    X = df.drop(["id", "EV"], axis=1)
    y = df["EV"]

    # undersample majority class
    undersample = RandomUnderSampler(random_state=0)
    X_balanced, y_balanced = undersample.fit_resample(X, y)

    return X_balanced, y_balanced


def split_and_scale(X_balanced, y_balanced, features_cont=continuous, scaler=RobustScaler()):

    # split 80% training and 20% test
    train_X, test_X, train_y, test_y = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=0, stratify=y_balanced)

    # robust scaling
    train_X_scaled = train_X.copy()
    test_X_scaled = test_X.copy()
    train_X_scaled[features_cont] = scaler.fit_transform(train_X[features_cont])
    test_X_scaled[features_cont] = scaler.transform(test_X[features_cont])

    print("Size of training set:", len(train_X_scaled))
    print("Size of test set:", len(test_X_scaled))

    return train_X_scaled, train_y, test_X_scaled, test_y


print("Dataset with no filtering")
X_balanced_1, y_balanced_1 = preprocess(df_no_filter)
train_X_1, train_y_1, test_X_1, test_y_1 = split_and_scale(X_balanced_1, y_balanced_1)
print("---------------------")
print("Dataset with MS filtering")
X_balanced_2, y_balanced_2 = preprocess(df_MS_filter)
train_X_2, train_y_2, test_X_2, test_y_2 = split_and_scale(X_balanced_2, y_balanced_2)
print("---------------------")
print("Dataset with MS and isolation method filtering")
X_balanced_3, y_balanced_3 = preprocess(df_MS_iso_filter)
train_X_3, train_y_3, test_X_3, test_y_3 = split_and_scale(X_balanced_3, y_balanced_3)


# ROC plot cross-validation

# adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
def roc_cross_val(name, X, y, splits=5, features_cont=continuous, scaler=RobustScaler()):
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=splits)
    classifier = RandomForestClassifier(random_state=0, n_estimators=1000, max_features=10)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for i, (train, test) in enumerate(cv.split(X, y)):

        # robust scaling
        train_X_scaled = X.loc[train].copy()
        test_X_scaled = X.loc[test].copy()
        train_X_scaled[features_cont] = scaler.fit_transform(X.loc[train][features_cont])
        test_X_scaled[features_cont] = scaler.transform(X.loc[test][features_cont])

        classifier.fit(train_X_scaled, y.loc[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            test_X_scaled,
            y.loc[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()
    # fig.savefig(os.getcwd() + '/' + name + '.png', dpi=300, bbox_inches='tight')

    return mean_tpr, mean_auc, std_auc


mean_tpr_1, mean_auc_1, std_auc_1 = roc_cross_val("no_filter", X_balanced_1, y_balanced_1)
mean_tpr_2, mean_auc_2, std_auc_2 = roc_cross_val("MS_filter", X_balanced_2, y_balanced_2)
mean_tpr_3, mean_auc_3, std_auc_3 = roc_cross_val("MS_iso_filter", X_balanced_3, y_balanced_3)


# Train the random forest

rf_1 = RandomForestClassifier(random_state=0, n_estimators=1000, max_features=10)
rf_1.fit(train_X_1, train_y_1)

rf_2 = RandomForestClassifier(random_state=0, n_estimators=1000, max_features=10)
rf_2.fit(train_X_2, train_y_2)

rf_3 = RandomForestClassifier(random_state=0, n_estimators=1000, max_features=10)
rf_3.fit(train_X_3, train_y_3)


# use only sequence-based features
seq_features = ['length', 'hydr_count', 'polar_count', 'molecular_weight', 'helix', 'turn', 'sheet', 'A', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'thsa_netsurfp2', 'tasa_netsurfp2',
                'rhsa_netsurfp2', 'disorder', 'A_exposed', 'C_exposed', 'D_exposed', 'E_exposed', 'F_exposed',
                'G_exposed', 'H_exposed', 'I_exposed', 'K_exposed', 'L_exposed', 'M_exposed', 'N_exposed', 'P_exposed',
                'Q_exposed', 'R_exposed', 'S_exposed', 'T_exposed', 'V_exposed', 'W_exposed', 'Y_exposed',
                'Probability_solubility', 'Aggregation_propensity', 'Aromaticity', 'Instability_index', 'Gravy',
                'Isoelectric_point', 'Charge_at_7', 'Charge_at_5', 'PTM_MSD', 'Phosphorylation_MSD',
                'Glycosylation_MSD', 'Ubiquitination_MSD', 'SUMOylation_MSD', 'Acetylation_MSD', 'Palmitoylation_MSD',
                'Methylation_MSD', 'coiled_coil', 'RAS_profile', 'ww_domain', 'EGF', 'RRM',
                'TMHMM', 'Polar_exposed', 'Hydrophobic_exposed']

mean_tpr_1_seq, mean_auc_1_seq, std_auc_1_seq = roc_cross_val("no_filter_seq", X_balanced_1[seq_features], y_balanced_1)
mean_tpr_2_seq, mean_auc_2_seq, std_auc_2_seq = roc_cross_val("MS_filter_seq", X_balanced_2[seq_features], y_balanced_2)
mean_tpr_3_seq, mean_auc_3_seq, std_auc_3_seq = roc_cross_val("MS_iso_filter_seq", X_balanced_3[seq_features], y_balanced_3)

train_X_1_seq = train_X_1[seq_features]
test_X_1_seq = test_X_1[seq_features]

train_X_2_seq = train_X_2[seq_features]
test_X_2_seq = test_X_2[seq_features]

train_X_3_seq = train_X_3[seq_features]
test_X_3_seq = test_X_3[seq_features]

rf_1_seq = RandomForestClassifier(random_state=0, n_estimators=1000, max_features=10)
rf_1_seq.fit(train_X_1_seq, train_y_1)

rf_2_seq = RandomForestClassifier(random_state=0, n_estimators=1000, max_features=10)
rf_2_seq.fit(train_X_2_seq, train_y_2)

rf_3_seq = RandomForestClassifier(random_state=0, n_estimators=1000, max_features=10)
rf_3_seq.fit(train_X_3_seq, train_y_3)


# AUC ROC curves

# figure settings
sns.set(style=("ticks"), font_scale=1.3)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 6))

mean_fpr = np.linspace(0, 1, 100)

ax1.plot(mean_fpr, mean_tpr_1, color="darkred", label=r"All features (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_1, std_auc_1))
ax1.plot(mean_fpr, mean_tpr_1_seq, color="palevioletred",
    label=r"Sequence-based (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_1_seq, std_auc_1_seq))
ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Unfiltered EV-annotated dataset")
ax1.legend()

ax2.plot(mean_fpr, mean_tpr_2, color="forestgreen", label=r"All features (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_2, std_auc_2))
ax2.plot(mean_fpr, mean_tpr_2_seq, color="lightgreen",
    label=r"Sequence-based (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_2_seq, std_auc_2_seq))
ax2.set(xlabel="False Positive Rate", ylabel=None, title="MS filtered EV-annotated dataset")
ax2.legend()

ax3.plot(mean_fpr, mean_tpr_3, color="darkblue", label=r"All features (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_3, std_auc_3))
ax3.plot(mean_fpr, mean_tpr_3_seq, color="cornflowerblue",
    label=r"Sequence-based (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_3_seq, std_auc_3_seq))
ax3.set(xlabel="False Positive Rate", ylabel=None, title="Isolation workflow and MS filtered \n EV-annotated discovery dataset")
ax3.legend()
fig.suptitle("ROC-AUC curves of random forest classifiers")

plt.show()
fig.savefig(os.getcwd() + '/Combined_AUC_filter_crossval.png', dpi=300, bbox_inches='tight', transparent=True)
