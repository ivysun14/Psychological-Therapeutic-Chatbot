import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics


def performance(y_true, y_pred, metric="accuracy"):
    """
    Taken from SVM script, defines the evaluation metrics used in this project

    Inputs:
    @y_true: true labels of each example, of shape (n, )
    @y_pred: predicted labels of each example, of shape (n, )
    @metric: a string specifying one of the six performance measures.
             'accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity'

    @return: a float representing performance score
    """

    # compute performance
    if metric == "accuracy":  # fraction of correctly classified samples
        score = metrics.accuracy_score(y_true, y_pred)
    elif metric == "f1_score":  # harmonic mean of the precision and recall
        score = metrics.f1_score(y_true, y_pred)
    elif metric == "auroc":
        score = metrics.roc_auc_score(y_true, y_pred)
    elif metric == "precision":  # precision aka. of all we predicted to have the symptom, what fraction actually has the symptom
        score = metrics.precision_score(y_true, y_pred)
    else:
        mcm = metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = mcm.ravel()
        if metric == "sensitivity":  # recall aka. of all who actually have the symptom, what fraction did we correctly predict as having it
            score = tp / (tp + fn)
        if metric == "specificity":  # of all who don't have the symptom, what fraction did we correctly predict as not having it
            score = tn / (tn + fp)

    return score


def compute_metrics(y_true, y_pred, metric_list):
    '''
    Args:
        y_true: testing label matrix, of shape (n, num_labels)
        y_pred: predicted label matrix from testing data, of shape (n, num_labels
        metric_list: a list of strings specifying evaluation metrics

    Returns: a pandas dataframe with specified metrics for each label
    '''
    scores = np.zeros((y_true.shape[1], len(metric_list)))
    for i in range(y_true.shape[1]):
        for j in range(len(metric_list)):
            scores[i][j] = performance(y_true[:, i], y_pred[:, i], metric_list[j])

    score = pd.DataFrame(scores, columns=metric_list)
    return score


stem = np.loadtxt('feature_matrix_stem.txt')
train_idx = np.loadtxt('training_example_indices.txt')
test_idx = np.loadtxt('testing_example_indices.txt')
train_idx = train_idx.astype(int)
test_idx = test_idx.astype(int)
X_training = stem[train_idx, :]
X_testing = stem[test_idx, :]
y_train = np.loadtxt("training_labels.txt")
y_test = np.loadtxt("testing_labels.txt")


scaler = StandardScaler()
PCA_stem = PCA()
X_train_normalized = scaler.fit_transform(X_training)
X_stem_train_pca = PCA_stem.fit_transform(X_train_normalized)
X_test_normalized = scaler.transform(X_testing)
X_stem_test_pca = PCA_stem.transform(X_test_normalized)

explained_variance_ratio = PCA_stem.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
np.savetxt('pca_cum_var_stem.txt', cumulative_variance_ratio)
np.savetxt('pca_single_var_stem.txt', explained_variance_ratio)

base_lr = LogisticRegression()
ovr = OneVsRestClassifier(base_lr)
ovr.fit(X_training, y_train)
Y_pred_ovr = ovr.predict(X_testing)

metric_list = ["accuracy", "f1_score", "auroc",
                "precision", "sensitivity", "specificity"]
scores_original = compute_metrics(y_test, Y_pred_ovr, metric_list)
print(scores_original)

pca_lr = LogisticRegression()
pca_ovr = OneVsRestClassifier(pca_lr)
pca_ovr.fit(X_stem_train_pca[:, :2000], y_train)
Y_pred_ovr_pca = pca_ovr.predict(X_stem_test_pca[:, :2000])
scores_pca = compute_metrics(y_test, Y_pred_ovr_pca, metric_list)
print(scores_pca)