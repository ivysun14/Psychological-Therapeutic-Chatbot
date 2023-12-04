'''
This file use feature_matrix_stem.txt, training_example_indices.txt, testing_example_indices.txt,
training_labels.txt, testing_labels.txt to fit a multinomial naive bayes classifier. The prediction
performance metrics are saved in 'naive_bayes_performance.txt'
'''

import numpy as np
import pandas
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# define a function that computes the evaluation metrics that we use
metric_list = ["accuracy", "f1_score", "auroc",
                "precision", "sensitivity", "specificity"]
def performance(y_true, y_pred, metric="accuracy"):
    """
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

# define a multinomial naive bayes classifiers that is able to handle multilabel and class priors
class MultiClassPriorsNB():
    def __init__(self, class_priors=None):
        self.class_priors = class_priors
        self.classifiers = None

    def fit(self, X, y):
        self.classifiers = [MultinomialNB(class_prior=[p, 1 - p]) for p in self.class_priors]
        for i, classifier in enumerate(self.classifiers):
            y_class = y[:, i]
            classifier.fit(X, y_class)
        return self

    def predict(self, X):
        return np.column_stack([classifier.predict(X) for classifier in self.classifiers])


# load in data
stem = np.loadtxt('feature_matrix_stem.txt')
train_idx = np.loadtxt('training_example_indices.txt')
test_idx = np.loadtxt('testing_example_indices.txt')
train_idx = train_idx.astype(int)
test_idx = test_idx.astype(int)
X_training = stem[train_idx, :]
X_testing = stem[test_idx, :]
y_training = np.loadtxt("training_labels.txt")
y_testing = np.loadtxt("testing_labels.txt")

# compute class priors
prior = np.sum(y_training, axis=0)
class_priors = prior / np.sum(prior)

# fit a naive bayes classifier using one versus rest for multilabel with class priors
multi_class_priors_nb = MultiClassPriorsNB(class_priors=class_priors)
multi_class_priors_nb.fit(X_training, y_training)
y_pred = multi_class_priors_nb.predict(X_testing)

# compute the evaluation metrics
scores = np.zeros((y_testing.shape[1], len(metric_list)))
for i in range(y_testing.shape[1]):
    for j in range(len(metric_list)):
        scores[i][j] = performance(y_testing[:, i], y_pred[:, i], metric_list[j])

score = pandas.DataFrame(scores, columns=metric_list)
print(score)
score.to_csv('naive_bayes_performance.txt', sep='\t', index=False)
