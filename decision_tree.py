'''
This file use word stem and dictionary feature to tune and fit a decision tree classifier.
The feature importance score of the top several word stems are saved as '{label}.png'.
'''

import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# define some global variables, the labels we want to check
# and the evaluation metrics
label_names = ['frustration']

metric_list = ["accuracy", "f1_score", "auroc",
                "precision", "sensitivity", "specificity"]

# load the data, symptom dictionary and word stem dictionary
X_training = np.loadtxt("training_examples.txt")
y_training = np.loadtxt("training_labels.txt")
X_testing = np.loadtxt("testing_examples.txt")
y_testing = np.loadtxt("testing_labels.txt")

f = open('symptom_dictionary_merged_with_none')
symptoms = json.load(f)
f.close()

file = open('stem_words_dictionary.json')
word_dic = json.load(file)
stem_words = list(word_dic.keys())


def performance(y_true, y_pred, metric="accuracy"):
    """
    Taken from SVM script, defines the evaluation metrics used in this project

    Inputs:
    @y_true: true labels of each example, of shape (n)
    @y_pred: predicted labels of each example, of shape (n)
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
        y_true: testing label matrix, of shape (n)
        y_pred: predicted label matrix from testing data, of shape (n)
        metric_list: a list of strings specifying evaluation metrics

    Returns: a pandas dataframe with specified metrics for each label
    '''
    scores = np.zeros((1, len(metric_list)))
    for j in range(len(metric_list)):
        scores[0][j] = performance(y_true, y_pred, metric_list[j])

    score = pd.DataFrame(scores, columns=metric_list)
    return score


def tune_decision_tree(X_train, y_train, X_test, y_test, dic, label='anxiety'):
    '''
    Args:
        X_train: training matrix, of shape (num_examples, num_features)
        y_train: label vector for the specific label passed, of shape (num_examples)
        X_test: testing matrix, of shape (num_examples, num_features)
        y_test: label vector for the testing examples, of shape (num_examples)
        dic: dictionary containing word stem mapping to indices
        label: the label to train classifier on
    Returns:
        scores: a pandas dataframe with specified evaluation summary of the classifier
                on the testing data
        best_dt: the optimal decision tree classifier according to grid search
    '''

    index = dic[label]
    y_train = y_train[:, index]
    y_test = y_test[:, index]

    dt = DecisionTreeClassifier(random_state=0)

    # create the parameter grid based on the results of random search
    random_grid = {
        'max_depth': [2, 3, 5, 10],
        #'max_depth': [2],
        'min_samples_leaf': [5, 10, 20, 50, 100],
        #'min_samples_leaf': [5],
        'min_samples_split': [2, 5, 10],
        #'min_samples_split': [2],
        'splitter': ["best", "random"],
        #'splitter': ['best']
    }

    # instantiate the grid search model
    grid_search = GridSearchCV(estimator=dt,
                               param_grid=random_grid,
                               cv=5, n_jobs=-1, verbose=1, scoring = 'f1')
    grid_search.fit(X_train, y_train)

    # print the best hyperparameters
    print("Best hyperparameters:", grid_search.best_params_)

    # get the best-tuned decision tree model
    best_dt = grid_search.best_estimator_

    # evaluate the tuned model on the test set
    y_pred = best_dt.predict(X_test)
    scores = compute_metrics(y_test, y_pred, metric_list)
    print(scores)

    return scores, best_dt

def feature_importance(label='anxiety'):
    '''
    This function calls tune_decision_tree to find the optimal classifier for
     the label, print out the most important feature scores, and plot feature
     importance (default plot 10 most important features)
    Args:
        label: the label to train classifier on
    '''

    # find the optimal decision tree
    scores, dt = tune_decision_tree(X_training, y_training, X_testing, y_testing, symptoms, label)
    # find the highest feature importance score
    feature_importance = dt.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    print('first 100 importance score')
    print(feature_importance[sorted_idx[:50]])
    print('feature indices')
    print(sorted_idx[:50])
    # retrieve the word stem that has the highest importance, excluding the dictionary features
    feature_names = []
    dic_features = []
    for i in range(sorted_idx.shape[0]):
        if sorted_idx[i] < 30770:
            feature_names.append(stem_words[list(word_dic.values()).index(sorted_idx[i])])
        else:
            dic_feature = sorted_idx[i] - 30770
            dic_features.append(i)
            print(i)
            print(dic_feature)
    # np.delete(sorted_idx, 3)
    # plot impurity based importance for the tuned model
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), feature_importance[sorted_idx[:10]], align="center")
    plt.yticks(range(10), feature_names[:10])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature importance in decision tree classifier for " + label)
    plt.savefig(f"{label}.png")


# fit a decision tree for each label
for label in label_names:
    feature_importance(label)
