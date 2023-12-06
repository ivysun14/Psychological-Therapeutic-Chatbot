'''
Created by Junwei (Ivy) Sun

This file contains python utility functions that are helpful in visualizing
model performance and model outcomes. It contains several graphing functions
that can be called.
'''

target_symptom = "anxiety"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import json

# Load your pre-trained SVM model and dictionary
model = joblib.load('results/linear/svm_model_linear.joblib')
# Define the percentage of the training data to use for each iteration
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

target_symptom = "anxiety"

f = open('symptom_dictionary_merged_with_none')
symptoms = json.load(f)
f.close()

def extract_symptom_labels(y, symptom_dict, symptom=target_symptom):
    """
    Inputs:
    @y: the label matrix with labels {0, 1}
    @symptom_dict: the symptom dictionary
    @symptom: the class / symptom we wish to build one-to-rest SVM classifier on, default "Anxiety"

    @return: a float representing the performance score of the classifier
    """
    
    index = symptom_dict[symptom]
    extract_label = y[:, index]
    extract_label[extract_label == 0] = -1
    return extract_label

# Load in labels
X_train = np.loadtxt("training_examples.txt")
y_train = np.loadtxt("training_labels.txt")
X_test = np.loadtxt("testing_examples.txt")
y_test = np.loadtxt("testing_labels.txt")

y_train = extract_symptom_labels(y_train, symptoms, symptom=target_symptom)
y_test = extract_symptom_labels(y_test, symptoms, symptom=target_symptom)


def plot_metric(metric):
    # Initialize lists to store training and test scores
    train_scores = []
    test_scores = []
        
    # Iterate over the training sizes
    for size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Split the data into a subset for training
        X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=size, random_state=42, stratify=y_train)
        
        # Train the model on the subset
        model.fit(X_subset, y_subset)
        
        # Make predictions on the training set
        y_train_pred = model.predict(X_subset)
        
        # Make predictions on the test set (for evaluation)
        y_test_pred = model.predict(X_test)
        
        
        # compute performance
        if metric == "accuracy":  # fraction of correctly classified samples
          train_score = metrics.accuracy_score(y_subset, y_train_pred)
          test_score = metrics.accuracy_score(y_test, y_test_pred)
        elif metric == "f1_score":  # harmonic mean of the precision and recall
          train_score = metrics.f1_score(y_subset, y_train_pred)
          test_score = metrics.f1_score(y_test, y_test_pred)
        elif metric == "auroc":
          train_score = metrics.roc_auc_score(y_subset, y_train_pred)
          test_score = metrics.roc_auc_score(y_test, y_test_pred)
        elif metric == "precision":  # precision aka. of all we predicted to have the symptom, what fraction actually has the symptom
          train_score = metrics.precision_score(y_subset, y_train_pred)
          test_score = metrics.precision_score(y_test, y_test_pred)
        else:
          mcm = metrics.confusion_matrix(y_subset, y_train_pred)
          mcm_test = metrics.confusion_matrix(y_test, y_test_pred)
          tn, fp, fn, tp = mcm.ravel()
          tnt, fpt, fnt, tpt = mcm_test.revel()
          if metric == "sensitivity":  # recall aka. of all who actually have the symptom, what fraction did we correctly predict as having it
            train_score = tp / (tp + fn)
            test_score = tpt/(tpt + fnt)
          if metric == "specificity":  # of all who don't have the symptom, what fraction did we correctly predict as not having it
            train_score = tn / (tn + fp)
            test_score = tnt/(tnt + fpt)        
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, label='Training score', color='blue', marker='o')
    plt.plot(train_sizes, test_scores, label='Test score', color='green', marker='o')
    
    plt.title('Learning Curve')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel(metric)
    plt.legend(loc='best')
    plt.grid()
    
    plt.savefig(f'svm_{metric}.png',bbox_inches='tight')
    
metric_list = ['accuracy', 'f1_score', "auroc", "precision", "sensitivity", "specificity"]

for m in metric_list:
    print("Generating plot for ", m)
    plot_metric(m)

