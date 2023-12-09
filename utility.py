'''
Created by Junwei (Ivy) Sun

This file contains python utility functions that are helpful in visualizing
model performance and model outcomes. It contains several graphing functions
that can be called.
'''

from itertools import cycle
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import joblib
import json

'''
# Load your pre-trained SVM model and dictionary
model = joblib.load('results/linear/svm_model_linear.joblib')
# Define the percentage of the training data to use for each iteration
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
'''

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

# Load full original label matrix
label_matrix = np.loadtxt("label_matrix_merge_with_none.txt")

'''
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
'''

def binary_histogram(label_matrix, symptom_labels):
    # Count the number of 1s and 0s for each symptom
    counts_1 = np.sum(label_matrix, axis=0)
    counts_0 = label_matrix.shape[0] - counts_1
    # Set up bar positions
    positions = np.arange(len(symptom_labels))

    # Plot the histogram
    plt.bar(positions, counts_1, label='1s', color='blue', alpha=0.7)
    plt.bar(positions, counts_0, bottom=counts_1, label='0s', color='orange', alpha=0.7)

    # Add labels and title
    plt.xlabel('Psychological Symptoms')
    plt.ylabel('Frequency')
    plt.title('Psychological symptom ditribution')
    # Set x-axis ticks and labels
    plt.xticks(positions, symptom_labels, rotation=45, ha='right')
    # Adjust layout to prevent x-axis label cutoff
    plt.tight_layout()
    # Add legend
    plt.legend()

    plt.savefig('label_distribution.png')


def plot_multiclass_roc(y_true, y_score, class_names):
    """
    Plot ROC curves for a multi-class classification problem.

    Parameters:
    - y_true: True labels in binary format (0 or 1) for each class.
    - y_score: Predicted scores for each class.
    - class_names: List of class names.

    Returns:
    - None (displays the plot).
    """
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.savefig('svm_auroc.png')


# ===========================================================================================
# Assume have three classifiers clf_anxiety, clf_anger, clf_depression, and X_test, y_test for testing data
class_names = ['anxiety', 'anger', 'depression','frustration']

# load model and make predictions/probabilities belonging to each class
clf_anxiety = joblib.load('results_anxiety/linear/svm_model_linear.joblib')
clf_anger = joblib.load('results_anger/linear/svm_model_linear.joblib')
clf_depression = joblib.load('results_depression/linear/svm_model_linear.joblib')
clf_frustration = joblib.load('results_frustration/linear/svm_model_linear.joblib')

y_pred_anxiety = clf_anxiety.decision_function(X_test)
y_pred_anger = clf_anger.decision_function(X_test)
y_pred_depression = clf_depression.decision_function(X_test)
y_pred_frustration = clf_frustration.decision_function(X_test)
y_pred_combined = np.column_stack((y_pred_anxiety, y_pred_anger, y_pred_depression, y_pred_frustration))
print(y_pred_combined.shape)

# combine true 0/1 labels for each class
y_anxiety = extract_symptom_labels(y_test, symptoms, symptom="anxiety")
y_anger = extract_symptom_labels(y_test, symptoms, symptom="anger")
y_depression = extract_symptom_labels(y_test, symptoms, symptom="depression")
y_frustration = extract_symptom_labels(y_test, symptoms, symptom="frustration")
y_test_combined = np.column_stack((y_anxiety, y_anger, y_depression, y_frustration))
print(y_test_combined.shape)

# Plot ROC curves
plot_multiclass_roc(y_test_combined, y_pred_combined, class_names)

# ===========================================================================================
# Assuming label_matrix with shape (num_samples, num_labels)
# and symptom_labels is a list of strings representing psychological symptoms
symptom_labels = ["none","anxiety","depression","low_self_esteem","anger","sadness",
                  "frustration","emotional_distress","fatigue","physical_issue","sleep_issue",
                  "suicidal_thoughts","cognitive_issue","mania","hallucination","indifferent","shame"]
binary_histogram(label_matrix, symptom_labels)
