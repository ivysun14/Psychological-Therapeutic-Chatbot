'''
This file use metadata.csv and processed json file to create label matrix.
Output of this file: 'label_matrix.txt' (naive label matrix by deleting labels that appear
                      less than 2% of the time), 'label_matrix_with_none.txt' (label_matrix
                      with an additional indicator column in the front representing no label
                      presented for this instance), 'label_matrix_merged.txt' (label matrix
                      with similar symptoms merged into one category), 'label_matrix_merged
                      _with_none.txt' (merged label matrix with first column representing
                      no label)
'''


import json
import numpy as np
import pandas as pd

file = open('./processed/meta_cleaned.json')
data = json.load(file)
file.close()
meta = pd.read_csv('./data/publication_metadata_cleaned.csv')
ids = meta['Entity_ID']

# create symptom dictionary
symptom_dict = {}
j = 0
for i in ids:
    symptom = data['%d'%i]['Symptoms']
    try:
        symptoms = symptom.split('; ')
    except:
        continue
    for s in symptoms:
        if s.startswith('[^]'):
            s = s[4:]
        if s not in symptom_dict.keys():
            symptom_dict[s] = j
            j += 1

with open('symptom_dictionary', 'w') as f:
    json.dump(symptom_dict, f)

# create naive label matrix
label_matrix = np.zeros((len(data), len(symptom_dict)))
m = 0
for i in ids:
    symptom = data['%d'%i]['Symptoms']
    try:
        symptoms = symptom.split('; ')
    except:
        m += 1
        continue
    for s in symptoms:
        if s.startswith('[^]'):
            s = s[4:]
        index = symptom_dict[s]
        label_matrix[m, index] += 1
    m += 1

# remove labels that appear less than 2% of all labels
n = len(data)
threshold = n * 0.02
label_matrix_filter = np.zeros((n, 1))
symp_list = list(symptom_dict)
labels = []
for label in range(len(symptom_dict)):
    symp = symp_list[label]
    if np.sum(label_matrix[:, label]) > threshold:
        print("number of instances with " + symp + ' ' + str(np.sum(label_matrix[:, label])))
        label_matrix_filter = np.append(label_matrix_filter, label_matrix[:, label].reshape(n, 1), axis=1)
        labels.append(symp)
label_matrix_filter = label_matrix_filter[:, 1:]
print(label_matrix_filter.shape)

# this writes the full label matrix after filtering to label_matrix.txt
np.savetxt('label_matrix.txt', label_matrix_filter)


# create a label matrix with first column == true if no symptom presents
label_matrix_none = np.zeros((label_matrix_filter.shape[0], label_matrix_filter.shape[1] + 1))
for i in range(label_matrix_filter.shape[0]):
    if np.sum(label_matrix_filter[i, :]) == 0:
        label_matrix_none[i, 0] = 1
label_matrix_none[:, 1:] = label_matrix_filter
print('number of instance with no label: ' + str(np.sum(label_matrix_none[:, 0])))

# this writes the full label matrix (with no symptom indicator) after filtering to txt
print(label_matrix_none.shape)
np.savetxt('label_matrix_with_none.txt', label_matrix_none)


# create a symptom dictionary that merges similar symptoms to one category
symptom_categories = {
    "anxiety": [1, 23],
    "depression": [3],
    "low_self_esteem": [5],
    "anger": [12, 19, 71],
    "sadness": [32],
    "frustration": [41, 77],
    "emotional_distress": [0, 2, 9, 16, 22, 35, 41, 78],
    "fatigue": [7, 26],
    "physical_issue": [8, 15, 24, 28, 29, 30, 36, 39, 40, 58, 62, 63, 70, 76, 84, 87],
    "sleep_issue": [4, 20, 27, 31, 52],
    "suicidal_thoughts": [18, 21, 51, 60, 65],
    "cognitive_issue": [17, 33, 38, 54, 59, 61, 66, 69, 92],
    "mania": [6, 48],
    "hallucination": [10, 11, 25, 67],
    "indifferent": [41, 68, 82, 85],
    "shame": [74, 79]
}

# manually merge similar symptoms and create a merged label matrix
label_matrix_merge = np.zeros((n, 1))
for category, indices in symptom_categories.items():
    new_column = np.logical_or.reduce(label_matrix[:, indices], axis=1)
    label_matrix_merge = np.column_stack((label_matrix_merge, new_column.astype(int)))
    print(category + ' ' + str(np.sum(new_column)))
label_matrix_merge = label_matrix_merge[:, 1:]

# this writes the merged label matrix to txt
print(label_matrix_merge.shape)
np.savetxt('label_matrix_merge.txt', label_matrix_merge)

# check the number of instance without any label after merging similar labels
# this should be lower than the number of instance without label in naive label matrix
# create a merged label matrix with first column == true if no symptom presents
label_matrix_merged_none = np.zeros((label_matrix_merge.shape[0], label_matrix_merge.shape[1] + 1))
for i in range(label_matrix_merge.shape[0]):
    if np.sum(label_matrix_merge[i, :]) == 0:
        label_matrix_merged_none[i, 0] = 1
label_matrix_merged_none[:, 1:] = label_matrix_merge
print('number of instance with no label in merged matrix: ' + str(np.sum(label_matrix_merged_none[:, 0])))

# this writes the merged label matrix with no symptom
print(label_matrix_merged_none.shape)
np.savetxt('label_matrix_merge_with_none.txt', label_matrix_merged_none)
