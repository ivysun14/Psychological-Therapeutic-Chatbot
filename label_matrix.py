import json
import numpy as np
import pandas as pd

file = open('./processed/meta_combined.json')
data = json.load(file)
file.close()
meta = pd.read_csv('./data/publication_metadata_combined.csv')
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

# remove three invalid instances
invalid = [3450, 3731, 3732]
label_matrix_filter = np.delete(label_matrix_filter, invalid, 0)

# this writes the full label matrix after filtering to label_matrix.txt
np.savetxt('label_matrix.txt', label_matrix_filter)

# create a label matrix with first column == true if no symptom presents
label_matrix_none = np.zeros((label_matrix_filter.shape[0], label_matrix_filter.shape[1] + 1))
for i in range(label_matrix_filter.shape[0]):
    if np.sum(label_matrix_filter[i, :]) == 0:
        label_matrix_none[i, 0] = 1
label_matrix_none[:, 1:] = label_matrix_filter
print(np.sum(label_matrix_none[:, 0]))
# this writes the full label matrix (with no symptom indicator) after filtering to txt
np.savetxt('label_matrix_with_none.txt', label_matrix_none)

# manually merge similar symptoms and create a merged label matrix
symptom_categories = {
    "anxiety": [2, 24],
    "depression": [4],
    "low_self_esteem": [6],
    "anger": [7, 20, 72],
    "sadness": [33],
    "frustration": [82, 42],
    "emotional_distress": [0, 1, 3, 11, 17, 23, 36, 42, 83],
    "fatigue": [9, 27],
    "physical_issue": [10, 16, 25, 29, 30, 31, 37, 40, 41, 58, 62, 63, 70, 71, 80, 88, 91],
    "sleep_issue": [5, 21, 28, 32, 52],
    "suicidal_thoughts": [19, 22, 51, 60, 65],
    "cognitive_issue": [18, 34, 39, 54, 59, 61, 66, 69, 96],
    "mania": [8, 48],
    "hallucination": [12, 13, 26, 67],
    "indifferent": [42, 68, 87, 89],
    "shame": [73, 84]
}

label_matrix_merge = np.zeros((n, 1))
for category, indices in symptom_categories.items():
    new_column = np.logical_or.reduce(label_matrix[:, indices], axis=1)
    label_matrix_merge = np.column_stack((label_matrix_merge, new_column.astype(int)))
    print(category + ' ' + str(np.sum(new_column)))
label_matrix_merge = label_matrix_merge[:, 1:]
label_matrix_merge = np.delete(label_matrix_merge, invalid, 0)
print(label_matrix_merge.shape)
np.savetxt('label_matrix_merge.txt', label_matrix_merge)

none = np.zeros((label_matrix_merge.shape[0], label_matrix_merge.shape[1] + 1))
for i in range(label_matrix_merge.shape[0]):
    if np.sum(label_matrix_merge[i, :]) == 0:
        none[i, 0] = 1
none[:, 1:] = label_matrix_merge
print(np.sum(none[:, 0]))

