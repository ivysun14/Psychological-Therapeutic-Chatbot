import json
import numpy as np
import pandas as pd

file = open('./processed/meta.json')
data = json.load(file)
file.close()
meta = pd.read_csv('./data/publication_metadata.csv')
ids = meta['Entity_ID']
symptom_dict = {}
j = 0
for i in ids:
    symptom = data['%d'%i]['Symptoms']
    try:
        symptoms = symptom.split('; ')
    except:
        continue
    for s in symptoms:
        if s not in symptom_dict.keys():
            symptom_dict[s] = j
            j += 1

with open('symptom_dictionary', 'w') as f:
    json.dump(symptom_dict, f)

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
        index = symptom_dict[s]
        label_matrix[m, index] += 1
    m += 1

np.savetxt('label_matrix.txt', label_matrix)

# create a label matrix with first column == true if no symptom presents
new_label_matrix = np.zeros((label_matrix.shape[0], label_matrix.shape[1] + 1))
for i in range(label_matrix.shape[0]):
    if np.sum(label_matrix[i, :]) == 0:
        new_label_matrix[i, 0] = 1
new_label_matrix[:, 1:] = label_matrix

np.savetxt('label_matrix_with_none.txt', new_label_matrix)