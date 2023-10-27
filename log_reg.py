#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:35:32 2023

@author: Siqi Ma
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

label = pd.read_csv("label_matrix_with_none.txt", sep =" ", header=None)

filtered_label = pd.DataFrame()

#Filter out symptoms that has less than 20
for col in label.columns:
    if sum(label[col]) > 20:
        filtered_label[col] = label[col]
        
data = pd.read_csv("dic_features/NRC_feature_matrix.csv")
data2 = pd.read_csv("dic_features/MOESM.csv")

data['concreteness_mean'] = data2['mean']

data.drop("Unnamed: 0", axis=1, inplace=True)

data_fit = data[data.columns[10:]]

dropped_filtered_label = filtered_label.drop(index=[1429, 1710, 1711])

###### Drop NA values
mis_list = np.array(data_fit['concreteness_mean'].isna())

drop_list = list(mis_list.astype(int))

dropped_filtered_label.drop(index = drop_list)
data_fit.drop(index = drop_list)

#clf = LogisticRegression().fit(data)



