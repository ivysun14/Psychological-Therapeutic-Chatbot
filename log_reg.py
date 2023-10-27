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
from sklearn.model_selection import train_test_split

label_dir = "label_matrix_with_none.txt"
data_dir = ["dic_features/NRC_feature_matrix.csv", "dic_features/MOESM.csv"]
def process_data(data_dir, label_dir):

    label = pd.read_csv(label_dir, sep =" ", header=None)
    
    filtered_label = pd.DataFrame()
    
    #Filter out symptoms that has less than 20
    """
    for col in label.columns:
        if sum(label[col]) > 20:
            filtered_label[col] = label[col]
    """
    filtered_label = label
            
    data = pd.read_csv(data_dir[0])
    data2 = pd.read_csv(data_dir[1])
    
    data['concreteness_mean'] = data2['mean']
    
    data.drop("Unnamed: 0", axis=1, inplace=True)
    
    data_fit = data[data.columns[10:]]
    
    dropped_filtered_label = filtered_label.drop(index=[1429, 1710, 1711])
    dropped_filtered_label.reset_index(drop=True, inplace=True)
    
    ###### Drop NA values
    mis_list = list(data_fit['concreteness_mean'].isna())
    
    drop_list = []
    
    for i in range(len(mis_list)):
        if mis_list[i]:
            drop_list.append(i)
    
    
    dropped_label = dropped_filtered_label.drop(index = drop_list)
    dropped_data = data_fit.drop(index = drop_list)
    return dropped_data, dropped_label

dropped_data, dropped_label = process_data(data_dir, label_dir)
x_train, x_test, y_train, y_test = train_test_split(dropped_data, dropped_label, test_size = 0.1)
mod_list = []

for lab in dropped_label.columns:
    clf = LogisticRegression().fit(x_train, y_train[lab])
    mod_list.append(clf)
    
pred = pd.DataFrame()

for i in range(len(mod_list)):
    pred[i] = mod_list[i].predict(x_test)
    
mis = np.matrix(y_test) - np.matrix(pred) 
mis = abs(mis)

error = np.mean(mis)

    




