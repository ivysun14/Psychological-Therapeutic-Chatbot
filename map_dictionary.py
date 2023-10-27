#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:39:07 2023

@author: Siqi Ma
"""
import json
import pandas as pd
from collections import OrderedDict, defaultdict

words_dic = json.load(open("unique_words_dictionary.json"))

NRC = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep="	", header=None)

words = set(NRC[0])
emotions = set(NRC[1])


dic = {}

for word in words:
    dic[word] = {}
    print(word)
    for emo in emotions:
        print(emo)
        try:
            dic[word][emo] = list(NRC[(NRC[0] == word )& (NRC[1] == emo)][2])[0]
        except:
            dic[word][emo] = float('nan')
            
with open('NRC_dic.json', "w") as fout:
    json.dump(dic, fout, indent=4)
        

