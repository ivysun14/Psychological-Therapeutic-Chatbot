#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This generates the topics.
"""

import sys
print(sys.prefix)
import json
import tomotopy as tp
import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


num_clusters = int(sys.argv[1])

input_path = "processed/meta.json"

mdl = tp.LDAModel(k=num_clusters)
punctuations = list(string.punctuation)
stop_words = set(stopwords.words('english'))

def preprocess(line):
    tokens = word_tokenize(line)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in punctuations]
    #tokens = [word for word in tokens if word not in word_list]
    return (" ").join(tokens)

with open(input_path, "r") as f:
    data = json.load(f)
    for d in data:
        for lines in data[d]["Rev_Text"]:
            lines = preprocess(lines)
            mdl.add_doc(lines.strip().split())

print("Training model...")
for i in range(0, 100, 10):
    mdl.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

for k in range(mdl.k):
    print('Ttop 10 words of topic #{}'.format(k))
    print(mdl.get_topic_words(k, top_n=10))

print("Summary wildfire..")


mdl.summary()


sys.stdout = open(f"lda{year}q{quarter}.txt", "w")


mdl.summary()


sys.stdout.close()


