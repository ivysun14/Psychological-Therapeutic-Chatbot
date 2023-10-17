# Psychological-Therapeutic-Chatbot

##The goal of version 0: Map conversation into mental disorder issue

### Data set to use
 - Patient conversation: Label is symptom, input is patient sentences.
 - LIWC (to be obtained): Label for individual words

### Proposed steps:
- Preprocess data:
- Convert to standard Unicode/ASCII
- Remove non-English
- Remove doctor language and [couldn’t identify] language, cut into the dictionary of words
- Remove stop words?

### Analysis: see https://www.nature.com/articles/s41746-022-00589-7#data-availability 
- Single word: Count word frequency in LIWC library
- N-gram language model using shallow neural networks
- BERT/transformer based language model using sentence level input
Classification
_ Using indicators created from analysis, classify into different groups using different methods

### Training a Multilabel model (if not train separate model for each label):
ML-KNN: https://www.sciencedirect.com/science/article/pii/S0031320307000027#sec4
	Library: http://scikit.ml/api/skmultilearn.adapt.mlknn.html#

### Evaluating a Multilabel model:
- A summary of different metrics with code:
- https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
- Summary paper on Multilabel classification:
- https://arxiv.org/abs/1609.00288

