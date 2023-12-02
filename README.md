# Psychological-Therapeutic-Chatbot

##Update as of Dec. 1

### Data set to use
 - Patient conversation Volume I and II: Label is symptom, input is patient sentences.
 - NRC dictionaryr
 - Concreteness dictionary

### Pipeline description:
- Preprocess data (organize.py)
	Given metadata (csv) and transcripts (txt), output a json (meta.json) that extracts useful metadata and process the text into client and patient.
- Feature engineering (calc_NRC.py)
	Given the processed data (meta.json), calculate the 10 dimentions based on NRC dictionary, concreteness score, and lengh of speech, output to dic_features/NRC_feature_matrix.csv and dic_features/MOESM.csv.
- Extract stem words (process_word.py)
	Given the processed data (meta.json), use stemming word to produce word feature matrix.

### Models:
- Logistic regression using engineered features (log_reg.py)
- Linear-kernel and RBF-kernel SVM
	- SVM_V0_params/: this folder contains selected hyperparameters and final evaluation metrics for SVM trained on word frequency matrix of Volumn II data, focusing only on anxiety classification.
   	- SVM_V1_params/: this folder contains hyperparameters, evaluation metrics, and prediction results for SVM trained on engineered feautures on Volumn I and II combined.

### Training a Multilabel model (if not train separate model for each label):
ML-KNN: https://www.sciencedirect.com/science/article/pii/S0031320307000027#sec4
	Library: http://scikit.ml/api/skmultilearn.adapt.mlknn.html#

### Evaluating a Multilabel model:
- A summary of different metrics with code:
- https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
- Summary paper on Multilabel classification:
- https://arxiv.org/abs/1609.00288

### How to run SVM files:
- 

