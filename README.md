# Psychological-Therapeutic-Chatbot

##Update as of Dec. 2, 2023

### Conda Environment Setting:

To run this project, set up a virtual environment with python version and package versions specified in **environment.yml** file using either conda or miniconda. This is to ensure result reproducibility. Add packages and their versions as needed into this file.

### File Descriptions:

**data/:**
- transcript/: original transcript data from Counseling and Psychotherapy Transcripts, Volume I and Volumn II.
- Publication metadata csv of Volumn I and Volumn II combined.
- volumn1_preprocess.py: Kept only threrapy sessions from Volumn I data and cleaned book transcript entries.

**data_preprocessing_scripts/:**
- see 'Pipeline description below.

**dic_features/:**
- NRC dictionary
- Concreteness dictionary

**processed/:**
- Contains processed metadata.json files for Volumn I and Volumn II combined.

### Pipeline description:

- Preprocess data (organize.py) <br >
	Given metadata (csv) and transcripts (txt), output a json (meta.json) that extracts useful metadata and process the text into client and patient.
- Naive word frequency matrix (feature_matrix.py) <br >
	Given the processed data (meta.json), construct a raw and a normalized naive word frequency matrix for all therapy sessions with each row representing a session and each column representing a unique word from all sessions. The normalized matrix is used to establish base line performance and is normalized by row (i.e frequency count / total count of a session)
- Feature engineering (calc_NRC.py) <br >
	Given the processed data (meta.json), calculate the 10 dimentions based on NRC dictionary, concreteness score, and lengh of speech, output to dic_features/NRC_feature_matrix.csv and dic_features/MOESM.csv.
- Extract stem words (process_word.py) <br >
	Given the processed data (meta.json), use stemming word to produce word feature matrix.
- Process symptom labels and create label matrix (label_matrix.py) <br >
	Given the processed data (meta.json) and metadata (csv), create four label matrices for different use (see script).
- Create 80/20 train test data split from feature and label matrices (train_test_split.py) <br >
	Given the feature matrix (feature_matrix_total.txt) and label matrix (label_matrix_merged_with_none.txt), split 80% of data as training set (train + dev) and 20% data as testing set. The resulting train test split matrices as well as training and testing data indices are stored in folder train_test_data/.

### Models:

- Logistic regression using engineered features (log_reg.py)
- Linear-kernel and RBF-kernel SVM (SVM_1.py, SVM_2.py)

### How to run SVM files:

- Make sure the below files are in the same directories as SVM_1.py and SVM_2.py files:
	- training_examples.txt
	- training_labels.txt
 	- testing_examples.txt
  	- testing_labels.txt
  	- symptom_dictionary_merged_with_none
- Nothing need to be changed in the actual code, just change the variables as needed in each of the SVM_1.py and SVM_2.py files at the top in the 'VARIABLES' section.
- First run SVM_1.py for hyperparameter tunning, then run SVM_2.py using the selected hyperparameters to train and test on test data.
- All results will be outputted in the result folder in the current working directory. Make sure to rename this folder to something like ${symptom}_SVM_results before running a second label.
- The two .joblib files in the output folder represent the final fitted model (saved so can go back to the models later when debugging). These are too large to be uploaded to GitHub so better to save in drive.
- utility.py is for diagnostic and plotting functions and is currently empty. Add functions if needed.