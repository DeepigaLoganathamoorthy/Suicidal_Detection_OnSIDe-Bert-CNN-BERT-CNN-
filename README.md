# Suicidal Detection OnSIDe-Bert-CNN-BERT-CNN-

This repository contains the code and resources for a project focused on detecting suicidal ideation in text using a hybrid deep learning model, OnSIDe-Bert-CNN. This model combines BERT (Bidirectional Encoder Representations from Transformers) for contextual embedding and Convolutional Neural Networks (CNNs) for feature extraction and classification.

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Sentiment Analysis](#sentiment-analysis)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## About the Project

This project aims to develop an effective and robust model for detecting suicidal ideation in text data. The model utilizes a hybrid architecture, OnSIDe-Bert-CNN, which leverages the strengths of both BERT and CNNs to achieve high accuracy in sentiment classification. This project is intended for research and educational purposes, specifically in the field of AI Applications.

## Dataset

The dataset used in this project is `suicide_detection.csv`, which contains text data labeled as either "suicide" or "non-suicide." The dataset is located in the `data/` directory.

## Dependencies

To run this project, you will need the following Python libraries:

- pandas
- numpy
- spacy
- unidecode
- contractions
- re
- wordninja
- collections
- pkg_resources
- spellchecker
- symspellpy
- matplotlib
- seaborn
- nltk
- empath
- vaderSentiment
- transformers
- torch
- scikit-learn
- tensorflow
- keras

You can install these dependencies using pip:


## Installation
1.Clone the repository:
```
git clone [https://github.com/DeepigaLoganathamoorthy/Suicidal_Detection_OnSIDe-Bert-CNN-BERT-CNN-.git](https://www.google.com/search q=https://github.com/DeepigaLoganathamoorthy/Suicidal_Detection_OnSIDe-Bert-CNN-BERT-CNN-.git)
cd Suicidal_Detection_OnSIDe-Bert-CNN-BERT-CNN-
```

2. Install the required dependencies (see Dependencies).
3. Ensure the dataset suicide_detection.csv is placed in the data/ folder.

## Usage
1. Preprocessing: Run the preprocessing script to clean and prepare the data:
Bash
# The cleaning and preprocessing code is located at the top of the python file.
The cleaned dataset is saved as suicide_final_cleaned.csv.

2. Exploratory Data Analysis (EDA): Run the EDA script to visualize and analyze the data:
Bash
# EDA code is included in the python file.

3. Sentiment Analysis: Run the sentiment analysis scripts for Empath and VADER analysis:
Bash
# Sentiment analysis code is included in the python file.

4. Model Training: Run the model training scripts for BERT, CNN, and OnSIDe-Bert-CNN models:
Bash
# Model training code for BERT, CNN, and OnSIDe-Bert-CNN is included in the python file.

5. Evaluation: The evaluation metrics and confusion matrices are displayed after each model training.

## Model Architecture
![image](https://github.com/user-attachments/assets/d3b730d5-a909-4c7a-9f2e-6ac1d56ba683)

1. BERT Model: Uses the bert-base-uncased pre-trained model for sequence classification.
2. CNN Model: A convolutional neural network model with embedding, convolutional, and dense layers.
3. OnSIDe-Bert-CNN Model: A hybrid model combining BERT embeddings with CNN layers for feature extraction and classification.

## Preprocessing
The preprocessing steps include:
a) Removing extra whitespaces
b) Removing accented characters
c) Removing URLs
d) Removing symbols and digits
e) Removing special characters
f) Fixing word lengthening
g) Expanding contractions
h) Lowercasing text
i) Removing stop words
j) Lemmatization
k) Spelling correction

## Exploratory Data Analysis (EDA)
The EDA includes:
1. Word count distribution for suicidal and non-suicidal texts
2. Top bi-grams for suicidal and non-suicidal texts
3. Text length distribution

## Sentiment Analysis
1. Empath Analysis: Analyzes the text for categories like sadness, anger, and fear.
2. VADER Analysis: Analyzes the text for positive, neutral, and negative sentiment scores.

## Model Training
1. BERT Model: Fine-tuned the pre-trained BERT model using the training data.
2. CNN Model: Trained a CNN model using word embeddings and convolutional layers.
3. OnSIDe-Bert-CNN Model: Trained a hybrid model combining BERT embeddings with CNN layers.

## Evaluation
The models are evaluated using the following metrics:
a) Accuracy
b) Precision
c) Recall
d) F1-score
e) Confusion matrices
f) Classification reports

## Results
The evaluation results are displayed after each model training. The OnSIDe-Bert-CNN model is expected to achieve high accuracy in detecting suicidal ideation.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues to improve this project.
