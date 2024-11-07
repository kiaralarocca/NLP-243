# NLP 243 HW 1
## Kiara LaRocca | UCSC | klarocca@ucsc.edu

# Relation Extraction with Multilayer Perceptron (MLP)

This repository contains code and resources for a supervised learning task of multi-class classification using a Multilayer Perceptron (MLP) model, developed as part of a homework assignment focused on relation extraction from a film-based schema. The project aims to classify relations in utterances from a dataset derived from the Freebase knowledge graph's film schema.

## Project Overview

The primary task is to build and evaluate a deep learning model for multi-class relation extraction. This project explores the performance of an MLP model trained on film-related utterances to predict core relations, such as actors, directors, or film genres.

### Features
- **Data Source**: The dataset contains utterances and their corresponding core relations derived from the Freebase film schema, split into `hw1_train.csv` and `hw1_test.csv`.
- **Model**: A simple MLP architecture implemented in PyTorch.
- **Embeddings**: Utilizes CountVectorizer for Bag-of-Words representation and experiments with GloVe embeddings.
- **Evaluation**: Performance measured using accuracy, F1-score, and confusion matrix visualizations.

### Files Included
- **`run.py`**: Main script to train and evaluate the MLP model.
- **`mlp_model.pth`**: Saved model weights from the training process.
- **`hw1_train.csv`**: Training dataset containing utterances and corresponding core relations.
- **`hw1_test.csv`**: Testing dataset with utterances for which relations need to be predicted.
- **`requirements.txt`**: List of dependencies required to set up the environment.

### Project Details

The project employs a basic MLP model with a three-layer architecture and Leaky ReLU activation. Dropout and weight-decay regularization techniques are used to prevent overfitting. The main embedding method employed is CountVectorizer, while some experiments with GloVe embeddings were also conducted.
Future improvements could include more comprehensive hyperparameter tuning, better metric tracking, and leveraging additional embeddings to enhance performance.
