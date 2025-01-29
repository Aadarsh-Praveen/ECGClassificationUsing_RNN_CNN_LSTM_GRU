# ECGClassificationUsing_RNN_CNN_LSTM_GRU


## Overview
This project implements deep learning models using TensorFlow and Keras to analyze and predict outcomes based on a given dataset. It involves data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)


## Installation
Ensure you have the required dependencies installed before running the project. You can install them using:

```bash
pip install mlxtend tensorflow pandas numpy seaborn matplotlib plotly scikit-learn
```

## Dataset
The dataset used in this project is loaded and preprocessed before training the neural network models. It undergoes feature scaling using `MinMaxScaler` and is split into training and testing sets using `train_test_split`.

Link for the dataset - https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data

## Exploratory Data Analysis (EDA)
The project includes a detailed analysis of the dataset, utilizing:
- **Seaborn** and **Matplotlib** for visualizations
- **Plotly** for interactive analysis
- Correlation heatmaps to understand feature relationships

## Model Architecture
The following deep learning models are implemented:
- **Simple RNN**
- **GRU (Gated Recurrent Unit)**
- **LSTM (Long Short-Term Memory)**
- **Attention-based models**

Each model is structured using TensorFlow/Keras with layers such as:
- Dense
- Dropout
- LSTM/GRU/SimpleRNN
- Attention Mechanism

Optimization is performed using the Adam optimizer, and performance metrics are evaluated using a confusion matrix and classification report.

## Training and Evaluation
The model training includes:
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Model checkpointing for saving the best weights

Performance is evaluated based on accuracy, precision, recall, and F1-score.

## Results
The evaluation section provides:
- Accuracy metrics
- Confusion matrix visualization using `mlxtend.plot_confusion_matrix`
- Comparative analysis of different model performances
