This repository contains a Python machine learning pipeline for credit card fraud detection. It uses pandas for data processing and trains an XGBoost classifier to predict binary fraud classes on highly imbalanced transaction data. The model is evaluated using ROC AUC and a confusion matrix.

✨ Key Features

Extreme Imbalance Handling: Strategically calculates the ratio of negative to positive classes to appropriately weight the XGBoost model (handling 577:1 class imbalance).

Stratified Splitting: Uses stratify=y during the Train-Test split to ensure the extreme minority class is proportionately represented in both datasets.

Robust Evaluation: Evaluates model performance beyond standard accuracy using ROC AUC and Confusion Matrices to get a true picture of precision and recall.

🛠️ Tech Stack

Language: Python 3

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn, xgboost

Environment: Jupyter Notebook / Google Colab

📊 Model Performance

The current XGBoost model achieved outstanding results on the test data:

ROC AUC Score: 0.962

Confusion Matrix: Successfully isolated fraudulent transactions while maintaining a low false-positive rate across 56,000+ test samples.

🚀 Getting Started

Prerequisites

Make sure you have Python installed, along with the required libraries. You can install the dependencies using pip:

pip install pandas numpy scikit-learn xgboost


Dataset

This notebook is designed to work with the standard Credit Card Fraud Detection dataset (commonly found on Kaggle).

Download creditcard.csv.

Place it in the root directory (or update the file path in the notebook).

Running the Notebook

Clone this repository and launch Jupyter:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
jupyter notebook "creditCard (1).ipynb"


💡 Future Enhancements

Implement Precision-Recall AUC (PR-AUC) metrics.

Feature engineering: Transform the Time feature into a cyclical Hour of Day variable.

Hyperparameter tuning using GridSearchCV or Optuna.
