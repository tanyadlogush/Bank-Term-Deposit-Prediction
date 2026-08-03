# Bank Term Deposit Prediction

## Project Overview

### Business Problem

Banks regularly conduct marketing campaigns to encourage customers to open term deposits. The goal of this project is to develop a machine learning model that predicts whether a customer will subscribe to a term deposit after a marketing contact. Such a model can help the bank focus its marketing efforts on the most promising customers, improve campaign efficiency, and reduce costs.

### Project Goal

Build and compare several binary classification models to predict whether a customer will subscribe to a term deposit and select the best-performing model.

---

## Dataset

This project uses the **[Bank Marketing Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv.)** available on Kaggle, which is based on the original dataset from the UCI Machine Learning Repository.

The dataset contains information collected during direct marketing campaigns (phone calls) conducted by a Portuguese banking institution.

**Dataset characteristics:**

- **41,188** observations
- **20** features and one target variable
- **10** numerical features
- **10** categorical features
- No missing values
- Binary classification task

**Target variable:**

- **yes** – the customer subscribed to a term deposit
- **no** – the customer did not subscribe to a term deposit

The dataset is highly imbalanced:

- **no** – approximately **89%**
- **yes** – approximately **11%**

---

## Models

The following machine learning algorithms were implemented and compared:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- XGBoost
- LightGBM

A common preprocessing pipeline was used for all models to ensure a fair comparison.

---

## Evaluation

Since the dataset is highly imbalanced, **AUROC** was selected as the primary evaluation metric because it measures the model's ability to distinguish between classes independently of the classification threshold.

**F1-score** was used as a secondary metric to evaluate the balance between Precision and Recall at the selected decision threshold.

---

## Results

| Model | Validation AUROC | Validation F1 |
|-------|-----------------:|--------------:|
| Logistic Regression | 0.8015 | 0.4665 |
| k-Nearest Neighbors | 0.7912 | 0.3141 |
| Decision Tree | 0.8055 | 0.4824 |
| XGBoost | 0.8170 | 0.5301 |
| **LightGBM** | **0.8208** | **0.5081** |

The **LightGBM** model achieved the highest AUROC on the validation set and was selected as the final model.

---

## Conclusions

The final model demonstrates good overall discrimination between customers who are likely and unlikely to subscribe to a term deposit. However, the analysis indicates that its performance can be further improved.

Possible directions for future work include:

- optimizing the classification threshold;
- balancing the training data;
- engineering additional predictive features;
- performing deeper error analysis to better understand misclassified cases.

---

## Project Structure

```text
.
├── data/
├── models/
├── notebooks/
├── outputs/
├── src/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tanyadlogush/Bank-Term-Deposit-Prediction 
cd Bank-Term-Deposit-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- XGBoost
- Hyperopt
- SHAP
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Requirements

- pandas==3.0.1
- numpy==2.4.3
- scikit-learn==1.8.0
- lightgbm==4.6.0
- xgboost==3.2.0
- hyperopt==0.2.7
- matplotlib==3.10.8
- seaborn==0.13.2
- shap==0.51.0

---

## Data Source

- **Kaggle:** https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv
- Original dataset: UCI Machine Learning Repository