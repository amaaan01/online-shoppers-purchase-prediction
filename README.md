# Online Shoppers Purchase Prediction

Predicting whether an online store visitor will complete a purchase based on their browsing behavior using a Decision Tree Classifier.

## Project Overview

This project analyzes online shopping session data and builds a machine learning model to predict whether a visitor will generate revenue (make a purchase).

The dataset contains information about user interactions with an e-commerce website, including page visits, session duration, traffic sources, visitor type, and browsing behavior.

## Business Problem

E-commerce businesses receive thousands of visitors every day, but only a small percentage complete a purchase.

The goal of this project is to identify visitors with a high likelihood of purchasing so that businesses can improve marketing strategies, personalize user experiences, and increase conversion rates.

## Dataset

- 12,330 user sessions
- 18 features
- Binary target variable: `Revenue`
- Mix of numerical and categorical features
- Imbalanced classification problem

### Target Variable

| Revenue | Meaning |
|----------|----------|
| 0 | No Purchase |
| 1 | Purchase Completed |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Dataset inspection
- Class distribution analysis
- Feature understanding
- Correlation analysis
- Visualization of important features

### 2. Data Preprocessing

- Feature selection
- Numerical and categorical feature separation
- Standardization using `StandardScaler`
- One-Hot Encoding using `OneHotEncoder`
- Pipeline creation using `ColumnTransformer`

### 3. Model Building

A Decision Tree Classifier was implemented using:

```python
DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)
```

### 4. Pipeline

```python
Pipeline([
    ("preprocess", preprocessor),
    ("model", decision_tree)
])
```

### 5. Model Evaluation

The model was evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
- Classification Report

## Repository Structure

```text
online-shoppers-purchase-prediction/
│
├── .gitignore
├── README.md
├── online_shoppers_purchase_prediction.ipynb
├── requirements.txt
└── shop_smart_ecommerce.csv
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/online-shoppers-purchase-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter lab
```

## Results

| Metric | Score |
|----------|----------|
| Accuracy | XX.XX |
| Precision | XX.XX |
| Recall | XX.XX |
| F1 Score | XX.XX |

> Replace the values above with your actual model results.

## Key Learnings

- Handling imbalanced classification problems
- Feature preprocessing with Scikit-Learn
- Building reusable ML Pipelines
- Decision Tree pruning techniques
- Model evaluation using F1 Score
- Working with real-world e-commerce data

## Future Improvements

- Hyperparameter tuning with GridSearchCV
- Random Forest Classifier
- XGBoost Classifier
- SMOTE for class balancing
- Cross-validation
- Model deployment using Flask or FastAPI

## Author

Amaan Shaikh

Machine Learning & Data Science Enthusiast
