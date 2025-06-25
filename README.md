# Machine Learning Projects

This repository contains various machine learning regression models implemented in Python. Each sub-directory represents a different regression algorithm, complete with its dataset, trained model, and a Jupyter Notebook demonstrating the implementation.

## Project Structure

Each regression model is organized into its own directory with the following structure:

```
<model_name>/
├── dataset/
│   └── <dataset_file>.<ext>
├── model/
│   └── model.pkl
└── notebook/
    └── <model_name>-1.ipynb
```

## Implemented Models

- **Decision Tree Regression**: Implementation of Decision Tree for regression tasks.
- **ElasticNet Regression**: Implementation of ElasticNet regularization for linear regression.
- **Gradient Boosting Regression**: Implementation of Gradient Boosting for regression tasks.
- **K-Nearest Neighbors (KNN) Regression**: Implementation of KNN for regression tasks.
- **Lasso Regression**: Implementation of Lasso regularization for linear regression.
- **LightGBM Regression**: Implementation of LightGBM for regression tasks.
- **Linear Regression**: Basic implementation of Linear Regression.
- **Logistic Regression**: Implementation of Logistic Regression (though typically used for classification, it might be used for regression in some contexts here).
- **Multiple Regression**: Implementation of Multiple Linear Regression.
- **Naive Bayes Classification**: (Note: This is a classification algorithm, included for completeness if part of a broader ML repository).
- **Polynomial Regression**: Implementation of Polynomial Regression.
- **Random Forest Regression**: Implementation of Random Forest for regression tasks.
- **Ridge Regression**: Implementation of Ridge regularization for linear regression.
- **Support Vector Machine (SVM) Regression**: Implementation of SVM for regression tasks.
- **XGBoost Regression**: Implementation of XGBoost for regression tasks.

## How to Use

To explore any of the models:

1. Navigate to the specific model's directory (e.g., `linear_regression`).
2. Open the Jupyter Notebook located in the `notebook/` directory (e.g., `linear-regression-1.ipynb`).
3. Follow the steps in the notebook to understand data loading, preprocessing, model training, and evaluation.

## Dependencies

Each notebook specifies its dependencies, but common libraries used across these projects include:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `lightgbm` (for LightGBM Regression)
- `xgboost` (for XGBoost Regression)

You can install these using pip:

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm xgboost
```

## Data Sources

Datasets are typically sourced from Kaggle and are included in the `dataset/` subdirectory of each model.