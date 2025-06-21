# Random Forest Regression on Boston Housing Dataset

This repository contains a practical example of using a Random Forest Regressor to predict house prices based on the Boston Housing dataset.

![](image-1.png)

## Features

- Data preprocessing with categorical encoding (OneHotEncoder)
- Use of sklearn Pipeline for streamlined workflow
- Model training, prediction, and evaluation (Mean Squared Error)
- Feature importance extraction and visualization

## Dataset

The Boston Housing dataset is loaded directly from Kaggle using `kagglehub`. It contains housing data with numerical and categorical features.

## Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn kagglehub
