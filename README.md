# Growthlink_task
Iris Flower Classification Project

Overview

This project builds a machine learning model to classify Iris flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica. The dataset contains measurements of sepal length, sepal width, petal length, and petal width. We use Linear Discriminant Analysis (LDA) for dimensionality reduction and Logistic Regression for classification.

Dataset

Source: Kaggle - Iris Dataset

Features:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Target: Species (Iris-setosa, Iris-versicolor, Iris-virginica)

Steps Performed

# 1. Data Preprocessing

Load the dataset using Pandas.

Dropped unnecessary columns like 'Id'.

Separated features (X) and target (y).

Standardized the data using StandardScaler to ensure all features contribute equally.



# 2. Applying Linear Discriminant Analysis (LDA)

Performed LDA transformation to reduce the dataset to two components while maximizing class separability.


# 3. Model Training

Split data into training and testing sets.

Trained a Logistic Regression model on the transformed data.

# 4. Model Evaluation

Evaluated the model’s performance (accuracy, precision, recall) — you can extend this further!

Predicted on a new flower example.


# Results

The model successfully classified the flower as Iris-virginica.

The dataset is balanced with 50 samples per species, ensuring fair training.




