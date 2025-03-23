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

1. Data Preprocessing

Load the dataset using Pandas.

Dropped unnecessary columns like 'Id'.

Separated features (X) and target (y).

Standardized the data using StandardScaler to ensure all features contribute equally.

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Iris.csv")
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

2. Applying Linear Discriminant Analysis (LDA)

Performed LDA transformation to reduce the dataset to two components while maximizing class separability.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

3. Model Training

Split data into training and testing sets.

Trained a Logistic Regression model on the transformed data.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

4. Model Evaluation

Evaluated the model’s performance (accuracy, precision, recall) — you can extend this further!

Predicted on a new flower example.

new_flower = [[6.7, 3.0, 5.2, 2.3]]
new_flower_scaled = scaler.transform(new_flower)
new_flower_lda = lda.transform(new_flower_scaled)
predicted_species = classifier.predict(new_flower_lda)

print("Predicted species:", predicted_species[0])

Results

The model successfully classified the flower as Iris-virginica.

The dataset is balanced with 50 samples per species, ensuring fair training.

Next Steps

Add performance metrics like accuracy, precision, recall, and confusion matrix.

Visualize decision boundaries after LDA transformation.

Hyperparameter tuning for improved accuracy.

Repository Structure

📁 Iris-Classification-Project
│
├── 📄 Iris.csv
├── 📄 iris_classification.ipynb
└── 📄 README.md

Acknowledgments

Dataset: UCI Machine Learning Repository

Libraries: pandas, sklearn, numpy

Happy coding! 🚀


